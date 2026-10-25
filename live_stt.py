"""
live_stt.py - optional live captions of the user's speech, shown in English.

While you speak, the recent audio is sent to Gemini (the same API key the app
already uses) at a relaxed interval and the caption line is rewritten in place;
when you pause it is finalised.

Why this version is faster than the old one
-------------------------------------------
* The network call no longer runs inside the audio-consuming loop. Before, one
  slow request froze the loop, the queue filled up and audio was dropped.
  Now requests run on a single worker thread.
* At most ONE partial request is in flight at a time. If the previous partial
  has not come back yet, the next one is skipped instead of piling up.
* Partials are sent every PARTIAL_EVERY seconds (3.0s, was 1.2s) and only
  include the last PARTIAL_WINDOW seconds of audio, so requests stay small.
* A stale partial can never overwrite a line that was already finalised.

If anything fails, main.py falls back to Gemini's own (later) transcript.
"""
from __future__ import annotations

import collections
import io
import queue
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CAPTION_MODEL  = "gemini-flash-latest"
PARTIAL_EVERY  = 3.0     # seconds between in-place updates while you speak
PARTIAL_WINDOW = 10.0    # only the last N seconds are sent for a partial
END_SILENCE    = 0.8     # seconds of quiet that finalises a sentence
MAX_UTTERANCE  = 15.0    # force-finalise very long sentences
MIN_AUDIO      = 0.5     # don't decode anything shorter than this


class LiveSTT:
    def __init__(self, base_dir, on_text, logger=print, sample_rate: int = 16000, api_key=None):
        """on_text(text: str, final: bool) is called from a worker thread."""
        self._dir = base_dir
        self._on_text = on_text
        self._log = logger
        self._sr = sample_rate
        self._q: queue.Queue = queue.Queue(maxsize=600)
        self._api_key = api_key           # str or zero-arg callable returning the key
        self._client = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-stt-req")
        self._partial_busy = False
        self._last_final_id = 0
        self._said_error = False
        self._said_hearing = False
        self._said_empty = False
        self.ready = False
        threading.Thread(target=self._boot, daemon=True, name="live-stt").start()

    # called from the mic callback - must be instant
    def feed(self, data: bytes) -> None:
        if not self.ready:
            return
        try:
            self._q.put_nowait(data)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    def _boot(self) -> None:
        try:
            from google import genai
            key = self._api_key() if callable(self._api_key) else self._api_key
            self._client = genai.Client(api_key=key)
            self.ready = True
            self._log("Live captions ready (any language -> English).")
        except Exception as e:
            self._log(f"Live captions unavailable ({e}). Using Gemini's transcript instead.")
            return
        try:
            self._run()
        except Exception as e:            # never let this thread kill the app
            self.ready = False
            self._log(f"Live captions stopped ({e}). Using Gemini's transcript instead.")

    def _translate(self, audio: np.ndarray) -> str:
        if len(audio) < self._sr * MIN_AUDIO:
            return ""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._sr)
            w.writeframes(pcm)
        try:
            from google.genai import types
            resp = self._client.models.generate_content(
                model=CAPTION_MODEL,
                contents=[
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
                    "Transcribe the speech in this audio and translate it into English. "
                    "Reply with ONLY the English text. If there is no clear speech, reply with nothing.",
                ],
            )
            text = (resp.text or "").strip()
        except Exception as e:            # one failed call must not stop the captions
            if not self._said_error:
                self._said_error = True
                self._log(f"Caption request failed: {str(e)[:120]}")
            return ""
        if not text and not self._said_empty:
            self._said_empty = True
            self._log("Live captions heard you but produced no text.")
        return text

    # ------------------------------------------------------------------
    def _job(self, audio: np.ndarray, utt_id: int, final: bool) -> None:
        """Runs on the single worker thread, so requests stay in order."""
        try:
            text = self._translate(audio)
            if final:
                self._last_final_id = max(self._last_final_id, utt_id)
                self._on_text(text, True)
            elif text and utt_id > self._last_final_id:
                self._on_text(text, False)
        except Exception as e:
            self._log(f"Caption job error: {str(e)[:120]}")
        finally:
            if not final:
                self._partial_busy = False

    def _submit(self, audio: np.ndarray, utt_id: int, final: bool) -> None:
        try:
            self._pool.submit(self._job, audio, utt_id, final)
        except RuntimeError:              # pool shut down
            if not final:
                self._partial_busy = False

    def _run(self) -> None:
        sr = self._sr
        preroll = collections.deque(maxlen=6)     # ~0.4 s kept from before speech starts
        utter: list[np.ndarray] = []
        speaking = False
        noise = 0.004
        quiet = 0.0
        since = 0.0
        total = 0.0
        utt_id = 0

        def finish():
            nonlocal utter, speaking, quiet, since, total
            if utter:
                self._submit(np.concatenate(utter), utt_id, True)
            utter, speaking, quiet, since, total = [], False, 0.0, 0.0, 0.0

        while True:
            try:
                data = self._q.get(timeout=0.4)
            except queue.Empty:
                if speaking:              # audio stopped arriving (e.g. mic paused) -> close the sentence
                    finish()
                continue

            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(x) / sr
            rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
            loud = rms > max(0.012, noise * 2.5)
            if not loud and not speaking:
                noise = 0.95 * noise + 0.05 * rms  # track room noise

            if not speaking:
                preroll.append(x)
                if loud:
                    if not self._said_hearing:
                        self._said_hearing = True
                        self._log("Live captions: hearing your voice.")
                    speaking = True
                    utt_id += 1
                    utter = list(preroll)
                    preroll.clear()
                    total = sum(len(a) for a in utter) / sr
                    quiet, since = 0.0, 0.0
                continue

            utter.append(x)
            total += dur
            since += dur
            quiet = 0.0 if loud else quiet + dur

            if quiet >= END_SILENCE or total >= MAX_UTTERANCE:
                finish()
            elif since >= PARTIAL_EVERY and not self._partial_busy:
                since = 0.0
                audio = np.concatenate(utter)
                audio = audio[-int(PARTIAL_WINDOW * sr):]
                self._partial_busy = True
                self._submit(audio, utt_id, False)