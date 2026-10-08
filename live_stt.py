"""
live_stt.py - live captions with NO extra installs. Whatever language you speak,
the transcript panel shows it in ENGLISH while you talk.

While you speak, the current sentence is sent to Gemini (the same API key the
app already uses) about every 2 seconds and the line is rewritten in place;
when you pause it is finalised. If anything fails the app falls back to
Gemini's own (late) transcript, so text never disappears.
"""
from __future__ import annotations

import collections
import io
import queue
import threading
import wave

import numpy as np

CAPTION_MODEL = "gemini-flash-latest"   # same model name the app already uses elsewhere
PARTIAL_EVERY = 1.2              # seconds between in-place updates while you speak
END_SILENCE   = 0.8              # seconds of quiet that finalises a sentence
MAX_UTTERANCE = 25.0             # force-finalise very long sentences
MIN_AUDIO     = 0.5              # don't decode anything shorter than this


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
        self._said_error = False
        self.ready = False
        self._said_hearing = False
        self._said_empty = False
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
        except Exception as e:                                # never let this thread kill the app
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
        except Exception as e:                       # one failed call must not stop the captions
            if not self._said_error:
                self._said_error = True
                self._log(f"Caption request failed: {str(e)[:120]}")
            return ""
        if not text and not self._said_empty:
            self._said_empty = True
            self._log("Live captions heard you but produced no text.")
        return text

    def _run(self) -> None:
        sr = self._sr
        preroll = collections.deque(maxlen=6)     # ~0.4 s kept from before speech starts
        utter: list[np.ndarray] = []
        speaking = False
        noise = 0.004
        quiet = 0.0
        since = 0.0
        total = 0.0

        def finish():
            nonlocal utter, speaking, quiet, since, total
            if utter:
                text = self._translate(np.concatenate(utter))
                self._on_text(text, True)
            utter, speaking, quiet, since, total = [], False, 0.0, 0.0, 0.0

        while True:
            try:
                data = self._q.get(timeout=0.4)
            except queue.Empty:
                if speaking:                      # audio stopped arriving (e.g. mic paused) -> close the sentence
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
            elif since >= PARTIAL_EVERY:
                since = 0.0
                text = self._translate(np.concatenate(utter))
                if text:
                    self._on_text(text, False)