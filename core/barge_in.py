"""
core/barge_in.py - detects the user talking OVER the assistant ("barge-in").

While JARVIS speaks, main.py ignores the microphone so JARVIS does not hear its
own voice. This detector watches the same mic blocks during that time and
answers one question: "is a real person talking over me right now?"

How it decides (pure local maths, no network, safe inside the audio callback):
  * It keeps a rolling history of mic loudness while JARVIS is speaking and
    uses the MEDIAN as the "echo level" (JARVIS's own voice leaking into the
    mic). The median is not moved by a short burst of your voice, so your
    voice cannot teach it that talking is normal.
  * A block counts as "voice" only if it is clearly louder than the echo level
    (margin) AND louder than an absolute floor (min_rms) AND most of its energy
    is in the speech band (300-3400 Hz), which rejects clicks, thumps and hum.
  * It triggers when `hold_blocks` of the last `window_blocks` blocks are
    voice. This is tolerant of tiny gaps between syllables (the old version
    needed an unbroken run and lost count on every quiet block).

Tuning: set debug=True. It prints rms / echo / threshold a few times per second
so you can see the real numbers on your mic today and adjust min_rms / margin.
Headphones remove the echo problem completely.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np


class BargeInDetector:
    def __init__(
        self,
        min_rms: float = 800.0,        # absolute loudness floor (int16 RMS)
        margin: float = 1.8,           # voice must be this many times louder than echo
        hold_blocks: int = 3,          # voice blocks needed ...
        window_blocks: int = 5,        # ... out of the last N blocks (~192 ms of speech)
        warmup_blocks: int = 4,        # blocks used to learn the echo level first
        cooldown: float = 0.5,         # seconds before it can trigger again
        sample_rate: int = 16000,
        speech_band_ratio: float = 0.35,  # min share of energy in 300-3400 Hz
        echo_history: int = 40,        # blocks of history for the echo estimate
        debug: bool = False,
    ):
        self.min_rms = float(min_rms)
        self.margin = float(margin)
        self.hold_blocks = int(hold_blocks)
        self.window_blocks = max(int(window_blocks), int(hold_blocks))
        self.warmup_blocks = int(warmup_blocks)
        self.cooldown = float(cooldown)
        self.sample_rate = int(sample_rate)
        self.speech_band_ratio = float(speech_band_ratio)
        self.echo_history = int(echo_history)
        self.debug = bool(debug)
        self._last_trigger = 0.0
        self.reset()

    def reset(self) -> None:
        """Call whenever JARVIS starts a new stretch of speech."""
        self._hist: deque[float] = deque(maxlen=self.echo_history)
        self._recent: deque[bool] = deque(maxlen=self.window_blocks)
        self._seen = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_float(samples) -> np.ndarray:
        try:
            return np.asarray(samples, dtype=np.float32).reshape(-1)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)))

    def _speech_ratio(self, x: np.ndarray) -> float:
        """Share of the block's energy that sits in the human-speech band."""
        if x.size < 64:
            return 1.0
        try:
            spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            freqs = np.fft.rfftfreq(x.size, d=1.0 / self.sample_rate)
            total = float(spec.sum())
            if total <= 0.0:
                return 0.0
            band = float(spec[(freqs >= 300.0) & (freqs <= 3400.0)].sum())
            return band / total
        except Exception:
            return 1.0

    # --------------------------------------------------------------------- main
    def feed(self, samples) -> bool:
        """Feed one mic block (int16). Returns True exactly when the user has
        started talking over the assistant."""
        x = self._to_float(samples)
        rms = self._rms(x)
        self._seen += 1

        # Warm-up: just learn how loud JARVIS's own voice is in the mic.
        if self._seen <= self.warmup_blocks:
            self._hist.append(rms)
            return False

        echo = float(np.median(self._hist)) if self._hist else 0.0
        threshold = max(self.min_rms, echo * self.margin)

        is_voice = rms > threshold and self._speech_ratio(x) >= self.speech_band_ratio
        self._recent.append(is_voice)

        # Only quiet / echo-level blocks update the echo history, so a talking
        # user never raises the threshold against themselves.
        if not is_voice:
            self._hist.append(rms)

        if self.debug and self._seen % 4 == 0:
            print(
                f"[barge-in] rms={rms:7.0f} echo={echo:7.0f} "
                f"thr={threshold:7.0f} voice={sum(self._recent)}/{self.hold_blocks}"
            )

        if sum(self._recent) >= self.hold_blocks:
            self._recent.clear()
            now = time.monotonic()
            if now - self._last_trigger >= self.cooldown:
                self._last_trigger = now
                return True
        return False