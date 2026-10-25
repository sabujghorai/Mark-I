"""
core/barge_in.py - detects the user talking OVER the assistant ("barge-in").

While JARVIS speaks, main.py normally ignores the microphone so JARVIS does not
hear its own voice. This detector listens to the same microphone blocks during
that time and answers one question: "is a real person talking right now?"

How it decides (pure local maths, no network, runs in the audio callback):
  * It learns the room/echo level during the first few blocks after JARVIS
    starts speaking (the sound of its own voice coming back into the mic).
  * Your voice counts only if it is clearly louder than that echo level AND
    louder than an absolute floor (min_rms).
  * It must stay loud for a short time (hold_blocks), so a cough, a click or a
    single loud syllable of echo does not cut JARVIS off.

Tip: headphones remove the echo problem completely. On open speakers, raise
min_rms (BARGE_IN_MIN_RMS in main.py) if JARVIS interrupts itself, or lower it
if you have to shout.
"""
from __future__ import annotations

import time

import numpy as np


class BargeInDetector:
    def __init__(
        self,
        min_rms: float = 1200.0,     # absolute loudness floor (int16 RMS)
        margin: float = 2.2,         # voice must be this many times louder than the echo
        hold_blocks: int = 4,        # ~4 x 64 ms = ~256 ms of sustained speech
        warmup_blocks: int = 3,      # blocks used to learn the echo level
        cooldown: float = 1.0,       # seconds before it can trigger again
    ):
        self.min_rms = float(min_rms)
        self.margin = float(margin)
        self.hold_blocks = int(hold_blocks)
        self.warmup_blocks = int(warmup_blocks)
        self.cooldown = float(cooldown)
        self.reset()
        self._last_trigger = 0.0

    def reset(self) -> None:
        """Call whenever JARVIS starts a new stretch of speech."""
        self._baseline = 0.0
        self._seen = 0
        self._count = 0

    @staticmethod
    def _rms(samples) -> float:
        try:
            x = np.asarray(samples, dtype=np.float32)
            if x.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(x * x)))
        except Exception:
            return 0.0

    def feed(self, samples) -> bool:
        """Feed one mic block (int16). Returns True exactly when the user has
        started talking over the assistant."""
        rms = self._rms(samples)
        self._seen += 1

        # Warm-up: learn how loud JARVIS's own voice is in the mic.
        if self._seen <= self.warmup_blocks:
            self._baseline = rms if self._baseline == 0.0 else 0.5 * self._baseline + 0.5 * rms
            return False

        threshold = max(self.min_rms, self._baseline * self.margin)

        if rms > threshold:
            self._count += 1
            # Loud blocks barely move the baseline, so a talking user does not
            # teach the detector that talking is "normal".
            self._baseline = 0.99 * self._baseline + 0.01 * rms
        else:
            self._count = max(0, self._count - 1)
            self._baseline = 0.9 * self._baseline + 0.1 * rms

        if self._count >= self.hold_blocks:
            now = time.monotonic()
            self._count = 0
            if now - self._last_trigger >= self.cooldown:
                self._last_trigger = now
                return True
        return False