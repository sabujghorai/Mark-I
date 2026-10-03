"""
Local wake-word and gesture detection for JARVIS.

Wake triggers:
    1. "Hey Jarvis"
    2. Double clap
    3. Finger snap

All detection is local.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


# ============================================================
# WAKE WORD SETTINGS
# ============================================================

WAKE_MODEL = "hey_jarvis"

# OpenWakeWord confidence threshold.
DEFAULT_THRESHOLD = 0.5

# Microphone input rate used by main.py.
SAMPLE_RATE = 16000


# ============================================================
# GESTURE SETTINGS
# ============================================================

# Turn clap/snap wake detection on/off.
GESTURE_WAKE_ENABLED = True

# Two clap events must occur within this amount of time.
DOUBLE_CLAP_WINDOW = 1.2

# Minimum time between two detections of the same sound.
GESTURE_COOLDOWN = 0.20

# Basic loudness threshold.
GESTURE_RMS_THRESHOLD = 700.0

# Minimum peak amplitude.
GESTURE_PEAK_THRESHOLD = 1800.0

# High-frequency ratio used to identify a sharp snap-like sound.
SNAP_HIGH_FREQ_RATIO = 0.30

# Crest factor used to identify a sharp sound.
SNAP_CREST_FACTOR = 5.0


# ============================================================
# PACKAGE / MODEL CHECKS
# ============================================================

def is_installed() -> bool:
    """Return True when openwakeword is installed."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("openwakeword") is not None
        )

    except Exception:
        return False


def is_ready() -> bool:
    """
    Return True when openwakeword and its model files exist.
    Does not construct a Model object.
    """
    if not is_installed():
        return False

    try:
        import openwakeword

        models_dir = (
            Path(openwakeword.__file__).resolve().parent
            / "resources"
            / "models"
        )

        if not models_dir.is_dir():
            return False

        has_wake = (
            any(models_dir.glob(f"{WAKE_MODEL}*.onnx"))
            or any(models_dir.glob(f"{WAKE_MODEL}*.tflite"))
        )

        has_mel = (
            any(models_dir.glob("melspectrogram*.onnx"))
            or any(models_dir.glob("melspectrogram*.tflite"))
        )

        has_embedding = (
            any(models_dir.glob("embedding_model*.onnx"))
            or any(models_dir.glob("embedding_model*.tflite"))
        )

        return bool(
            has_wake
            and has_mel
            and has_embedding
        )

    except Exception:
        return False


def install_and_download(
    logger: Callable[[str], None] = print
) -> tuple[bool, str]:
    """
    Install openwakeword and download the pretrained wake model.
    """

    try:
        if not is_installed():
            logger(
                "Wake word: installing openwakeword..."
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "openwakeword",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                lines = (
                    result.stderr
                    or result.stdout
                    or ""
                ).strip().splitlines()

                last_line = (
                    lines[-1]
                    if lines
                    else "Unknown installation error"
                )

                return (
                    False,
                    f"pip install failed: {last_line[:160]}"
                )

        logger(
            "Wake word: downloading models..."
        )

        try:
            import openwakeword.utils as utils

            try:
                utils.download_models([WAKE_MODEL])
            except TypeError:
                utils.download_models()

        except Exception as error:
            return (
                False,
                f"Model download failed: {error}"
            )

        if not is_ready():
            return (
                False,
                "Installed, but wake model is not ready."
            )

        logger("Wake word: ready.")

        return (
            True,
            "Wake word installed and ready."
        )

    except Exception as error:
        return False, f"Setup error: {error}"


# ============================================================
# WAKE WORD + GESTURE DETECTOR
# ============================================================

class WakeWordDetector:
    """
    Runs wake-word and gesture detection in a background thread.

    The microphone callback only calls feed().
    Actual detection runs in this thread.
    """

    def __init__(
        self,
        on_detect: Callable[[], None],
        threshold: float = DEFAULT_THRESHOLD,
        logger: Callable[[str], None] = print,
    ):
        self._on_detect = on_detect
        self._threshold = threshold
        self._logger = logger

        # Audio queue.
        self._queue: queue.Queue = queue.Queue(
            maxsize=50
        )

        # Thread state.
        self._thread: threading.Thread | None = None
        self._running = False

        # OpenWakeWord model.
        self._model = None
        self._ready = False

        # ----------------------------------------------------
        # Gesture state
        # ----------------------------------------------------

        self._last_gesture_time = 0.0

        self._clap_count = 0
        self._last_clap_time = 0.0

        # Estimate normal background noise.
        self._noise_floor = 100.0

    # ========================================================
    # START
    # ========================================================

    def start(self) -> bool:
        """
        Load OpenWakeWord and start the background detector.
        """

        if self._running:
            return True

        try:
            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=[WAKE_MODEL],
                inference_framework="onnx",
            )

        except Exception as error:

            self._logger(
                f"Wake word: could not load model - {error}"
            )

            self._model = None
            self._ready = False

            return False

        self._running = True
        self._ready = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="WakeWordThread",
        )

        self._thread.start()

        self._logger(
            "Wake system: listening for "
            "'Hey Jarvis', double clap, and snap."
        )

        return True

    # ========================================================
    # STOP
    # ========================================================

    def stop(self) -> None:
        """Stop detector thread and release the model."""

        self._running = False

        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        self._thread = None
        self._model = None
        self._ready = False

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def ready(self) -> bool:
        return self._ready

    # ========================================================
    # AUDIO FEED
    # ========================================================

    def feed(self, frame_int16) -> None:
        """
        Called from the real-time microphone callback.

        Must remain fast and non-blocking.
        """

        if not self._running:
            return

        try:
            # Convert possible 2-D mono array to 1-D.
            if getattr(frame_int16, "ndim", 1) > 1:
                data = frame_int16[:, 0].copy()
            else:
                data = frame_int16.copy()

            self._queue.put_nowait(data)

        except queue.Full:
            # Drop frames instead of blocking the microphone.
            pass

        except Exception:
            pass

    # ========================================================
    # GESTURE DETECTOR
    # ========================================================

    def _detect_gesture(self, frame) -> str | None:
        """
        Detect:
            - one sharp finger snap
            - two clap-like sounds within the configured window

        Returns:
            "snap"
            "double_clap"
            None
        """

        if not GESTURE_WAKE_ENABLED:
            return None

        try:
            import numpy as np

            x = np.asarray(
                frame,
                dtype=np.float32,
            ).flatten()

            if x.size == 0:
                return None

            now = time.monotonic()

            # ------------------------------------------------
            # Volume
            # ------------------------------------------------

            rms = float(
                np.sqrt(
                    np.mean(x * x)
                )
            )

            peak = float(
                np.max(
                    np.abs(x)
                )
            )

            # Slowly learn room noise.
            if rms < self._noise_floor * 2.0:
                self._noise_floor = (
                    self._noise_floor * 0.95
                    + rms * 0.05
                )

            dynamic_threshold = max(
                GESTURE_RMS_THRESHOLD,
                self._noise_floor * 3.5,
            )

            if rms < dynamic_threshold:
                return None

            if peak < GESTURE_PEAK_THRESHOLD:
                return None

            # ------------------------------------------------
            # Prevent duplicate detection
            # ------------------------------------------------

            if (
                now - self._last_gesture_time
                < GESTURE_COOLDOWN
            ):
                return None

            # ------------------------------------------------
            # Frequency analysis
            # ------------------------------------------------

            window = np.hanning(
                len(x)
            )

            spectrum = np.abs(
                np.fft.rfft(
                    x * window
                )
            )

            frequencies = np.fft.rfftfreq(
                len(x),
                d=1.0 / SAMPLE_RATE,
            )

            total_band = spectrum[
                (frequencies >= 300)
                & (frequencies <= 8000)
            ]

            high_band = spectrum[
                (frequencies >= 2000)
                & (frequencies <= 8000)
            ]

            if total_band.size == 0:
                return None

            total_energy = float(
                np.sum(total_band)
            )

            high_energy = float(
                np.sum(high_band)
            )

            if total_energy <= 0:
                return None

            high_ratio = (
                high_energy
                / total_energy
            )

            crest_factor = (
                peak
                / max(rms, 1.0)
            )

            # ------------------------------------------------
            # Finger snap detection
            # ------------------------------------------------

            snap_like = (
                crest_factor >= SNAP_CREST_FACTOR
                and high_ratio >= SNAP_HIGH_FREQ_RATIO
            )

            if snap_like:

                self._last_gesture_time = now

                self._clap_count = 0
                self._last_clap_time = 0.0

                self._logger(
                    "Gesture wake detected: finger snap."
                )

                return "snap"

            # ------------------------------------------------
            # Clap detection
            # ------------------------------------------------

            if (
                now - self._last_clap_time
                <= DOUBLE_CLAP_WINDOW
            ):
                self._clap_count += 1
            else:
                self._clap_count = 1

            self._last_clap_time = now

            if self._clap_count >= 2:

                self._last_gesture_time = now

                self._clap_count = 0
                self._last_clap_time = 0.0

                self._logger(
                    "Gesture wake detected: double clap."
                )

                return "double_clap"

            return None

        except Exception as error:

            self._logger(
                f"Gesture detection error - {error}"
            )

            return None

    # ========================================================
    # MAIN DETECTION LOOP
    # ========================================================

    def _loop(self) -> None:
        import numpy as np

        while self._running:

            try:
                frame = self._queue.get()

                if (
                    frame is None
                    or not self._running
                ):
                    break

                # ------------------------------------------------
                # First check gestures
                # ------------------------------------------------

                gesture = self._detect_gesture(
                    frame
                )

                if gesture is not None:

                    # Remove old frames so the same gesture
                    # cannot immediately fire again.
                    self._drain()

                    try:
                        self._on_detect()

                    except Exception as error:
                        self._logger(
                            "Wake callback error - "
                            f"{error}"
                        )

                    continue

                # ------------------------------------------------
                # Then check "Hey Jarvis"
                # ------------------------------------------------

                if self._model is None:
                    continue

                scores = self._model.predict(
                    np.asarray(
                        frame,
                        dtype=np.int16,
                    )
                )

                score = 0.0

                if isinstance(scores, dict):

                    # Prefer the Jarvis score.
                    for key, value in scores.items():

                        if (
                            "jarvis"
                            in str(key).lower()
                        ):
                            score = max(
                                score,
                                float(value),
                            )

                    # Fallback if key format changes.
                    if (
                        score == 0.0
                        and scores
                    ):
                        score = max(
                            float(value)
                            for value in scores.values()
                        )

                if score >= self._threshold:

                    self._drain()

                    try:
                        self._on_detect()

                    except Exception as error:
                        self._logger(
                            "Wake callback error - "
                            f"{error}"
                        )

            except Exception as error:

                self._logger(
                    f"Wake inference error - {error}"
                )

    # ========================================================
    # DRAIN QUEUE
    # ========================================================

    def _drain(self) -> None:

        try:

            while True:
                self._queue.get_nowait()

        except queue.Empty:
            pass

        except Exception:
            pass