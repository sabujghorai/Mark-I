"""
Screen & webcam capture for JARVIS vision.

Provides:
    _capture_screen()
    _capture_camera()

and a persistent webcam preview:

    _open_camera_preview()
    _close_camera_preview()
    _is_camera_preview_running()

The normal capture functions are one-shot captures for Gemini vision.
The preview functions keep the webcam open until JARVIS explicitly closes it.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False


# ============================================================================
# PATH / CONFIG
# ============================================================================

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"


def _load_config() -> dict:
    try:
        return json.loads(
            _CONFIG_PATH.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def _save_config_key(key: str, value) -> None:
    try:
        cfg = _load_config()
        cfg[key] = value

        _CONFIG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        _CONFIG_PATH.write_text(
            json.dumps(cfg, indent=4),
            encoding="utf-8"
        )

    except Exception as e:
        print(
            f"[Vision] ⚠️ Could not save config key "
            f"'{key}': {e}"
        )


def _get_os() -> str:
    """
    Get the operating system stored in JARVIS config.

    Supported values:
        windows
        mac
        linux
    """

    return _load_config().get(
        "os_system",
        "windows"
    ).lower()


# ============================================================================
# IMAGE SETTINGS
# ============================================================================

_IMG_MAX_W = 1280
_IMG_MAX_H = 720
_JPEG_Q = 82


def _compress(
    img_bytes: bytes,
    source_format: str = "PNG"
) -> tuple[bytes, str]:

    if not _PIL:
        return (
            img_bytes,
            f"image/{source_format.lower()}"
        )

    try:
        img = PIL.Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB")

        img.thumbnail(
            (_IMG_MAX_W, _IMG_MAX_H),
            PIL.Image.BILINEAR
        )

        buf = io.BytesIO()

        img.save(
            buf,
            format="JPEG",
            quality=_JPEG_Q,
            optimize=False
        )

        return (
            buf.getvalue(),
            "image/jpeg"
        )

    except Exception as e:
        print(
            f"[Vision] ⚠️ Image compress failed: {e}"
        )

        return (
            img_bytes,
            f"image/{source_format.lower()}"
        )


# ============================================================================
# SCREEN CAPTURE
# ============================================================================

def _capture_screen() -> tuple[bytes, str]:
    """
    Capture one screenshot.

    Returns:
        (image_bytes, mime_type)
    """

    if not _MSS:
        raise RuntimeError(
            "mss is not installed. Run: pip install mss"
        )

    with mss.mss() as sct:

        monitors = sct.monitors

        # [0] = all combined monitors
        # [1..n] = real monitors
        target = (
            monitors[1]
            if len(monitors) > 1
            else monitors[0]
        )

        shot = sct.grab(target)

        png = mss.tools.to_png(
            shot.rgb,
            shot.size
        )

    return _compress(
        png,
        "PNG"
    )


# ============================================================================
# OPENCV CAMERA BACKEND
# ============================================================================

def _cv2_backend() -> int:
    """
    Return the best OpenCV camera backend
    for the current operating system.
    """

    if not _CV2:
        return 0

    os_name = _get_os()

    if os_name == "windows":
        return cv2.CAP_DSHOW

    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION

    return cv2.CAP_ANY


# ============================================================================
# CAMERA DETECTION
# ============================================================================

def _probe_camera(
    index: int,
    backend: int,
    warmup: int = 5
) -> bool:

    if not _CV2:
        return False

    cap = cv2.VideoCapture(
        index,
        backend
    )

    if not cap.isOpened():
        cap.release()
        return False

    try:

        for _ in range(warmup):
            cap.read()

        ret, frame = cap.read()

    finally:
        cap.release()

    if not ret or frame is None:
        return False

    try:
        return bool(
            np.mean(frame) > 8
        )
    except Exception:
        return True


def _detect_camera_index() -> int:

    backend = _cv2_backend()

    print(
        "[Vision] 🔍 Auto-detecting camera..."
    )

    for idx in range(6):

        if _probe_camera(
            idx,
            backend
        ):

            print(
                f"[Vision] ✅ Camera found "
                f"at index {idx}"
            )

            _save_config_key(
                "camera_index",
                idx
            )

            return idx

        print(
            f"[Vision] ⚠️ Camera index {idx}: "
            f"no usable frame"
        )

    print(
        "[Vision] ⚠️ No camera found — "
        "defaulting to index 0"
    )

    _save_config_key(
        "camera_index",
        0
    )

    return 0


def _get_camera_index() -> int:

    cfg = _load_config()

    if "camera_index" in cfg:

        try:
            return int(
                cfg["camera_index"]
            )
        except Exception:
            pass

    return _detect_camera_index()


# ============================================================================
# ONE-SHOT CAMERA CAPTURE
# ============================================================================

def _capture_camera() -> tuple[bytes, str]:
    """
    Capture ONE frame from the webcam.

    This function is intentionally one-shot.

    It is used when Gemini/JARVIS needs to LOOK
    at something.

    It does NOT keep the camera open.

    For a persistent camera window use:

        _open_camera_preview()
    """

    if not _CV2:
        raise RuntimeError(
            "OpenCV (cv2) is not installed. "
            "Run: pip install opencv-python"
        )

    index = _get_camera_index()
    backend = _cv2_backend()

    cap = cv2.VideoCapture(
        index,
        backend
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {index} could not be opened."
        )

    try:

        # Allow the camera to warm up.
        for _ in range(10):
            cap.read()

        ret, frame = cap.read()

    finally:

        # IMPORTANT:
        # This capture is only for Gemini vision.
        # Therefore it is safe to release it here.
        cap.release()

    if not ret or frame is None:
        raise RuntimeError(
            "Camera returned no frame."
        )

    if _PIL:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = PIL.Image.fromarray(
            rgb
        )

        img.thumbnail(
            (_IMG_MAX_W, _IMG_MAX_H),
            PIL.Image.BILINEAR
        )

        buf = io.BytesIO()

        img.save(
            buf,
            format="JPEG",
            quality=_JPEG_Q
        )

        return (
            buf.getvalue(),
            "image/jpeg"
        )

    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            _JPEG_Q
        ]
    )

    if not ok:
        raise RuntimeError(
            "Failed to encode camera frame."
        )

    return (
        buf.tobytes(),
        "image/jpeg"
    )


# ============================================================================
# PERSISTENT CAMERA PREVIEW
# ============================================================================

_camera_preview_thread = None
_camera_preview_stop = None
_camera_preview_lock = threading.Lock()
_camera_preview_running = False


def _is_camera_preview_running() -> bool:
    """
    Return True if the persistent camera preview
    is currently running.
    """

    global _camera_preview_running

    with _camera_preview_lock:
        return bool(
            _camera_preview_running
        )


def _camera_preview_worker(
    camera_index: int,
    window_name: str,
    stop_event: threading.Event
) -> None:

    global _camera_preview_running

    cap = None

    try:

        backend = _cv2_backend()

        print(
            f"[Vision] 📷 Opening camera preview "
            f"at index {camera_index}"
        )

        cap = cv2.VideoCapture(
            camera_index,
            backend
        )

        if not cap.isOpened():

            print(
                "[Vision] ❌ Could not open "
                "camera preview."
            )

            return

        # Try to use a reasonable preview size.
        try:
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                1280
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                720
            )

        except Exception:
            pass

        # Warm up camera.
        for _ in range(10):

            if stop_event.is_set():
                return

            cap.read()

        with _camera_preview_lock:
            _camera_preview_running = True

        print(
            "[Vision] ✅ Camera preview is running."
        )

        while not stop_event.is_set():

            ret, frame = cap.read()

            if not ret or frame is None:

                print(
                    "[Vision] ⚠️ Camera frame "
                    "could not be read."
                )

                time.sleep(0.05)
                continue

            # Mirror the webcam so it behaves
            # like a normal camera preview.
            frame = cv2.flip(
                frame,
                1
            )

            cv2.imshow(
                window_name,
                frame
            )

            # waitKey is required by OpenCV
            # to process the window.
            key = cv2.waitKey(1) & 0xFF

            # ESC or Q closes the preview.
            if key in (
                27,
                ord("q"),
                ord("Q")
            ):

                stop_event.set()
                break

            # Check whether the user manually
            # closed the OpenCV window.
            try:

                visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE
                )

                if visible < 1:

                    stop_event.set()
                    break

            except Exception:
                pass

    except Exception as e:

        print(
            f"[Vision] ❌ Camera preview error: {e}"
        )

    finally:

        # Always release the camera when
        # the preview is actually stopped.
        if cap is not None:

            try:
                cap.release()
            except Exception:
                pass

        try:
            cv2.destroyWindow(
                window_name
            )
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        with _camera_preview_lock:
            _camera_preview_running = False

        print(
            "[Vision] 📷 Camera preview closed."
        )


def _open_camera_preview(
    window_name: str = "JARVIS CAMERA"
) -> bool:
    """
    Open the webcam and KEEP IT OPEN.

    Calling this function multiple times will not
    create multiple camera windows.

    Returns:
        True  -> preview started/already running
        False -> preview could not be started
    """

    global _camera_preview_thread
    global _camera_preview_stop

    if not _CV2:

        print(
            "[Vision] ❌ OpenCV is not installed."
        )

        return False

    with _camera_preview_lock:

        if (
            _camera_preview_thread is not None
            and _camera_preview_thread.is_alive()
        ):

            print(
                "[Vision] 📷 Camera preview "
                "is already running."
            )

            return True

        camera_index = _get_camera_index()

        _camera_preview_stop = (
            threading.Event()
        )

        _camera_preview_thread = threading.Thread(
            target=_camera_preview_worker,
            args=(
                camera_index,
                window_name,
                _camera_preview_stop
            ),
            daemon=True,
            name="JARVIS-Camera-Preview"
        )

        _camera_preview_thread.start()

    return True


def _close_camera_preview() -> bool:
    """
    Stop the persistent camera preview.

    Returns:
        True  -> stop requested
        False -> preview was not running
    """

    global _camera_preview_stop
    global _camera_preview_thread

    with _camera_preview_lock:

        if (
            _camera_preview_thread is None
            or not _camera_preview_thread.is_alive()
        ):

            _camera_preview_running = False

            print(
                "[Vision] 📷 Camera preview "
                "is not running."
            )

            return False

        if _camera_preview_stop is not None:
            _camera_preview_stop.set()

        thread = _camera_preview_thread

    # Give the preview thread a short amount
    # of time to release the camera.
    if (
        thread is not None
        and thread is not threading.current_thread()
    ):

        thread.join(
            timeout=2.0
        )

    with _camera_preview_lock:

        if (
            _camera_preview_thread is not None
            and not _camera_preview_thread.is_alive()
        ):

            _camera_preview_thread = None
            _camera_preview_stop = None

    return True


# ============================================================================
# SHUTDOWN HELPER
# ============================================================================

def _shutdown_camera() -> None:
    """
    Safely close the persistent camera preview.

    Call this when JARVIS itself is shutting down.
    """

    try:
        _close_camera_preview()
    except Exception as e:
        print(
            f"[Vision] ⚠️ Camera shutdown error: {e}"
        )