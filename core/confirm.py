
"""
core/confirm.py — human confirmation manager for JARVIS.

Actions normally require confirmation through the HUD.
Trusted actions can optionally execute automatically when
the calling code explicitly enables auto_execute.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

TIMEOUT_SECONDS = 90.0


@dataclass
class _Pending:
    key: str
    title: str
    detail: str
    run: Callable[[], str]
    at: float


_pending: Optional[_Pending] = None
_lock = threading.Lock()

_show_cb: Optional[Callable[[str, str], None]] = None
_hide_cb: Optional[Callable[[], None]] = None
_log_cb: Optional[Callable[[str], None]] = None


def bind(show, hide, log=None) -> None:
    """Connect this module to the JARVIS HUD."""
    global _show_cb, _hide_cb, _log_cb
    _show_cb, _hide_cb, _log_cb = show, hide, log


def _log(msg: str) -> None:
    if _log_cb:
        try:
            _log_cb(msg)
        except Exception:
            pass


def _run_async(key: str, title: str,
               run: Callable[[], str]) -> None:
    """Run an action in a background thread."""

    def _worker():
        try:
            result = run() or "Done."
            _log(f"SYS: Completed — {title}. {result}")
        except Exception as e:
            _log(f"ERR: {title} failed — {e}")

    threading.Thread(
        target=_worker,
        daemon=True,
        name=f"confirm-{key}"
    ).start()


def request(
    key: str,
    title: str,
    detail: str,
    run: Callable[[], str],
    auto_execute: bool = False
) -> str:
    """
    Request confirmation for an action.

    auto_execute must be explicitly set by trusted calling code,
    never directly from an AI-generated tool parameter.
    """

    global _pending

    # Explicitly authorized automatic execution
    if auto_execute:
        with _lock:
            if _pending is not None:
                if time.monotonic() - _pending.at <= TIMEOUT_SECONDS:
                    return (
                        "Another confirmation is pending. "
                        "Resolve it before starting another action."
                    )
                _pending = None

        _log(f"SYS: Automatic execution — {title}")
        _run_async(key, title, run)
        return f"{title} has been started automatically."

    # Manual confirmation requires the interface
    if _show_cb is None:
        return (
            f"I cannot confirm '{title}' because the interface "
            "is unavailable. Nothing was done."
        )

    with _lock:
        if _pending is not None:
            if time.monotonic() - _pending.at <= TIMEOUT_SECONDS:
                return (
                    "Another confirmation is already pending. "
                    "Please resolve it first."
                )
            _pending = None

        _pending = _Pending(
            key=key,
            title=title,
            detail=detail,
            run=run,
            at=time.monotonic()
        )

    try:
        _show_cb(title, detail)
    except Exception as e:
        with _lock:
            if _pending is not None and _pending.key == key:
                _pending = None
        return f"Could not ask for confirmation: {e}. Nothing was done."

    _log(f"SYS: Awaiting confirmation — {title}")

    return (
        f"[CONFIRMATION_PENDING] Confirmation is required for: {title}. "
        "Ask the user to confirm on the HUD. Do not claim it is done."
    )


def resolve(accepted: bool) -> None:
    """Handle the user's CONFIRM or CANCEL choice."""

    global _pending

    with _lock:
        p, _pending = _pending, None

    if _hide_cb:
        try:
            _hide_cb()
        except Exception:
            pass

    if p is None:
        return

    if time.monotonic() - p.at > TIMEOUT_SECONDS:
        _log(f"SYS: Confirmation expired — {p.title}")
        return

    if not accepted:
        _log(f"SYS: Cancelled — {p.title}")
        return

    _log(f"SYS: Confirmed — {p.title}")
    _run_async(p.key, p.title, p.run)


def pending_title() -> str:
    """Return the active confirmation title, or an empty string."""

    with _lock:
        if _pending is None:
            return ""

        if time.monotonic() - _pending.at > TIMEOUT_SECONDS:
            return ""

        return _pending.title