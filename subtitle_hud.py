"""
subtitle_hud.py - drop-in replacements for the right-hand transcript and the orb.

  * TranscriptWidget : game-subtitle look. Every sentence is its own translucent
    block, words appear one at a time (for both YOU and the assistant), and a
    blinking cursor box trails the line that is currently being typed.
    Same constructor and methods ui.py already uses: feed(), live(),
    set_ai_name().

  * RedGlowOrb       : OrbCanvas + a soft red glow inside the left side of the orb.
    Tune the three constants at the top of the class if the glow does not sit
    exactly on your orb.
"""
from __future__ import annotations

import math
import sys
import time

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from orb_hud import OrbCanvas


# ── theme access ─────────────────────────────────────────────────────────────
def _theme():
    """The live colour class C from ui.py, so the accent-colour picker still
    recolours this widget. Returns None if ui.py is not loaded (fallbacks used)."""
    m = sys.modules.get("ui") or sys.modules.get("__main__")
    return getattr(m, "C", None)


def _col(name: str, fallback: str, alpha: int = 255) -> QColor:
    c = _theme()
    q = QColor(getattr(c, name, fallback) if c else fallback)
    q.setAlpha(max(0, min(255, int(alpha))))
    return q


# ── transcript ───────────────────────────────────────────────────────────────
class _Turn:
    __slots__ = ("role", "raw", "shown", "open")

    def __init__(self, role: str):
        self.role = role      # "you" | "ai"
        self.raw = ""         # full text received so far
        self.shown = 0        # how many words are revealed
        self.open = True      # still receiving text


class TranscriptWidget(QWidget):
    _REVEAL_MS = 60           # one word per tick (faster automatically when behind)
    _MAX_TURNS = 14

    def __init__(self, assistant_name: str = "JARVIS", you_color: str = "#d8f8ff",
                 ai_color: str = "#00d4ff", bg: str = "#010d14",
                 border: str = "#0d3347", parent=None):
        super().__init__(parent)
        self._ai_name = assistant_name
        self._you_hex, self._ai_hex = you_color, ai_color
        self._bg_hex, self._border_hex = bg, border
        self._turns: list[_Turn] = []
        self._blink = True

        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._reveal = QTimer(self)
        self._reveal.setInterval(self._REVEAL_MS)
        self._reveal.timeout.connect(self._tick)

        self._blink_tmr = QTimer(self)
        self._blink_tmr.setInterval(520)
        self._blink_tmr.timeout.connect(self._on_blink)
        self._blink_tmr.start()

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(440, 300)

    # ── public API (matches the old widget) ──────────────────────────────────
    def set_ai_name(self, name: str) -> None:
        self._ai_name = name
        self.update()

    def feed(self, role: str, text: str) -> None:
        """Append a chunk of speech. role: 'you' | 'ai' | 'end' (close the turn)."""
        if role == "end":
            self._close_all()
            return
        if role not in ("you", "ai") or not text:
            return
        t = self._open_turn(role)
        t.raw += text
        self._kick()

    def live(self, role: str, text: str, final: bool = False) -> None:
        """In-place caption: `text` is the whole utterance so far."""
        if role not in ("you", "ai"):
            return
        t = self._open_turn(role)
        t.raw = text or ""
        t.shown = min(t.shown, len(t.raw.split()))
        if final:
            t.open = False
        self._kick()

    # ── turn bookkeeping ─────────────────────────────────────────────────────
    def _close_all(self) -> None:
        for t in self._turns:
            t.open = False

    def _open_turn(self, role: str) -> _Turn:
        for t in self._turns:                       # the other speaker is done
            if t.open and t.role != role:
                t.open = False
        if self._turns and self._turns[-1].open and self._turns[-1].role == role:
            return self._turns[-1]
        t = _Turn(role)
        self._turns.append(t)
        if len(self._turns) > self._MAX_TURNS:
            self._turns.pop(0)
        return t

    def _kick(self) -> None:
        if not self._reveal.isActive():
            self._reveal.start()
        self.update()

    def _tick(self) -> None:
        backlog, target = 0, None
        for t in self._turns:
            n = len(t.raw.split())
            if t.shown < n:
                backlog += n - t.shown
                if target is None:
                    target = (t, n)
        if target is None:
            self._reveal.stop()
            return
        t, n = target
        step = 1 if backlog < 6 else 2 if backlog < 14 else 4
        t.shown = min(n, t.shown + step)
        self.update()

    def _on_blink(self) -> None:
        self._blink = not self._blink
        if self._turns and self.isVisible():
            self.update()

    # ── layout helpers ───────────────────────────────────────────────────────
    def _blocks(self):
        """[(role, is_first_of_turn, [words])] - one block per sentence/phrase,
        built only from the words revealed so far."""
        out = []
        for t in self._turns:
            words = t.raw.split()[: t.shown]
            if not words:
                continue
            cur, first = [], True
            for w in words:
                cur.append(w)
                end = w[-1] in ".?!…"
                soft = w[-1] in ",;:" and len(cur) >= 6
                if end or soft or len(cur) >= 12:
                    out.append((t.role, first, cur))
                    cur, first = [], False
            if cur:
                out.append((t.role, first, cur))
        return out

    @staticmethod
    def _wrap(fm: QFontMetrics, words, width: int):
        lines, cur = [], ""
        for w in words:
            w = w.upper()
            trial = w if not cur else f"{cur} {w}"
            if cur and fm.horizontalAdvance(trial) > width:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        p.setPen(QPen(_col("BORDER", self._border_hex), 1))
        p.setBrush(QBrush(_col("PANEL", self._bg_hex)))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 10, 10)

        font = QFont("Consolas" if sys.platform == "win32" else "Courier New", 10)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        small = QFont(font)
        small.setPointSize(7)
        fm, sfm = QFontMetrics(font), QFontMetrics(small)

        blocks = self._blocks()
        if not blocks:
            p.setFont(small)
            p.setPen(_col("TEXT_DIM", "#3a8a9a"))
            p.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       "AWAITING VOICE INPUT")
            p.end()
            return

        pad, bar, px, py, gap = 12, 3, 10, 7, 7
        text_w = W - 2 * pad - bar - 2 * px
        lh = fm.height() + 2
        x0 = pad + bar + px

        p.setClipRect(QRectF(0, pad - 4, W, H - 2 * pad + 8))
        y = H - pad
        last_index = len(blocks) - 1
        for n, (role, first, words) in enumerate(reversed(blocks)):
            if y <= pad:
                break
            lines = self._wrap(fm, words, text_w)
            label_h = (sfm.height() + 2) if first else 0
            bh = len(lines) * lh + 2 * py
            top = y - bh
            fade = max(0.22, 1.0 - 0.26 * n)

            accent = _col("PRI", self._ai_hex) if role == "ai" else _col("WHITE", self._you_hex)
            rect = QRectF(pad, top, W - 2 * pad, bh)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_col("PRI", self._ai_hex, 40 * fade) if role == "ai"
                              else _col("WHITE", self._you_hex, 22 * fade)))
            p.drawRoundedRect(rect, 4, 4)
            p.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(),
                                     int(255 * fade))))
            p.drawRoundedRect(QRectF(pad, top, bar, bh), 1.5, 1.5)

            p.setFont(font)
            txt = _col("WHITE", "#d8f8ff", 255 * fade)
            p.setPen(txt)
            for i, line in enumerate(lines):
                p.drawText(QPointF(x0, top + py + i * lh + fm.ascent()), line)

            if n == 0 and self._blink:                      # cursor on the live line
                cx = x0 + fm.horizontalAdvance(lines[-1]) + 5
                cy = top + py + (len(lines) - 1) * lh + 2
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(_col("PRI", self._ai_hex, 235)))
                p.drawRect(QRectF(cx, cy, 8, fm.ascent() - 2))

            y = top
            if first:
                p.setFont(small)
                p.setPen(_col("TEXT_DIM", "#3a8a9a", 255 * fade))
                name = "YOU" if role == "you" else self._ai_name.upper()
                p.drawText(QPointF(pad + 2, y - 3), name)
                y -= label_h
            y -= gap
        p.end()


# ── orb with red glow ────────────────────────────────────────────────────────
class RedGlowOrb(OrbCanvas):
    """OrbCanvas plus a soft red glow clipped to the left side of the orb."""

    ORB_RADIUS_FRAC = 0.30    # orb radius as a fraction of min(width, height)
    ORB_CENTER_Y    = 0.47    # orb centre height as a fraction of widget height
    GLOW_X_SHIFT    = 0.55    # glow centre, in radii, to the left of the orb centre

    def paintEvent(self, e):
        super().paintEvent(e)
        try:
            self._paint_red_glow()
        except Exception:
            pass

    def _paint_red_glow(self) -> None:
        W, H = self.width(), self.height()
        cx, cy = W / 2, H * self.ORB_CENTER_Y
        r = min(W, H) * self.ORB_RADIUS_FRAC

        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), r, r)
        p.setClipPath(clip)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        pulse = 0.85 + 0.15 * math.sin(time.time() * 2.2)
        g = QRadialGradient(QPointF(cx - r * self.GLOW_X_SHIFT, cy), r)
        g.setColorAt(0.0, QColor(255, 30, 60, int(150 * pulse)))
        g.setColorAt(0.5, QColor(255, 20, 50, int(60 * pulse)))
        g.setColorAt(1.0, QColor(255, 0, 40, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawRect(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.end()