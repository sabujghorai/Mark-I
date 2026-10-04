"""
orb_hud.py - dotted rotating orb + live transcript panel for the JARVIS UI.

OrbCanvas is a drop-in replacement for HudCanvas (same constructor shape and
the same attributes: muted, speaking, state, _assistant_name, set_audio_level).

  * ~4000 dots on a sphere that ALWAYS rotates slowly.
  * Audio level (mic while you talk, JARVIS voice while it talks) makes the
    dots break apart and fly out in all directions; when sound stops they
    pull back together into the sphere.

TranscriptWidget prints what you say and what the assistant says, chunk by
chunk, as the words arrive.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QTextCharFormat
from PyQt6.QtWidgets import QSizePolicy, QTextEdit, QWidget

_LAT, _LON = 46, 90          # 4140 dots - raise for denser, lower for speed
_NC = 32                     # colour buckets across the screen (left -> right)
_DEPTH = 4                   # depth buckets (back dim/small -> front bright/big)

# yellow (left)  ->  green (middle)  ->  blue (right)
_STOPS = [(0.0, (226, 204, 36)), (0.5, (52, 168, 92)), (1.0, (28, 62, 226))]


def _grad(t: float) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in zip(_STOPS, _STOPS[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return tuple(int(c0[i] + (c1[i] - c0[i]) * u) for i in range(3))
    return _STOPS[-1][1]


class OrbCanvas(QWidget):
    def __init__(self, face_path: str = "", assistant_name: str = "J.A.R.V.I.S",
                 parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.muted = False
        self.speaking = False
        self.state = "INITIALISING"
        self._assistant_name = assistant_name

        # --- sphere geometry (latitude / longitude grid) --------------------
        lat = np.linspace(-math.pi / 2 * 0.985, math.pi / 2 * 0.985, _LAT)
        lon = np.linspace(0, 2 * math.pi, _LON, endpoint=False)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        la, lo = la.ravel(), lo.ravel()
        self._base = np.stack(
            [np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], axis=1
        ).astype(np.float32)

        rng = np.random.default_rng(7)
        n = len(la)
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self._dir = d.astype(np.float32)                         # scatter direction
        self._mag = rng.uniform(0.3, 1.0, n).astype(np.float32)  # scatter distance
        self._phase = rng.uniform(0, 2 * math.pi, n).astype(np.float32)

        self._colors = [_grad(i / (_NC - 1)) for i in range(_NC)]
        self._alpha = [60, 120, 185, 245]
        self._size = [1.6, 2.4, 3.2, 4.0]

        # --- animation state -------------------------------------------------
        self._yaw = 0.0
        self._pitch = 0.32
        self._t = 0.0
        self._live_amp = 0.0     # written by audio threads
        self._amp = 0.0          # smoothed
        self._scatter = 0.0      # 0 = intact sphere, 1 = fully burst

        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(25)      # 40 fps

    # thread-safe: called from audio threads with a 0..1 level
    def set_audio_level(self, level: float) -> None:
        try:
            lv = min(1.0, max(0.0, float(level)))
        except (TypeError, ValueError):
            return
        if lv > self._live_amp:
            self._live_amp = lv

    def _step(self) -> None:
        self._live_amp *= 0.88
        self._amp += (self._live_amp - self._amp) * 0.4

        target = 0.0
        if not self.muted:
            target = min(1.0, self._amp * 1.8)
            if self.speaking:
                target = max(target, 0.35)          # keep it alive while JARVIS talks
            if self.state in ("THINKING", "PROCESSING"):
                target = max(target, 0.12)
        self._scatter += (target - self._scatter) * 0.18

        # always rotating; a touch faster while the dots are bursting
        self._yaw += 0.005 + 0.010 * self._scatter
        self._t += 0.025
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#010304"))

        W, H = self.width(), self.height()
        cx, cy = W / 2, H * 0.47
        R = min(W, H * 0.9) * 0.33
        s = self._scatter

        # break-apart: every dot drifts along its own random direction
        wob = 0.5 + 0.5 * np.sin(self._t * 3.0 + self._phase)
        disp = (s * 0.45 * self._mag * (0.15 + 0.85 * wob))[:, None]
        pos = self._base * (1.0 + 0.10 * s * wob[:, None]) + self._dir * disp

        # rotate: yaw (Y axis) then fixed pitch (X axis)
        cyw, syw = math.cos(self._yaw), math.sin(self._yaw)
        x1 = pos[:, 0] * cyw + pos[:, 2] * syw
        z1 = -pos[:, 0] * syw + pos[:, 2] * cyw
        y1 = pos[:, 1]
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp

        k = 3.4 / (3.4 - z2)                               # perspective
        sx = cx + x1 * k * R
        sy = cy - y2 * k * R

        depth = np.clip((z2 + 1.6) / 3.2, 0.0, 0.999)
        db = (depth * _DEPTH).astype(np.int32)
        xn = np.clip((sx - cx) / (R * 1.05), -1.0, 1.0)
        ci = ((xn + 1.0) * 0.5 * (_NC - 1)).astype(np.int32)

        key = db * _NC + ci
        order = np.argsort(key, kind="stable")
        bounds = np.searchsorted(key[order], np.arange(_DEPTH * _NC + 1))
        xs, ys = sx[order].tolist(), sy[order].tolist()
        grow = 1.0 + 0.35 * s

        for kk in range(_DEPTH * _NC):
            a, b = bounds[kk], bounds[kk + 1]
            if a == b:
                continue
            d_i, c_i = divmod(kk, _NC)
            if self.muted:
                col = QColor(255, 51, 102, self._alpha[d_i] // 2)
            else:
                r, g, bl = self._colors[c_i]
                col = QColor(r, g, bl, self._alpha[d_i])
            pen = QPen(col, self._size[d_i] * grow)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawPoints(QPolygonF([QPointF(x, y) for x, y in zip(xs[a:b], ys[a:b])]))

        # status line under the orb
        if self.muted:
            txt, col = "MUTED", QColor("#ff3366")
        elif self.speaking:
            txt, col = "SPEAKING", QColor("#ff6b00")
        elif self.state in ("THINKING", "PROCESSING"):
            txt, col = self.state, QColor("#ffcc00")
        elif self.state == "LISTENING":
            txt, col = "LISTENING", QColor("#00ff88")
        else:
            txt, col = self.state, QColor("#5ab8cc")
        p.setPen(col)
        p.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        p.drawText(0, int(cy + R * 1.38), W, 22, Qt.AlignmentFlag.AlignCenter,
                   f"\u25cf  {txt}")
        p.end()


class TranscriptWidget(QTextEdit):
    """Live transcript. feed("you", "hello ") / feed("ai", "Hi sir") append text
    to the current line while the same speaker keeps talking, and start a new
    labelled line when the speaker changes."""

    def __init__(self, ai_name: str = "JARVIS", you_color="#d8f8ff",
                 ai_color="#00d4ff", bg="#010d14", border="#0d3347", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.document().setMaximumBlockCount(300)
        self.setFont(QFont("Courier New", 9))
        self.setStyleSheet(
            f"QTextEdit {{ background: {bg}; color: {you_color}; "
            f"border: 1px solid {border}; border-radius: 4px; padding: 6px; }}"
        )
        self._ai_name = ai_name.upper()
        self._colors = {"you": QColor(you_color), "ai": QColor(ai_color)}
        self._role: str | None = None
        self._live_start: int | None = None     # start of the in-progress (partial) text
        self._live_cap = False

    def set_ai_name(self, name: str) -> None:
        self._ai_name = name.upper()

    def _open_role(self, cur, role: str) -> None:
        """Start a new labelled line if the speaker changed."""
        if role != self._role:
            if self._role is not None:
                cur.insertBlock()
            self._role = role
            lab = QTextCharFormat()
            lab.setForeground(self._colors[role])
            lab.setFontWeight(QFont.Weight.Bold)
            cur.insertText(("YOU" if role == "you" else self._ai_name) + "  ", lab)

    def live(self, role: str, text: str, final: bool) -> None:
        """Streaming text that is rewritten in place until it is final."""
        role = "you" if role == "you" else "ai"
        text = (text or "").strip()
        if not text:
            if final:
                self._live_start = None
            return
        cur = self.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        if self._live_start is None or role != self._role:
            self._live_cap = role != self._role        # capitalise only at the start of a line
            self._open_role(cur, role)
            self._live_start = cur.position()
        else:
            cur.setPosition(self._live_start)
            cur.movePosition(cur.MoveOperation.End, cur.MoveMode.KeepAnchor)
            cur.removeSelectedText()
        if self._live_cap:
            text = text[0].upper() + text[1:]
        body = QTextCharFormat()
        body.setForeground(self._colors[role])
        body.setFontWeight(QFont.Weight.Normal)
        cur.insertText(text + (" " if final else ""), body)
        if final:
            self._live_start = None
        self.setTextCursor(cur)
        self.ensureCursorVisible()

    def feed(self, role: str, text: str) -> None:
        self._live_start = None
        if role == "end":
            self.end_turn()
            return
        role = "you" if role == "you" else "ai"
        if not text:
            return
        cur = self.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        if role != self._role:
            if self._role is not None:
                cur.insertBlock()
            self._role = role
            lab = QTextCharFormat()
            lab.setForeground(self._colors[role])
            lab.setFontWeight(QFont.Weight.Bold)
            cur.insertText(("YOU" if role == "you" else self._ai_name) + "  ", lab)
            text = text.lstrip()
        body = QTextCharFormat()
        body.setForeground(self._colors[role])
        body.setFontWeight(QFont.Weight.Normal)
        cur.insertText(text, body)
        self.setTextCursor(cur)
        self.ensureCursorVisible()

    def end_turn(self) -> None:
        if self._role is not None:
            self._role = "end"       # forces a new line for the next speaker