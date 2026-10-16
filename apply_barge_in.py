"""
Adds voice barge-in (interrupt JARVIS by talking over it) to Mark-LIII's main.py.

Usage (from the Mark-LIII-main folder, with your venv active):
    python apply_barge_in.py            # patches ./main.py
    python apply_barge_in.py path/to/main.py

- Makes a backup first: main.py.bak
- Safe: if any anchor line is not found exactly once, nothing is changed.
- Safe to re-run: it detects an already-patched file and exits.
"""
import re
import shutil
import sys
import textwrap
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "main.py")
if not target.exists():
    sys.exit(f"File not found: {target}")

src = target.read_text(encoding="utf-8")
if "BARGE_IN_ENABLED" in src:
    print("Already patched - nothing to do.")
    sys.exit(0)


def patch(text, pattern, make_replacement, label):
    new, n = re.subn(pattern, make_replacement, text, flags=re.M)
    if n != 1:
        sys.exit(f"Could not patch '{label}': expected 1 match, found {n}. No changes made.")
    return new


# 1. Settings constants -------------------------------------------------------
CONSTS = '''
# ── Voice barge-in: talk over JARVIS to stop it ──────────────────────────────
BARGE_IN_ENABLED = True
BARGE_IN_RMS     = 1800.0   # raw int16 loudness; raise this if JARVIS interrupts itself
BARGE_IN_BLOCKS  = 4        # consecutive loud mic blocks (~64 ms each => ~250 ms)'''

src = patch(src, r"^CHUNK_SIZE\s*=\s*1024[^\n]*$",
            lambda m: m.group(0) + "\n" + CONSTS, "constants")

# 2. Import -------------------------------------------------------------------
src = patch(src, r"^from pathlib import Path[^\n]*$",
            lambda m: m.group(0) + "\nfrom collections import deque", "import")

# 3. State in __init__ --------------------------------------------------------
INIT = '''self._barge_hits = 0
self._barge_buf  = deque(maxlen=8)   # recent mic blocks, replayed so your first words are not lost'''

src = patch(
    src,
    r"^([ \t]*)self\._interrupted\s*=\s*False\s*#\s*True while draining[^\n]*$",
    lambda m: m.group(0) + "\n" + textwrap.indent(INIT, m.group(1)),
    "__init__ state",
)

# 4. Mic callback: detect speech while JARVIS is talking ----------------------
CALLBACK = '''if jarvis_speaking and BARGE_IN_ENABLED and not self.ui.muted and not self._phone_active:
    self._barge_buf.append(indata.tobytes())
    rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
    self._barge_hits = self._barge_hits + 1 if rms > BARGE_IN_RMS else 0
    if self._barge_hits >= BARGE_IN_BLOCKS:
        self._barge_hits = 0
        pre = list(self._barge_buf)
        self._barge_buf.clear()

        def _barge_in():
            self.interrupt()
            for b in pre:   # replay what you already said so Gemini hears it
                try:
                    self.out_queue.put_nowait({"data": b, "mime_type": "audio/pcm"})
                except asyncio.QueueFull:
                    break
        loop.call_soon_threadsafe(_barge_in)
    return
self._barge_hits = 0
'''

src = patch(
    src,
    r"^[ \t]*if not jarvis_speaking and not self\.ui\.muted and not self\._phone_active:",
    lambda m: textwrap.indent(CALLBACK, re.match(r"[ \t]*", m.group(0)).group(0)) + m.group(0),
    "mic callback",
)

# 5. Honour the server's own 'interrupted' signal -----------------------------
RECV = '''if getattr(sc, "interrupted", False):
    self.interrupt()
    self._interrupted = False   # the next reply must play normally'''

src = patch(
    src,
    r"^([ \t]*)sc = response\.server_content[ \t]*$",
    lambda m: m.group(0) + "\n" + textwrap.indent(RECV, m.group(1)),
    "server interrupted signal",
)

# Write -----------------------------------------------------------------------
backup = target.with_name(target.name + ".bak")
shutil.copy2(target, backup)
target.write_text(src, encoding="utf-8")
print(f"Patched {target}  (backup: {backup})")
print("Tune BARGE_IN_RMS near the top of main.py if JARVIS interrupts itself.")