"""Screen capture.

Deliberately *not* a video feed. Continuous frames cost a fortune on a cloud vision
model and add nothing on a screen that is static most of the time, so capture happens
on demand — when something the user said is actually about the screen — and a change
gate stops repeat questions re-sending an identical image.

Privacy shapes the design as much as cost: capture is off until switched on, every
capture is announced to the overlay, and nothing is written to disk.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass

import mss
import numpy as np
from PIL import Image

from ..config import VisionConfig

log = logging.getLogger(__name__)


@dataclass
class Screenshot:
    data: bytes
    width: int
    height: int
    monitor: int
    captured_at: float

    @property
    def kilobytes(self) -> float:
        return len(self.data) / 1024


def list_monitors() -> list[dict]:
    """Monitor 0 is every screen stitched together; 1..n are the individual ones."""
    with mss.MSS() as sct:
        return [dict(m) for m in sct.monitors]


class ScreenCapture:
    """Grabs, downscales and encodes a screenshot. Blocking — call it in a thread."""

    def __init__(self, cfg: VisionConfig):
        self._cfg = cfg
        self._last_thumb: np.ndarray | None = None

    def capture(self, monitor: int | None = None) -> Screenshot:
        index = self._cfg.monitor if monitor is None else monitor
        # An MSS instance belongs to the thread that made it, and this runs on the
        # asyncio worker pool where that thread is not guaranteed to be the same one
        # twice. Creating one per grab is cheap and avoids the whole problem.
        with mss.MSS() as sct:
            if index >= len(sct.monitors):
                log.warning("monitor %d does not exist, using primary", index)
                index = 1
            raw = sct.grab(sct.monitors[index])

        image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        image = self._downscale(image)
        self._last_thumb = _thumbprint(image)

        buffer = io.BytesIO()
        # PNG, not JPEG: the entire job is reading small text, and JPEG ringing around
        # 10px glyphs is exactly the artefact that turns a legible error into a guess.
        image.save(buffer, format="PNG", optimize=True)

        return Screenshot(
            data=buffer.getvalue(),
            width=image.width,
            height=image.height,
            monitor=index,
            captured_at=time.perf_counter(),
        )

    def _downscale(self, image: Image.Image) -> Image.Image:
        longest = max(image.width, image.height)
        if longest <= self._cfg.edge:
            return image
        scale = self._cfg.edge / longest
        return image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.LANCZOS,  # keeps text edges crisp; BILINEAR turns small glyphs to mush
        )

    def has_changed(self, monitor: int | None = None) -> bool:
        """Cheap check for whether the screen differs from the last capture.

        Used by the throttled path so a screen nobody has touched is not re-sent. Grabs
        at thumbnail scale, so it costs a few milliseconds rather than a full encode.
        """
        index = self._cfg.monitor if monitor is None else monitor
        with mss.MSS() as sct:
            index = min(index, len(sct.monitors) - 1)
            raw = sct.grab(sct.monitors[index])
        thumb = _thumbprint(Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"))

        if self._last_thumb is None:
            return True
        difference = float(np.abs(thumb - self._last_thumb).mean()) / 255.0
        return difference >= self._cfg.change_threshold


def _thumbprint(image: Image.Image) -> np.ndarray:
    """A 64x64 greyscale fingerprint — enough to notice a window change, cheap enough
    to run often, and blind to cursor movement and blinking carets."""
    small = image.convert("L").resize((64, 64), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32)
