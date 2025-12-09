"""Webcam capture utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2


@dataclass
class Frame:
    image: "cv2.Mat"
    timestamp: float


class WebcamStream:
    """Lightweight webcam iterator using OpenCV."""

    def __init__(self, device_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "WebcamStream":
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.device_index}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def frames(self) -> Iterator[Frame]:
        if self.cap is None:
            raise RuntimeError("WebcamStream must be entered with context manager before use.")
        while True:
            ok, img = self.cap.read()
            if not ok:
                break
            yield Frame(image=img, timestamp=time.time())

