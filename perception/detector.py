"""Face detection helper."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# Optional MediaPipe; fallback to Haar if not installed.
try:  # pragma: no cover
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover
    _HAS_MEDIAPIPE = False


class HaarFaceDetector:
    """Haar cascade detector; baseline, no extra deps."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors)
        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]


class MediaPipeFaceDetector:
    """MediaPipe face detector; more robust, requires mediapipe install."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0) -> None:
        if not _HAS_MEDIAPIPE:
            raise ImportError("mediapipe is not installed")
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_confidence
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w, _ = frame_bgr.shape
        results = self.detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes: List[Tuple[int, int, int, int]] = []
        if results.detections:
            for det in results.detections:
                rel = det.location_data.relative_bounding_box
                x1 = int(rel.xmin * w)
                y1 = int(rel.ymin * h)
                x2 = int((rel.xmin + rel.width) * w)
                y2 = int((rel.ymin + rel.height) * h)
                # Clamp to image bounds (MediaPipe boxes can be slightly outside the frame).
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes
