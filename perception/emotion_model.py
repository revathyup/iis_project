"""Emotion model wrapper (ONNXRuntime if available)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None


class EmotionModel:
    """Wraps an ONNX emotion classifier. Expects a single input tensor."""

    def __init__(self, model_path: str, labels: Iterable[str]) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required to load ONNX models.")
        self.labels: List[str] = list(labels)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Model has no inputs.")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize to model input size and normalize to 0-1 float32."""
        import cv2

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Determine target H,W from model input (assumes NCHW)
        h = 224
        w = 224
        if len(self.input_shape) == 4:
            if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0:
                h = self.input_shape[2]
            if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0:
                w = self.input_shape[3]
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return np.expand_dims(arr, axis=0)

    def predict_logits(self, face_bgr: np.ndarray) -> Dict[str, float]:
        tensor = self.preprocess(face_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            raise RuntimeError("Model returned no outputs.")
        raw = outputs[0].squeeze()
        return {label: float(raw[idx]) for idx, label in enumerate(self.labels)}
