"""OpenFace feature extraction + lightweight emotion classifier (7 classes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class OpenFacePrediction:
    logits: np.ndarray  # shape: (C,)
    labels: tuple[str, ...]


class OpenFaceEmotionClassifier:
    """Predict 7-emotion logits using OpenFace multitask features + a trained classifier."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        classifier_path: str | Path,
        labels: Sequence[str],
        device: str = "cpu",
        features: str = "emotion+au",
    ) -> None:
        self.labels = tuple(labels)
        self.features = features

        from openface.multitask_model import MultitaskPredictor

        self._predictor = MultitaskPredictor(str(weights_path), device=device)

        try:
            import joblib  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("joblib is required (pip install joblib).") from exc

        payload = joblib.load(str(classifier_path))
        if isinstance(payload, dict) and "model" in payload:
            self._clf = payload["model"]
            file_labels = payload.get("labels")
            file_features = payload.get("features")
            if file_labels and tuple(file_labels) != self.labels:
                raise ValueError(f"Classifier label order mismatch. File has {tuple(file_labels)}, CLI has {self.labels}.")
            if file_features and str(file_features) != self.features:
                raise ValueError(f"Classifier feature mode mismatch. File has {file_features}, CLI has {self.features}.")
        else:
            self._clf = payload

    def _featurize(self, face_bgr: np.ndarray) -> np.ndarray:
        emo, _gaze, au = self._predictor.predict(face_bgr)
        emo = emo.detach().cpu().numpy().reshape(-1)
        au = au.detach().cpu().numpy().reshape(-1)
        if self.features == "au":
            return au
        if self.features == "emotion":
            return emo
        return np.concatenate([emo, au], axis=0)

    def predict_logits(self, face_bgr: np.ndarray) -> OpenFacePrediction:
        x = self._featurize(face_bgr).reshape(1, -1)
        # Prefer decision_function (true logits), fallback to log-probabilities.
        if hasattr(self._clf, "decision_function"):
            logits = np.asarray(self._clf.decision_function(x)).reshape(-1)
        else:
            probs = np.asarray(self._clf.predict_proba(x)).reshape(-1)
            logits = np.log(np.clip(probs, 1e-9, 1.0))
        if logits.shape[0] != len(self.labels):
            raise ValueError(f"Classifier outputs {logits.shape[0]} classes, expected {len(self.labels)}.")
        return OpenFacePrediction(logits=logits, labels=self.labels)
