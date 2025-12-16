"""Bucket mapping and smoothing for affect signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math
import time

# Default label set for MultiEmoVA.
DEFAULT_EMOTIONS = ("HighNegative", "MediumNegative", "LowNegative", "neutral", "MediumPositive", "HighPositive")

# Common label set for DiffusionFER (7-class).
DIFFUSIONFER_EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


@dataclass
class EngagementSignal:
    bucket: str
    candidate_bucket: str
    top_emotion: str
    confidence: float
    activity: float
    timestamp: float


def softmax(logits: Dict[str, float], labels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Numerically stable softmax over provided labels (or keys of logits)."""
    keys = list(labels) if labels is not None else list(logits.keys())
    if not keys:
        raise ValueError("emotion_logits is empty; cannot compute softmax.")

    values = {k: float(logits.get(k, 0.0)) for k in keys}
    max_logit = max(values.values())
    exps = {k: math.exp(v - max_logit) for k, v in values.items()}
    denom = sum(exps.values())
    if denom <= 0.0:
        # Fallback to uniform if everything underflows.
        uniform = 1.0 / len(exps)
        return {k: uniform for k in exps}
    return {k: exps[k] / denom for k in exps}


class EngagementBucketMapper:
    """Maps emotion logits to smoothed engagement buckets with persistence guard."""

    def __init__(
        self,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        bucket_mode: str = "confidence",
        guard_seconds: float = 0.8,
        alpha: float = 0.6,
        labels: Iterable[str] = DEFAULT_EMOTIONS,
        margin: Optional[float] = None,
        activity_threshold: Optional[float] = None,
        activity_alpha: float = 0.5,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 < activity_alpha <= 1.0:
            raise ValueError("activity_alpha must be in (0, 1].")
        if bucket_mode not in {"confidence", "emotion"}:
            raise ValueError("bucket_mode must be 'confidence' or 'emotion'.")
        self.labels = tuple(labels)
        self.bucket_mode = bucket_mode

        # If the user is running DiffusionFER labels, apply more suitable defaults unless overridden.
        if set(self.labels) == set(DIFFUSIONFER_EMOTIONS):
            # DiffusionFER outputs tend to be noisy on webcam; we bucket by valence-like groups
            # with stricter thresholds and a small margin to avoid jitter.
            self.low_threshold = 0.55 if low_threshold is None else float(low_threshold)
            self.high_threshold = 0.55 if high_threshold is None else float(high_threshold)
            self.margin = 0.08 if margin is None else float(margin)
            # Demo-friendly option: allow "high engagement" when the model output changes a lot.
            # This is useful for noisy webcam emotion models, but it can also cause unexpected
            # "high" spikes even when the user stays neutral. Therefore:
            # - default to OFF for `bucket_mode="emotion"`
            # - default to ON for `bucket_mode="confidence"`
            if activity_threshold is None:
                self.activity_threshold = 0.0 if bucket_mode == "emotion" else 0.10
            else:
                self.activity_threshold = float(activity_threshold)
        else:
            self.low_threshold = 0.4 if low_threshold is None else float(low_threshold)
            self.high_threshold = 0.7 if high_threshold is None else float(high_threshold)
            self.margin = 0.1 if margin is None else float(margin)
            self.activity_threshold = 0.0 if activity_threshold is None else float(activity_threshold)

        self.guard_seconds = guard_seconds
        self.alpha = alpha
        self.activity_alpha = activity_alpha

        # Infer which emotions correspond to low/high engagement for this label set.
        self._low_emotions, self._high_emotions = self._infer_groups(self.labels)

        self._smoothed: Dict[str, float] = {}
        self._prev_smoothed: Optional[Dict[str, float]] = None
        self._activity_ema: float = 0.0
        self._last_candidate: str = "medium"
        self._last_bucket: str = "medium"
        self._pending_bucket: Optional[str] = None
        self._pending_since: Optional[float] = None

    def reset(self) -> None:
        self._smoothed = {}
        self._prev_smoothed = None
        self._activity_ema = 0.0
        self._last_candidate = "medium"
        self._last_bucket = "medium"
        self._pending_bucket = None
        self._pending_since = None

    def _ema(self, probs: Dict[str, float]) -> Dict[str, float]:
        if not self._smoothed:
            self._smoothed = probs
        else:
            updated = {}
            for k in self.labels:
                prev = self._smoothed.get(k, 0.0)
                curr = probs.get(k, 0.0)
                updated[k] = self.alpha * curr + (1.0 - self.alpha) * prev
            # Renormalize to stay on the simplex.
            total = sum(updated.values()) or 1.0
            updated = {k: v / total for k, v in updated.items()}
            self._smoothed = updated
        return self._smoothed

    def _update_activity(self, smoothed: Dict[str, float]) -> float:
        """Track how much the probability distribution is changing over time."""
        if self._prev_smoothed is None:
            self._prev_smoothed = dict(smoothed)
            self._activity_ema = 0.0
            return self._activity_ema

        # L1 distance between consecutive smoothed distributions.
        dist = 0.0
        for k in self.labels:
            dist += abs(smoothed.get(k, 0.0) - self._prev_smoothed.get(k, 0.0))
        self._prev_smoothed = dict(smoothed)

        self._activity_ema = self.activity_alpha * dist + (1.0 - self.activity_alpha) * self._activity_ema
        return self._activity_ema

    @staticmethod
    def _infer_groups(labels: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Infer low/high emotion groups based on the label vocabulary."""
        label_set = set(labels)

        # MultiEmoVA-style label set.
        if {"HighNegative", "MediumNegative", "LowNegative", "MediumPositive", "HighPositive", "neutral"} <= label_set:
            low_emotions = ("HighNegative", "MediumNegative", "LowNegative")
            high_emotions = ("MediumPositive", "HighPositive")
            return low_emotions, high_emotions

        # DiffusionFER-style label set.
        if {"angry", "disgust", "fear", "sad", "neutral", "happy", "surprise"} <= label_set:
            # Treat negative affect as low engagement.
            low_emotions = ("angry", "disgust", "fear", "sad")
            high_emotions = ("happy", "surprise")
            return low_emotions, high_emotions

        # Unknown label set: don't force low/high; everything stays medium unless the user customizes.
        return tuple(), tuple()

    def _bucket_from_probs(self, probs: Dict[str, float]) -> str:
        """Bucket selection: either by confidence+margin, or by top-emotion group."""
        low_emotions = self._low_emotions
        high_emotions = self._high_emotions
        # Find top and second-best
        sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_emotion, top_score = sorted_probs[0]
        second_score = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

        if self.bucket_mode == "emotion":
            if low_emotions and top_emotion in low_emotions and top_score >= self.low_threshold:
                return "low"
            if high_emotions and top_emotion in high_emotions and top_score >= self.high_threshold:
                return "high"
            if self.activity_threshold > 0.0 and self._activity_ema >= self.activity_threshold:
                return "high"
            return "medium"

        # Require margin over second-best to avoid jittery mislabels
        if low_emotions and top_emotion in low_emotions and top_score >= self.low_threshold and (top_score - second_score) >= self.margin:
            return "low"
        if high_emotions and top_emotion in high_emotions and top_score >= self.high_threshold and (top_score - second_score) >= self.margin:
            return "high"

        # If label-based high doesn't trigger (common on webcam), allow "high engagement" when
        # the model output is changing significantly (user reacting/expressing).
        if self.activity_threshold > 0.0 and self._activity_ema >= self.activity_threshold:
            return "high"
        return "medium"

    def _apply_guard(self, candidate: str, timestamp: float) -> str:
        if candidate == self._last_bucket:
            self._pending_bucket = None
            self._pending_since = None
            return self._last_bucket

        if self._pending_bucket != candidate:
            self._pending_bucket = candidate
            self._pending_since = timestamp
            return self._last_bucket

        if self._pending_since is not None and (timestamp - self._pending_since) >= self.guard_seconds:
            self._last_bucket = candidate
            self._pending_bucket = None
            self._pending_since = None
        return self._last_bucket

    def update(self, emotion_logits: Dict[str, float], timestamp: Optional[float] = None) -> EngagementSignal:
        """Convert raw logits to a stable engagement bucket."""
        ts = float(timestamp if timestamp is not None else time.time())
        probs = softmax(emotion_logits, self.labels)
        smoothed = self._ema(probs)
        activity = self._update_activity(smoothed)
        candidate_bucket = self._bucket_from_probs(smoothed)
        self._last_candidate = candidate_bucket
        bucket = self._apply_guard(candidate_bucket, ts)

        low_score = sum(smoothed.get(k, 0.0) for k in self._low_emotions) if self._low_emotions else 0.0
        high_score = sum(smoothed.get(k, 0.0) for k in self._high_emotions) if self._high_emotions else 0.0
        top_emotion = max(smoothed, key=smoothed.get)
        top_score = smoothed[top_emotion]

        if bucket == "low":
            confidence = top_score
        elif bucket == "high":
            confidence = top_score
        else:
            # neutral-ish confidence: how far from extremes
            confidence = max(0.0, 1.0 - max(low_score, high_score))

        return EngagementSignal(
            bucket=bucket,
            candidate_bucket=candidate_bucket,
            top_emotion=top_emotion,
            confidence=float(confidence),
            activity=float(activity),
            timestamp=ts,
        )
