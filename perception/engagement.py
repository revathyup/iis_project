"""Bucket mapping and smoothing for affect signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math
import time

# Default label set for MultiEmoVA. Extend if your model differs.
# Mapping used:
# - low: HighNegative, MediumNegative, LowNegative
# - medium: neutral
# - high: MediumPositive, HighPositive
DEFAULT_EMOTIONS = (
    "HighNegative",
    "MediumNegative",
    "LowNegative",
    "neutral",
    "MediumPositive",
    "HighPositive",
)


@dataclass
class EngagementSignal:
    bucket: str
    top_emotion: str
    confidence: float
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
        low_threshold: float = 0.4,
        high_threshold: float = 0.7,
        guard_seconds: float = 0.8,
        alpha: float = 0.6,
        labels: Iterable[str] = DEFAULT_EMOTIONS,
        margin: float = 0.1,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.guard_seconds = guard_seconds
        self.alpha = alpha
        self.labels = tuple(labels)
        self.margin = margin

        self._smoothed: Dict[str, float] = {}
        self._last_bucket: str = "medium"
        self._pending_bucket: Optional[str] = None
        self._pending_since: Optional[float] = None

    def reset(self) -> None:
        self._smoothed = {}
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

    def _bucket_from_probs(self, probs: Dict[str, float]) -> str:
        """Bucket selection: only switch to low/high when that category is top and confident."""
        low_emotions = ("HighNegative", "MediumNegative", "LowNegative")
        high_emotions = ("MediumPositive", "HighPositive")
        # Find top and second-best
        sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_emotion, top_score = sorted_probs[0]
        second_score = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

        # Require margin over second-best to avoid jittery mislabels
        if top_emotion in low_emotions and top_score >= self.low_threshold and (top_score - second_score) >= self.margin:
            return "low"
        if top_emotion in high_emotions and top_score >= self.high_threshold and (top_score - second_score) >= self.margin:
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
        candidate_bucket = self._bucket_from_probs(smoothed)
        bucket = self._apply_guard(candidate_bucket, ts)

        low_score = (
            smoothed.get("angry", 0.0)
            + smoothed.get("disgust", 0.0)
            + smoothed.get("fear", 0.0)
            + smoothed.get("sad", 0.0)
        )
        high_score = smoothed.get("happy", 0.0) + smoothed.get("surprise", 0.0)
        top_emotion = max(smoothed, key=smoothed.get)
        top_score = smoothed[top_emotion]

        if bucket == "low":
            confidence = top_score
        elif bucket == "high":
            confidence = top_score
        else:
            # neutral-ish confidence: how far from extremes
            confidence = max(0.0, 1.0 - max(low_score, high_score))

        return EngagementSignal(bucket=bucket, top_emotion=top_emotion, confidence=float(confidence), timestamp=ts)
