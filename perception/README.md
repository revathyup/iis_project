Perception Subsystem
====================

Purpose
-------
- Webcam -> face detection -> DiffusionFER emotion head -> smoothing + engagement buckets (`low`/`medium`/`high`), with jitter guard and logging.
- Outputs drive the dialogue manager via simple events: `{bucket, top_emotion, confidence, timestamp}`.

Pipeline Sketch
---------------
- Capture: OpenCV/MediaPipe capture loop (CPU acceptable for baseline).
- Detect: MediaPipe/RetinaFace face crop with fallback to last good bbox.
- Classify: DiffusionFER (pretrained) -> logits -> softmax.
- Smooth/bucket: exponential moving average + persistence guard (see `engagement.py`), then bucket by confidence bands.
- Export: ONNX and runtime stub for lightweight inference; log per-frame latency.

Immediate Tasks
---------------
- [ ] Wire webcam capture + face detector (baseline).
- [ ] Hook pretrained DiffusionFER head; run inference; save logits.
- [ ] Integrate `EngagementBucketMapper` and log `{bucket, top_emotion, confidence, latency_ms}`.
- [ ] Write small held-out evaluation script for macro-F1 over buckets.
- [ ] Add ONNX export path for the emotion head (stretch if time).

Interfaces
----------
- `EngagementBucketMapper.update(emotion_logits, timestamp=None)` -> `EngagementSignal`.
- Downstream listeners subscribe to bucket events; keep the output schema stable for integration.
