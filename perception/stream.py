"""Webcam perception loop producing engagement buckets (OpenFace pipeline)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .camera import WebcamStream
from .detector import HaarFaceDetector, MediaPipeFaceDetector
from .engagement import EngagementBucketMapper, softmax


def crop_face(frame, box, margin=0.2):
    """Crop face with optional margin; returns (crop, new_box)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    w *= 1 + margin
    h *= 1 + margin
    nx1 = int(max(cx - w / 2, 0))
    ny1 = int(max(cy - h / 2, 0))
    nx2 = int(min(cx + w / 2, frame.shape[1]))
    ny2 = int(min(cy + h / 2, frame.shape[0]))
    return frame[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)

def draw_overlay(frame: np.ndarray, box: Tuple[int, int, int, int], bucket: str, top_emotion: str, confidence: float) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{bucket} ({top_emotion}, {confidence:.2f})"
    cv2.putText(frame, text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Webcam affect -> engagement buckets.")
    parser.add_argument("--openface-weights", default="weights/MTL_backbone.pth", help="OpenFace multitask weights (backend=openface).")
    parser.add_argument("--openface-classifier", default="models/openface_emotion_clf.pkl", help="Trained OpenFace classifier (backend=openface).")
    parser.add_argument("--openface-features", choices=["emotion+au", "au", "emotion"], default="emotion+au", help="Feature mode for OpenFace classifier.")
    parser.add_argument("--openface-device", default="cpu", help="Device for OpenFace (cpu or cuda:0).")
    parser.add_argument("--labels", nargs="+", required=True, help="Label order for the model outputs.")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--display", action="store_true", help="Show live overlay window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until interrupted).")
    parser.add_argument("--verbose", action="store_true", help="Print top-3 emotion probabilities for debugging.")
    parser.add_argument("--crop-margin", type=float, default=0.2, help="Extra margin around face crop (0..1).")
    parser.add_argument("--low-threshold", type=float, default=None, help="Min score to enter 'low' bucket.")
    parser.add_argument("--high-threshold", type=float, default=None, help="Min score to enter 'high' bucket.")
    parser.add_argument("--margin", type=float, default=None, help="Min margin between top-1 and top-2 emotions.")
    parser.add_argument(
        "--bucket-mode",
        choices=["confidence", "emotion"],
        default="confidence",
        help="How to map emotions to buckets (confidence=thresholds+margin, emotion=top-emotion groups).",
    )
    parser.add_argument("--guard-seconds", type=float, default=0.8, help="Seconds candidate must persist before switching.")
    parser.add_argument("--alpha", type=float, default=0.6, help="EMA smoothing factor (higher = more reactive).")
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=None,
        help="Optional: treat rapid changes in the model output as 'high' engagement (demo-friendly).",
    )
    parser.add_argument(
        "--activity-alpha",
        type=float,
        default=0.5,
        help="Smoothing factor for activity (higher = more reactive).",
    )
    args = parser.parse_args(argv)

    from .openface_classifier import OpenFaceEmotionClassifier

    clf = OpenFaceEmotionClassifier(
        weights_path=args.openface_weights,
        classifier_path=args.openface_classifier,
        labels=args.labels,
        device=args.openface_device,
        features=args.openface_features,
    )

    def predict_logits(face_bgr: np.ndarray) -> Dict[str, float]:
        pred = clf.predict_logits(face_bgr)
        return {label: float(pred.logits[i]) for i, label in enumerate(pred.labels)}
    # Use MediaPipe if available, otherwise Haar.
    try:
        detector = MediaPipeFaceDetector()
    except Exception:
        detector = HaarFaceDetector()
    mapper_kwargs = {
        "labels": args.labels,
        "guard_seconds": args.guard_seconds,
        "alpha": args.alpha,
        "bucket_mode": args.bucket_mode,
    }
    if args.low_threshold is not None:
        mapper_kwargs["low_threshold"] = args.low_threshold
    if args.high_threshold is not None:
        mapper_kwargs["high_threshold"] = args.high_threshold
    if args.margin is not None:
        mapper_kwargs["margin"] = args.margin
    if args.activity_threshold is not None:
        mapper_kwargs["activity_threshold"] = args.activity_threshold
    mapper_kwargs["activity_alpha"] = args.activity_alpha
    mapper = EngagementBucketMapper(**mapper_kwargs)

    last_box: Optional[Tuple[int, int, int, int]] = None
    frame_count = 0

    with WebcamStream(device_index=args.device) as stream:
        for frame in stream.frames():
            frame_count += 1
            faces = detector.detect(frame.image)
            if faces:
                last_box = faces[0]
            elif last_box is None:
                continue  # wait for a face

            box = last_box
            face_crop, box = crop_face(frame.image, box, margin=args.crop_margin)
            if face_crop.size == 0:
                continue

            t0 = time.time()
            logits = predict_logits(face_crop)
            signal = mapper.update(logits, timestamp=frame.timestamp)
            latency_ms = (time.time() - t0) * 1000.0
            bucket_changed = False
            if not hasattr(main, "_last_bucket"):
                setattr(main, "_last_bucket", None)
            last_bucket = getattr(main, "_last_bucket")
            if last_bucket != signal.bucket:
                bucket_changed = True
                setattr(main, "_last_bucket", signal.bucket)

            if args.verbose:
                probs = softmax(logits, labels=args.labels)
                top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                sys.stderr.write(f"Top3: {top3}\n")

            out = {
                "timestamp": signal.timestamp,
                "bucket": signal.bucket,
                "candidate_bucket": signal.candidate_bucket,
                "top_emotion": signal.top_emotion,
                "confidence": signal.confidence,
                "activity": signal.activity,
                "bucket_changed": bucket_changed,
                "latency_ms": latency_ms,
            }
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()

            if args.display:
                draw_overlay(frame.image, box, signal.bucket, signal.top_emotion, signal.confidence)
                cv2.imshow("engagement", frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
