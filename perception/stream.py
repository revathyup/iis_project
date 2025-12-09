"""Webcam perception loop producing engagement buckets."""

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
from .emotion_model import EmotionModel
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
    parser.add_argument("--model", required=True, help="Path to ONNX emotion model.")
    parser.add_argument("--labels", nargs="+", required=True, help="Label order for the model outputs.")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--display", action="store_true", help="Show live overlay window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until interrupted).")
    parser.add_argument("--verbose", action="store_true", help="Print top-3 emotion probabilities for debugging.")
    args = parser.parse_args(argv)

    model = EmotionModel(args.model, labels=args.labels)
    # Use MediaPipe if available, otherwise Haar.
    try:
        detector = MediaPipeFaceDetector()
    except Exception:
        detector = HaarFaceDetector()
    mapper = EngagementBucketMapper(labels=args.labels)

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
            face_crop, box = crop_face(frame.image, box)
            if face_crop.size == 0:
                continue

            t0 = time.time()
            logits = model.predict_logits(face_crop)
            signal = mapper.update(logits, timestamp=frame.timestamp)
            latency_ms = (time.time() - t0) * 1000.0

            if args.verbose:
                probs = softmax(logits, labels=args.labels)
                top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                sys.stderr.write(f"Top3: {top3}\n")

            out = {
                "timestamp": signal.timestamp,
                "bucket": signal.bucket,
                "top_emotion": signal.top_emotion,
                "confidence": signal.confidence,
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
 
