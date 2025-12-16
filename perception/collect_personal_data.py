"""Collect a small personal facial-expression dataset from webcam.

This is the fastest way to make the model respond to *your* face and lighting.
It saves cropped face images into class folders (ImageFolder-compatible).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .camera import WebcamStream
from .detector import HaarFaceDetector, MediaPipeFaceDetector
from .stream import crop_face


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect personal face images by label.")
    parser.add_argument("--out", type=Path, default=Path("data/personal"), help="Output root folder.")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional session name to prefix saved filenames (use different sessions for better generalization).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        help="Label folders to collect (created under --out).",
    )
    parser.add_argument("--per-label", type=int, default=120, help="How many images to collect per label.")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--margin", type=float, default=0.2, help="Crop margin around detected face.")
    parser.add_argument("--min-interval-ms", type=int, default=120, help="Min time between saved frames.")
    parser.add_argument(
        "--burst",
        type=int,
        default=1,
        help="How many images to save per SPACE press (use 5-10 for faster collection).",
    )
    args = parser.parse_args(argv)

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    for lab in args.labels:
        (out_root / lab).mkdir(parents=True, exist_ok=True)

    try:
        detector = MediaPipeFaceDetector()
    except Exception:
        detector = HaarFaceDetector()

    print("Controls:")
    print(" - SPACE: save one image for current label")
    print(" - N: next label")
    print(" - P: previous label")
    print(" - Q: quit")
    if args.session:
        print(f"Session: {args.session}")

    label_index = 0
    last_box = None
    last_save_t = 0.0
    session_prefix = (args.session or time.strftime("%Y%m%d_%H%M%S")).strip()

    with WebcamStream(device_index=args.device) as stream:
        for frame in stream.frames():
            faces = detector.detect(frame.image)
            if faces:
                last_box = faces[0]
            if last_box is None:
                cv2.imshow("collect", frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            box = last_box
            face_crop, box = crop_face(frame.image, box, margin=args.margin)
            if face_crop.size == 0:
                continue

            label = args.labels[label_index]
            label_dir = out_root / label
            count = len(list(label_dir.glob("*.jpg")))

            view = frame.image.copy()
            x1, y1, x2, y2 = box
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                view,
                f"label={label}  {count}/{args.per_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                view,
                "SPACE=save  N/P=label  Q=quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("collect", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                label_index = min(label_index + 1, len(args.labels) - 1)
            if key == ord("p"):
                label_index = max(label_index - 1, 0)
            if key == ord(" "):
                now = time.time()
                if (now - last_save_t) * 1000.0 < args.min_interval_ms:
                    continue
                if count >= args.per_label:
                    continue
                # Save a short burst to capture subtle variations.
                for i in range(int(max(1, args.burst))):
                    now_i = time.time()
                    out_path = label_dir / f"{session_prefix}_{int(now_i * 1000)}_{i}.jpg"
                    cv2.imwrite(str(out_path), face_crop)
                    count += 1
                    if count >= args.per_label:
                        break
                last_save_t = now

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
