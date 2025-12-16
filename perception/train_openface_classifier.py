"""Train a lightweight 7-emotion classifier from OpenFace multitask features.

This satisfies the course requirement of training/evaluating your own ML model,
while using OpenFace only as a feature extractor (2-stage pipeline).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


EMOTIONS_7 = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def iter_samples(root: Path, labels: Iterable[str]) -> list[Sample]:
    samples: list[Sample] = []
    for label in labels:
        folder = root / label
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append(Sample(path=p, label=label))
    return samples


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train AU-based emotion classifier using OpenFace features.")
    parser.add_argument("--data", required=True, help="Dataset root: subfolders per label (e.g., data/personal).")
    parser.add_argument("--weights", default="weights/MTL_backbone.pth", help="OpenFace multitask weights path.")
    parser.add_argument("--out", default="models/openface_emotion_clf.pkl", help="Output path for trained classifier.")
    parser.add_argument("--labels", nargs="+", default=list(EMOTIONS_7), help="Label order (default: 7 emotions).")
    parser.add_argument("--features", choices=["emotion+au", "au", "emotion"], default="emotion+au")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import joblib  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("scikit-learn + joblib are required: pip install scikit-learn joblib") from exc

    from openface.multitask_model import MultitaskPredictor

    labels = tuple(args.labels)
    data_root = Path(args.data)
    samples = iter_samples(data_root, labels)
    if not samples:
        raise SystemExit(f"No images found under {data_root} for labels: {labels}")

    print(f"Found {len(samples)} images in {data_root}")
    predictor = MultitaskPredictor(str(args.weights), device="cpu")

    x_list: list[np.ndarray] = []
    y_list: list[str] = []

    for s in samples:
        img = load_image(s.path)
        emo, _gaze, au = predictor.predict(img)
        emo = emo.detach().cpu().numpy().reshape(-1)
        au = au.detach().cpu().numpy().reshape(-1)
        if args.features == "au":
            feat = au
        elif args.features == "emotion":
            feat = emo
        else:
            feat = np.concatenate([emo, au], axis=0)
        x_list.append(feat.astype(np.float32))
        y_list.append(s.label)

    x = np.stack(x_list, axis=0)
    y = np.array(y_list)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=float(args.val_split), random_state=int(args.seed), stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_val)
    acc = accuracy_score(y_val, pred)
    cm = confusion_matrix(y_val, pred, labels=list(labels))
    print(f"Val accuracy: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred) in label order:")
    print(list(labels))
    print(cm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "labels": labels,
            "features": args.features,
            "model": clf,
        },
        out,
    )
    print(f"Saved classifier to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

