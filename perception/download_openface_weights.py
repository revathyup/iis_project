"""Download OpenFace model weights (robust, resumable).

This uses Hugging Face Hub with resume support because the built-in OpenFace
downloader can fail on flaky connections on Windows.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO_ID = "nutPace/openface_weights"
REQUIRED_FILES = (
    "MTL_backbone.pth",
    "Alignment_RetinaFace.pth",
    "mobilenetV1X0.25_pretrain.tar",
)


def _all_present(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in REQUIRED_FILES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OpenFace weights to a local folder.")
    parser.add_argument("--out-dir", default="weights", help="Where to place weights (default: weights).")
    parser.add_argument("--retries", type=int, default=5, help="Retry attempts on transient download failures.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between retries.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _all_present(out_dir):
        print(f"OK: weights already present in {out_dir}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required for this downloader. Install with: pip install huggingface_hub"
        ) from exc

    patterns = list(REQUIRED_FILES) + [f"**/{name}" for name in REQUIRED_FILES]

    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"Downloading {REPO_ID} (attempt {attempt}/{args.retries}) ...")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=patterns,
                max_workers=1,
            )
            if _all_present(out_dir):
                print(f"Saved weights to {out_dir}")
                return 0
            raise RuntimeError(f"Download finished but files missing in {out_dir}: {REQUIRED_FILES}")
        except Exception as exc:
            last_exc = exc
            if attempt < args.retries:
                time.sleep(args.sleep)

    raise SystemExit(f"Failed to download OpenFace weights after {args.retries} attempts: {last_exc}")


if __name__ == "__main__":
    raise SystemExit(main())
