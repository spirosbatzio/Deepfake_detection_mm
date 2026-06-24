from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from src import config

OUT = config.ROOT / "ood" / "robust"
EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def split_dir(src: str, name: str, train_frac: float) -> None:
    files = sorted(f for f in Path(src).rglob("*") if f.suffix.lower() in EXTS)
    random.seed(0)
    random.shuffle(files)
    k = int(len(files) * train_frac)
    for split, subset in [("train", files[:k]), ("test", files[k:])]:
        dst = OUT / split / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in subset:
            shutil.copy(f, dst / f.name)
        print(f"  {name:5} {split:5}: {len(subset)} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="ood/real_librispeech")
    ap.add_argument("--fake-dir", default="ood/fake_xtts")
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()
    print(f"Splitting OOD generators (train_frac={args.train_frac}) -> {OUT}")
    split_dir(args.real_dir, "real", args.train_frac)
    split_dir(args.fake_dir, "fake", args.train_frac)
    print("\nTrain dirs feed src.robust.train_robust (--extra-*-dir);")
    print("Test dirs feed src.ood.evaluate for the held-out comparison.")


if __name__ == "__main__":
    main()
