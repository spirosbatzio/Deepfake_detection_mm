from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src import config

URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
OOD = config.ROOT / "ood"
TARBALL = OOD / "dev-clean.tar.gz"
EXTRACT = OOD / "LibriSpeech"
DEST = OOD / "real_librispeech"


def download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"Already downloaded: {dst}")
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            while chunk := r.read(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="how many flac clips to sample")
    args = ap.parse_args()

    OOD.mkdir(exist_ok=True)
    download(URL, TARBALL)

    if not EXTRACT.exists():
        print("Extracting...")
        with tarfile.open(TARBALL) as t:
            t.extractall(OOD)

    flacs = list(EXTRACT.rglob("*.flac"))
    print(f"Found {len(flacs)} flac files")
    random.seed(0)
    sample = random.sample(flacs, min(args.n, len(flacs)))

    DEST.mkdir(exist_ok=True)
    for f in tqdm(sample, desc="copying sample"):
        shutil.copy(f, DEST / f.name)
    print(f"\nCopied {len(sample)} real clips -> {DEST}")
    print("Next: generate OOD fakes, then run src.eval_ood")


if __name__ == "__main__":
    main()
