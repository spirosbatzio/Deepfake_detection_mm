from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src import config


def wav_path_for(row) -> str:
    stem = f"{row.type}__{row.speaker_id}__{row.filename}".replace(".mp4", "")
    return str(config.AUDIO_DIR / f"{stem}.wav")


def extract_one(args) -> tuple[str, bool, str]:
    src_mp4, dst_wav = args
    from pathlib import Path
    if Path(dst_wav).exists():
        return dst_wav, True, "cached"
    cmd = [
        config.FFMPEG, "-y", "-i", src_mp4,
        "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE),
        dst_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return dst_wav, False, proc.stderr.strip().splitlines()[-1] if proc.stderr else "ffmpeg error"
    return dst_wav, True, "ok"


def select_clips(df: pd.DataFrame, per_speaker: int | None, limit: int | None) -> pd.DataFrame:
    if per_speaker is not None:
        df = (
            df.groupby(["speaker_id", "label"], group_keys=False)
              .head(per_speaker)
        )
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--per-speaker", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    df = pd.concat(
        [pd.read_csv(config.SPLITS_DIR / f"{s}.csv") for s in splits],
        ignore_index=True,
    )
    df = select_clips(df, args.per_speaker, args.limit)

    jobs = [(row.clip_path, wav_path_for(row)) for row in df.itertuples()]
    print(f"Extracting {len(jobs)} clips from split(s)={splits} "
          f"(per_speaker={args.per_speaker}, limit={args.limit}) "
          f"with {args.workers} workers...")

    ok = failed = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(extract_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            dst, success, msg = fut.result()
            if success:
                ok += 1
            else:
                failed += 1
                errors.append((dst, msg))

    print(f"\nDone. ok={ok}  failed={failed}")
    for dst, msg in errors[:10]:
        print(f"  FAIL {dst}: {msg}")


if __name__ == "__main__":
    main()
