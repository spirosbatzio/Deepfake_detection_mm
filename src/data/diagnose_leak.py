from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

from src import config
from src.data.extract_audio import wav_path_for


def load(split, splits_dir):
    d = pd.read_csv(splits_dir / f"{split}.csv")
    d["wav"] = [wav_path_for(r) for r in d.itertuples()]
    d["split"] = split
    return d[d["wav"].map(lambda p: Path(p).exists())].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=str(config.SPLITS_DIR))
    args = ap.parse_args()
    splits_dir = Path(args.splits_dir)
    print(f"Diagnosing splits in: {splits_dir}\n")

    train, test = load("train", splits_dir), load("test", splits_dir)

    # 1) speaker overlap
    st, se = set(train["speaker_id"]), set(test["speaker_id"])
    print("=== 1. Speaker overlap (train ∩ test) ===")
    print(f"  train speakers={len(st)}  test speakers={len(se)}  shared={len(st & se)}")

    # 2) content hash of raw audio bytes across splits
    print("\n=== 2. Identical audio content across splits ===")
    alldf = pd.concat([train, test], ignore_index=True)

    def md5(p):
        data, _ = sf.read(p, dtype="float32")
        return hashlib.md5(data.tobytes()).hexdigest()

    alldf["h"] = alldf["wav"].map(md5)
    dup = alldf.groupby("h")["split"].nunique()
    cross = dup[dup > 1].index
    print(f"  distinct audio hashes={alldf['h'].nunique()} of {len(alldf)} clips")
    print(f"  hashes appearing in BOTH train and test: {len(cross)}")
    if len(cross):
        ex = alldf[alldf["h"].isin(cross)].sort_values("h").head(6)
        print(ex[["split", "type", "speaker_id", "filename"]].to_string(index=False))

    # also: exact duplicate audio anywhere (reused tracks)
    dup_any = alldf["h"].value_counts()
    print(f"  any exact-duplicate audio (count>1): "
          f"{(dup_any > 1).sum()} groups, "
          f"{int(dup_any[dup_any > 1].sum())} clips involved")

    # 3) channel confound: duration by label
    print("\n=== 3. Duration (s) by label ===")
    def dur(p):
        i = sf.info(p); return i.frames / i.samplerate
    samp = alldf.sample(min(2000, len(alldf)), random_state=0).copy()
    samp["dur"] = samp["wav"].map(dur)
    print(samp.groupby("label")["dur"].describe()[["mean", "min", "25%", "50%", "75%", "max"]])


if __name__ == "__main__":
    main()
