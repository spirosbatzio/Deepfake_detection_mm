from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import config
from src.data.extract_audio import wav_path_for

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
CLEAN_DIR = config.SPLITS_DIR / "clean"
HASH_CACHE = config.SPLITS_DIR / "audio_hashes.csv"


def compute_hashes(df: pd.DataFrame) -> pd.DataFrame:
    if HASH_CACHE.exists():
        cached = pd.read_csv(HASH_CACHE)
        print(f"Loaded cached hashes ({len(cached)}) from {HASH_CACHE}")
        return df.merge(cached, on="wav", how="left")

    hashes = []
    for w in tqdm(df["wav"], desc="hashing audio"):
        data, _ = sf.read(w, dtype="float32")
        hashes.append(hashlib.md5(data.tobytes()).hexdigest())
    df["audio_hash"] = hashes
    df[["wav", "audio_hash"]].to_csv(HASH_CACHE, index=False)
    print(f"Wrote hash cache -> {HASH_CACHE}")
    return df


def stratified_split(unique: pd.DataFrame) -> dict[str, set]:
    """Assign unique audio_hashes to splits, stratified within each label."""
    rng = np.random.default_rng(SEED)
    buckets = {"train": set(), "val": set(), "test": set()}
    for _, grp in unique.groupby("label"):
        h = grp["audio_hash"].to_numpy().copy()
        rng.shuffle(h)
        n = len(h)
        n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
        buckets["train"] |= set(h[:n_tr])
        buckets["val"] |= set(h[n_tr:n_tr + n_va])
        buckets["test"] |= set(h[n_tr + n_va:])
    return buckets


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SPLITS_DIR / "index.csv")
    df["wav"] = [wav_path_for(r) for r in df.itertuples()]
    df = compute_hashes(df)

    # dedup: keep one representative clip per unique audio waveform
    unique = df.drop_duplicates("audio_hash").reset_index(drop=True)
    print(f"\n{len(df)} clips -> {len(unique)} unique audios "
          f"({len(df) - len(unique)} duplicates removed)")

    buckets = stratified_split(unique)

    print(f"\n{'split':6} {'clips':>7} {'real':>7} {'fake':>7}  fake%")
    seen = set()
    for name, hashes in buckets.items():
        assert not (hashes & seen), f"LEAK: {name} shares an audio hash!"
        seen |= hashes
        part = unique[unique["audio_hash"].isin(hashes)].reset_index(drop=True)
        part.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
        real = int((part["label"] == 0).sum())
        fake = int((part["label"] == 1).sum())
        print(f"{name:6} {len(part):7d} {real:7d} {fake:7d}  "
              f"{100 * fake / max(len(part), 1):4.1f}%")
    print("\nNo audio hash shared between splits")


if __name__ == "__main__":
    main()
