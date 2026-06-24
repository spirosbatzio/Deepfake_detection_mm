from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def make_splits() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(config.SPLITS_DIR / "index.csv")

    speakers = np.array(sorted(df["speaker_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    buckets = {
        "train": set(speakers[:n_train]),
        "val": set(speakers[n_train:n_train + n_val]),
        "test": set(speakers[n_train + n_val:]),
    }

    out = {}
    for name, spk in buckets.items():
        part = df[df["speaker_id"].isin(spk)].reset_index(drop=True)
        part.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
        out[name] = part
    return out


def main() -> None:
    splits = make_splits()
    all_speakers = set()
    print(f"{'split':6} {'clips':>7} {'speakers':>9} {'real':>7} {'fake':>7}  fake%")
    for name, part in splits.items():
        spk = set(part["speaker_id"])
        # sanity check: no speaker overlap with previously seen splits
        assert not (spk & all_speakers), f"LEAK: {name} shares speakers!"
        all_speakers |= spk
        real = int((part["label"] == 0).sum())
        fake = int((part["label"] == 1).sum())
        pct = 100 * fake / max(len(part), 1)
        print(f"{name:6} {len(part):7d} {len(spk):9d} {real:7d} {fake:7d}  {pct:4.1f}%")
    print("\nNo speaker overlap between splits")


if __name__ == "__main__":
    main()
