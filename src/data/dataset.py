from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src import config
from src.data.extract_audio import wav_path_for


class FakeAVCelebAudio(Dataset):
    def __init__(self, split: str, train: bool = False, splits_dir=None):
        splits_dir = splits_dir or config.SPLITS_DIR
        df = pd.read_csv(splits_dir / f"{split}.csv")
        df["wav"] = [wav_path_for(r) for r in df.itertuples()]

        from pathlib import Path
        df = df[df["wav"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(
                f"No extracted WAVs found for split={split}. Run extract_audio first."
            )
        self.df = df
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def _fix_length(self, wav: np.ndarray) -> np.ndarray:
        n = config.CLIP_SAMPLES
        if len(wav) >= n:
            # random crop while training, center crop otherwise
            start = np.random.randint(0, len(wav) - n + 1) if self.train else (len(wav) - n) // 2
            return wav[start:start + n]
        # pad short clips with zeros
        return np.pad(wav, (0, n - len(wav)))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = sf.read(row["wav"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = self._fix_length(wav)
        return {
            "input_values": torch.from_numpy(wav),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def class_weights(split: str = "train", splits_dir=None) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (mild imbalance)."""
    splits_dir = splits_dir or config.SPLITS_DIR
    df = pd.read_csv(splits_dir / f"{split}.csv")
    counts = df["label"].value_counts().sort_index()
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor([w.get(0, 1.0), w.get(1, 1.0)], dtype=torch.float32)
