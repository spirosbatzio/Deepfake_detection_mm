from __future__ import annotations

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.robust.augment import augment as augment_fn


def load_fixed(path: str, train: bool) -> np.ndarray:
    wav, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    n = config.CLIP_SAMPLES
    if len(wav) >= n:
        start = np.random.randint(0, len(wav) - n + 1) if train else (len(wav) - n) // 2
        wav = wav[start:start + n]
    else:
        wav = np.pad(wav, (0, n - len(wav)))
    return wav.astype(np.float32)


class ListAudioDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], train: bool = False, augment: bool = False):
        self.items = items
        self.train = train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav = load_fixed(path, self.train)
        if self.augment:
            wav = augment_fn(wav)
        return {
            "input_values": torch.from_numpy(wav),
            "label": torch.tensor(int(label), dtype=torch.long),
        }


def class_weights(items: list[tuple[str, int]]) -> torch.Tensor:
    labels = np.array([l for _, l in items])
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=float)
    w = counts.sum() / (2 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
