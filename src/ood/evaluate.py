from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForAudioClassification

from src import config
from src.metrics import compute_eer

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus"}


def load_clip(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    n = config.CLIP_SAMPLES
    if len(wav) >= n:
        return wav[(len(wav) - n) // 2: (len(wav) - n) // 2 + n]  # center crop
    return np.pad(wav, (0, n - len(wav)))


def list_audio(folder: str) -> list[Path]:
    p = Path(folder)
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)


@torch.no_grad()
def score_folder(model, device, folder: str) -> np.ndarray:
    files = list_audio(folder)
    if not files:
        raise RuntimeError(f"No audio files found under {folder}")
    scores = []
    for f in tqdm(files, desc=Path(folder).name):
        x = torch.from_numpy(load_clip(str(f))).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x).logits
        scores.append(float(torch.softmax(logits.float(), -1)[0, 1]))  # P(fake)
    return np.array(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--fake-dir", default=None)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if not (args.real_dir or args.fake_dir):
        ap.error("provide --real-dir and/or --fake-dir")

    ckpt = args.ckpt or str(config.CKPT_DIR / f"wav2vec2_{args.tag}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(
        ckpt, use_safetensors=True
    ).to(device).eval()
    print(f"Model: {ckpt}  (P(fake) threshold={args.threshold})\n")

    labels, scores = [], []
    if args.real_dir:
        s = score_folder(model, device, args.real_dir)
        acc = float((s < args.threshold).mean())  # correct if predicted real
        print(f"\nREAL  n={len(s):4d}  mean P(fake)={s.mean():.3f}  "
              f"accuracy(real)={acc:.3f}")
        labels += [0] * len(s); scores += list(s)
    if args.fake_dir:
        s = score_folder(model, device, args.fake_dir)
        acc = float((s >= args.threshold).mean())  # correct if predicted fake
        print(f"\nFAKE  n={len(s):4d}  mean P(fake)={s.mean():.3f}  "
              f"accuracy(fake)={acc:.3f}")
        labels += [1] * len(s); scores += list(s)

    if args.real_dir and args.fake_dir:
        labels, scores = np.array(labels), np.array(scores)
        print(f"\nOOD combined:  AUC={roc_auc_score(labels, scores):.4f}  "
              f"EER={compute_eer(labels, scores):.4f}")
        print("(Compare to in-distribution EER ~0.000 — a large gap = channel/"
              "generator shortcut, not a real detector.)")


if __name__ == "__main__":
    main()
