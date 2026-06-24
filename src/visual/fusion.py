from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForAudioClassification

from src import config
from src.metrics import compute_eer

ALL_TYPES = config.REAL_AUDIO_TYPES | config.FAKE_AUDIO_TYPES
ID_RE = re.compile(r"id\d+", re.IGNORECASE)


def wav_for_clip(clip_path: str) -> str:
    """Reconstruct the cached-wav path from a clip's mp4 path."""
    p = Path(clip_path)
    parts = p.parts
    ctype = next((t for t in ALL_TYPES if t in parts), parts[0])
    sid = next((x for x in parts if ID_RE.fullmatch(x)), "unk")
    return str(config.AUDIO_DIR / f"{ctype}__{sid}__{p.stem}.wav")


def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    n = config.CLIP_SAMPLES
    if len(wav) >= n:
        s = (len(wav) - n) // 2
        return wav[s:s + n]
    return np.pad(wav, (0, n - len(wav)))


@torch.no_grad()
def audio_scores(df: pd.DataFrame, tag: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(
        config.CKPT_DIR / f"wav2vec2_{tag}", use_safetensors=True
    ).to(device).eval()
    out = []
    for cp in tqdm(df["clip_path"], desc="audio"):
        w = wav_for_clip(cp)
        src = w if Path(w).exists() else cp
        x = torch.from_numpy(load_audio(src)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logit = model(x).logits
        out.append(float(torch.softmax(logit.float(), -1)[0, 1]))
    return np.array(out)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-9)


def report(name: str, labels: np.ndarray, scores: np.ndarray) -> None:
    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)
    acc = ((scores >= 0.5).astype(int) == labels).mean()
    print(f"  {name:14} acc={acc:.3f}  auc={auc:.3f}  eer={eer:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", default=str(config.SPLITS_DIR / "visual_test.csv"))
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--w-audio", type=float, default=0.6, help="fusion weight on audio")
    args = ap.parse_args()

    df = pd.read_csv(args.visual)
    labels = df["label"].to_numpy()

    a = audio_scores(df, args.tag)                 # P(fake) in [0,1]
    v = minmax(1.0 - df["max_corr"].to_numpy())    # low sync -> high fake score
    fused = args.w_audio * a + (1 - args.w_audio) * v

    print(f"\n=== Fusion ablation on {len(df)} clips ({args.visual}) ===")
    report("audio-only", labels, a)
    report("visual-only", labels, v)
    report(f"fused(w={args.w_audio})", labels, fused)

    print("\nPer-category mean fake-score (audio / visual):")
    df = df.assign(audio=a, visual=v)
    for t, g in df.groupby("type"):
        print(f"  {t:22} n={len(g):4d}  audio={g['audio'].mean():.3f}  "
              f"visual={g['visual'].mean():.3f}")

    # focused dubbing test: real-face clips only, sync alone
    rv = df[df["type"].isin(["RealVideo-RealAudio", "RealVideo-FakeAudio"])]
    if rv["label"].nunique() == 2:
        print("\nDubbing detection on real-face clips (RealVideo-*), VISUAL-ONLY:")
        report("sync-only", rv["label"].to_numpy(), rv["visual"].to_numpy())
        print("  (real face + fake/dubbed audio is the case sync is built to catch)")


if __name__ == "__main__":
    main()
