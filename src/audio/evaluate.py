from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForAudioClassification

from src import config
from src.metrics import compute_eer
from src.data.dataset import FakeAVCelebAudio


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores = []
    for batch in tqdm(loader, desc="eval"):
        x = batch["input_values"].to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x).logits
        scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tag", default="leaky", help="which model: 'leaky' or 'clean'")
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    ap.add_argument("--splits-dir", default=str(config.SPLITS_DIR))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ckpt = args.ckpt or str(config.CKPT_DIR / f"wav2vec2_{args.tag}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FakeAVCelebAudio(args.split, train=False, splits_dir=Path(args.splits_dir))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    model = AutoModelForAudioClassification.from_pretrained(
        ckpt, use_safetensors=True
    ).to(device)

    scores = predict(model, dl, device)
    labels = ds.df["label"].to_numpy()
    preds = (scores >= 0.5).astype(int)

    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds)

    print(f"\n=== {args.split} ({len(ds)} clips) ===")
    print(f"acc={acc:.4f}  auc={auc:.4f}  eer={eer:.4f}")
    print("confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)

    print("\nPer-category accuracy:")
    df = ds.df.copy()
    df["correct"] = (preds == labels)
    for t, g in df.groupby("type"):
        print(f"  {t:22} n={len(g):5d}  acc={g['correct'].mean():.3f}")

    # plots
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    a1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    a1.set(xlabel="FPR", ylabel="TPR", title=f"ROC ({args.split})")
    a1.legend()
    a2.hist(scores[labels == 0], bins=30, alpha=0.6, label="real")
    a2.hist(scores[labels == 1], bins=30, alpha=0.6, label="fake")
    a2.set(xlabel="P(fake)", ylabel="count", title="Score histogram")
    a2.legend()
    out = config.CKPT_DIR / f"eval_{args.tag}_{args.split}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
