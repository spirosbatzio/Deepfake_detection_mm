from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForAudioClassification

from src import config
from src.metrics import compute_eer
from src.data.dataset import FakeAVCelebAudio, class_weights

MODEL_NAME = "facebook/wav2vec2-base"


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_labels, all_scores = [], []
    for batch in loader:
        x = batch["input_values"].to(device)
        logits = model(x).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)
        all_scores.append(probs.cpu().numpy())
        all_labels.append(batch["label"].numpy())
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    preds = (scores >= 0.5).astype(int)
    return {
        "acc": float((preds == labels).mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "eer": compute_eer(labels, scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits-dir", default=str(config.SPLITS_DIR),
                    help="dir with train/val/test.csv (use splits/clean for the corrected flow)")
    ap.add_argument("--tag", default="leaky",
                    help="checkpoint name suffix, e.g. 'leaky' or 'clean'")
    args = ap.parse_args()
    splits_dir = Path(args.splits_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = FakeAVCelebAudio("train", train=True, splits_dir=splits_dir)
    val_ds = FakeAVCelebAudio("val", train=False, splits_dir=splits_dir)
    print(f"[{args.tag}] splits_dir={splits_dir}  train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_safetensors=True
    )
    model.freeze_feature_encoder()
    model.to(device)

    weights = class_weights("train", splits_dir=splits_dir).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_eer = 1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["input_values"].to(device)
            y = batch["label"].to(device)
            optim.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(x).logits
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")

        metrics = evaluate(model, val_dl, device)
        print(f"  val: acc={metrics['acc']:.3f}  auc={metrics['auc']:.3f}  "
              f"eer={metrics['eer']:.3f}")

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            ckpt = config.CKPT_DIR / f"wav2vec2_{args.tag}"
            model.save_pretrained(ckpt)
            print(f"  ↑ new best (eer={best_eer:.3f}) saved to {ckpt}")

    print(f"\nDone. best val EER = {best_eer:.3f}")


if __name__ == "__main__":
    main()
