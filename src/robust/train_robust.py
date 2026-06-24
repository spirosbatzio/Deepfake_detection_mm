from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForAudioClassification

from src import config
from src.audio.train import MODEL_NAME, evaluate
from src.data.extract_audio import wav_path_for
from src.robust.dataset import ListAudioDataset, class_weights

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def favceleb_items(split: str) -> list[tuple[str, int]]:
    df = pd.read_csv(config.SPLITS_DIR / "clean" / f"{split}.csv")
    items = []
    for r in df.itertuples():
        w = wav_path_for(r)
        if Path(w).exists():
            items.append((w, int(r.label)))
    return items


def dir_items(folder: str, label: int) -> list[tuple[str, int]]:
    if not folder:
        return []
    return [(str(f), label) for f in Path(folder).rglob("*") if f.suffix.lower() in EXTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="robust_aug")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true", help="disable augmentation")
    ap.add_argument("--extra-real-dir", default=None, help="OOD real (multi-gen)")
    ap.add_argument("--extra-fake-dir", default=None, help="OOD fake (multi-gen)")
    ap.add_argument("--oversample-ood", type=int, default=1,
                    help="replicate OOD items N times to balance vs FakeAVCeleb")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    augment = not args.no_augment

    train_items = favceleb_items("train")
    ood = (dir_items(args.extra_real_dir, 0) + dir_items(args.extra_fake_dir, 1)) * args.oversample_ood
    train_items += ood
    val_items = favceleb_items("val")
    print(f"[{args.tag}] device={device} augment={augment}  "
          f"train={len(train_items)} (FakeAVCeleb + {len(ood)} OOD)  val={len(val_items)}")

    train_dl = DataLoader(ListAudioDataset(train_items, train=True, augment=augment),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(ListAudioDataset(val_items, train=False, augment=False),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_safetensors=True)
    model.freeze_feature_encoder()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_items).to(device))
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_eer = 1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x, y = batch["input_values"].to(device), batch["label"].to(device)
            optim.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = criterion(model(x).logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / (pbar.n + 1):.4f}")

        m = evaluate(model, val_dl, device)
        print(f"  val(FakeAVCeleb): acc={m['acc']:.3f} auc={m['auc']:.3f} eer={m['eer']:.3f}")
        if m["eer"] <= best_eer:
            best_eer = m["eer"]
            ckpt = config.CKPT_DIR / f"wav2vec2_{args.tag}"
            model.save_pretrained(ckpt)
            print(f"  saved -> {ckpt}")

    print(f"\nDone. Now compare on held-out OOD:\n"
          f"  python -m src.ood.evaluate --real-dir ood/robust/test/real "
          f"--fake-dir ood/robust/test/fake --tag {args.tag}")


if __name__ == "__main__":
    main()
