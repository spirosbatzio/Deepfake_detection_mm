"""Generate OOD fake speech with Coqui XTTS v2 (a DIFFERENT generator than the
SV2TTS used in FakeAVCeleb). Standalone — run in an ISOLATED venv so Coqui's
pinned deps don't clobber the main wav2vec2 environment.

Setup (separate venv):
    python -m venv .venv_tts
    .venv_tts\\Scripts\\Activate.ps1
    pip install coqui-tts soundfile

Run:
    python tools\\gen_fake_tts.py --ref-dir ood\\real_librispeech --out ood\\fake_xtts --n 100

Output: ood/fake_xtts/*.wav  (16 kHz mono). Then deactivate and score with the
main venv:  python -m src.eval_ood --real-dir ood/real_librispeech --fake-dir ood/fake_xtts --tag clean
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # accept XTTS license non-interactively

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "Artificial intelligence is transforming how we live and work.",
    "She sells seashells by the seashore on a bright summer day.",
    "Climate change remains one of the greatest challenges of our time.",
    "He carefully packed his bags before the long journey ahead.",
    "The committee will reconvene next week to finalize the budget.",
    "A gentle breeze carried the scent of pine through the valley.",
    "Please remember to save your work before closing the program.",
    "The orchestra played a beautiful symphony late into the night.",
    "Scientists discovered a new species deep within the rainforest.",
    "Our flight was delayed because of the unexpected heavy storm.",
    "Reading books expands the mind and nurtures the imagination.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "They watched the sunset paint the sky in shades of orange.",
    "Innovation requires both creativity and relentless persistence.",
    "The museum unveiled a rare collection of ancient artifacts.",
    "Children laughed and played in the park all afternoon long.",
    "Economic forecasts suggest moderate growth in the coming year.",
    "The lighthouse guided ships safely through the foggy night.",
    "Learning a new language opens doors to different cultures.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="folder of reference speaker wavs/flac")
    ap.add_argument("--out", default="ood/fake_xtts")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = ap.parse_args()

    from TTS.api import TTS  # imported here so --help works without the dep

    refs = [p for p in Path(args.ref_dir).rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}]
    if not refs:
        raise SystemExit(f"No reference audio in {args.ref_dir}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import torch
    tts = TTS(args.model).to("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(0)
    for i in range(args.n):
        text = SENTENCES[i % len(SENTENCES)]
        ref = random.choice(refs)
        dst = out / f"xtts_{i:04d}.wav"
        try:
            tts.tts_to_file(text=text, speaker_wav=str(ref), language="en", file_path=str(dst))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {i}: {e}")
    print(f"\nDone. Wrote fakes to {out}")


if __name__ == "__main__":
    main()
