# Multimodal Deepfake Speech Detection

This project was developed as an assignment for the **Multi-Modal Machine Learning** course of the [MSc in Artificial Intelligence](https://msc-ai.iit.demokritos.gr/), offered jointly by the University of Piraeus and NCSR "Demokritos".

**Authors:** Kostis Matzorakis, George Manthos, Spiros Batziopoulos

Detect AI-generated speech using a fine-tuned **wav2vec2** audio classifier, with a
complementary **audio-visual sync** (lip-movement) stream, on **FakeAVCeleb v1.2**.


## Background

Modern text-to-speech and voice-conversion systems can clone a person's voice from
seconds of audio, making **synthetic (deepfake) speech** a real threat for fraud and
misinformation. The goal here is a **detector** that, given a short clip, decides whether
the *speech* is genuine or AI-generated. Because audio-only detectors are easy to fool, we
also add a **visual stream**: in a real recording the lip movements line up with the
sound, while in a dubbed/synthesised clip they drift out of sync — a cue that survives
even when the audio itself sounds convincing.

### Dataset & labels

We use **[FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) v1.2**, a dataset of
celebrity video clips in four combinations of real/fake video × real/fake audio:

| Category | Video | Audio | Our audio label |
|---|---|---|---|
| RealVideo-RealAudio | real | real | **0 (real)** |
| FakeVideo-RealAudio | fake (face-swap) | real | **0 (real)** |
| RealVideo-FakeAudio | real | synthesised | **1 (fake)** |
| FakeVideo-FakeAudio | fake | synthesised | **1 (fake)** |

Since the task is **fake-*speech*** detection, the label depends only on the audio:
categories with synthesised audio are positive (fake), the rest are negative (real).


## What this project does

Starting from raw FakeAVCeleb videos, the pipeline:

1. **Builds a labelled index** of every clip (real vs. fake *audio*) and extracts mono
   16 kHz waveforms.
2. **Fine-tunes a wav2vec2 detector** (frozen CNN feature encoder) and reports
   **EER / AUC** with a per-category breakdown.
3. **Exposes and fixes a data leak.** The naive speaker-disjoint split still shares the
   *same audio waveform* across many face-swapped videos (FakeAVCeleb reuses voices), so
   identical clips land in train and test. We diagnose it by audio content-hash and
   rebuild a deduplicated (`splits/clean`) flow. Both flows are kept so the leak can be
   discussed and compared.
4. **Tests generalization (OOD).** The in-distribution task is trivially separable, so we
   re-evaluate on unseen sources — **LibriSpeech** reals + **Coqui XTTS** fakes — to
   measure whether the model learned *synthesis artefacts* or just a *generator/channel
   shortcut*.
5. **Adds a multimodal stream.** A lip-sync feature (mouth-openness vs. audio RMS
   envelope cross-correlation, via MediaPipe FaceLandmarker) is **late-fused** with the
   audio score, with an ablation to show the streams are complementary.
6. **Hardens the model (extension).** Data augmentation (noise, gain, μ-law codec,
   band-limiting, reverb) and multi-generator training to recover cross-generator
   performance.

## Key findings

| Setting | EER | What it shows |
|---|---|---|
| In-distribution (leaky split) | ~0.001 | Identical audio in train/test → near-perfect |
| In-distribution (clean, deduped) | ~0.000 | Task is *trivially separable* even without the leak |
| **Generalization** (LibriSpeech + XTTS) | **0.37** | Model learned a generator shortcut, not synthesis cues — XTTS caught well below chance |
| + Augmentation only | ~0.30 | Helps, but XTTS never seen in training |
| + Multi-generator training | ~0.00 | Recovers performance (within-generator caveat) |

**Complementary failure modes:** the audio model catches synthesis but fails
cross-generator; the lip-sync stream catches audio dubbing (AUC ~0.70) but is blind to
lip-synced fakes (Wav2Lip). Late fusion combines their strengths.

**Takeaway:** a deepfake-speech detector can score near-perfectly on its own benchmark yet
collapse on a *new* generator. Honest evaluation therefore needs out-of-distribution
tests; multi-generator training and a complementary visual stream are what actually move
real-world robustness.

## Package layout

```
src/
  config.py            paths, label convention, ffmpeg locator
  metrics.py           compute_eer
  data/                build_index, splits (leaky), splits_clean, extract_audio,
                       dataset, diagnose_leak
  audio/               train, evaluate           (wav2vec2 detector)
  visual/              extract (lip-sync), fusion (late fusion + ablation)
  ood/                 librispeech (download), evaluate (score any audio folders)
  robust/              augment, dataset, prepare, train_robust   (extension)
tools/                 gen_fake_tts.py           (run in isolated .venv_tts)
```

## Setup

```powershell
# 1. Clone, then create the main environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # PyTorch is CUDA 12.1 (RTX 3070)

# 2. Download FakeAVCeleb v1.2 and place it at:
#    data/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/   (with meta_data.csv inside)
#    (path is defined in src/config.py)

# 3. ffmpeg — installed via winget (Gyan.FFmpeg); config.py auto-locates it
```

Then run the pipeline below in order. The XTTS fake-generation step uses a **separate**
`.venv_tts` (see *Environment*).

## Baseline pipeline (run from project root, main `.venv`)

```powershell
# 1. Index + audio extraction
python -m src.data.build_index
python -m src.data.extract_audio --split all

# 2. Splits — leaky baseline AND corrected (audio-hash) flow
python -m src.data.splits
python -m src.data.splits_clean
python -m src.data.diagnose_leak --splits-dir splits/clean   # cross-split hashes = 0

# 3. Train + evaluate the clean audio model
python -m src.audio.train    --splits-dir splits/clean --tag clean --epochs 3 --batch-size 8
python -m src.audio.evaluate --splits-dir splits/clean --tag clean --split test

# 4. Generalization (OOD): LibriSpeech reals + XTTS fakes (XTTS via tools/, isolated venv)
python -m src.ood.librispeech --n 200
python -m src.ood.evaluate --real-dir ood/real_librispeech --fake-dir ood/fake_xtts --tag clean

# 5. Multimodal: visual sync features + fusion ablation
python -m src.visual.extract --split test --splits-dir splits --per-category 40
python -m src.visual.fusion  --visual splits/visual_test.csv --tag clean
```

## Robustness extension

```powershell
python -m src.robust.prepare --real-dir ood/real_librispeech --fake-dir ood/fake_xtts --train-frac 0.5
python -m src.robust.train_robust --tag robust_aug --epochs 3
python -m src.robust.train_robust --tag robust_multigen --epochs 3 \
    --extra-real-dir ood/robust/train/real --extra-fake-dir ood/robust/train/fake --oversample-ood 20
python -m src.ood.evaluate --real-dir ood/robust/test/real --fake-dir ood/robust/test/fake --tag robust_aug
python -m src.ood.evaluate --real-dir ood/robust/test/real --fake-dir ood/robust/test/fake --tag robust_multigen
```

## Environment

PyTorch (CUDA 12.1) + `requirements.txt`. ffmpeg via winget (auto-located by
`config.py`). XTTS generation uses a **separate** `.venv_tts` to avoid dependency
clashes — see `tools/gen_fake_tts.py`.
