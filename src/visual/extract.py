from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data.extract_audio import wav_path_for

# FaceMesh lip landmarks (upper/lower inner lip + mouth corners); valid for the
# 478-point Tasks face_landmarker model too.
UPPER_LIP, LOWER_LIP = 13, 14
LEFT_CORNER, RIGHT_CORNER = 78, 308
FOREHEAD, CHIN = 10, 152  # for face-height normalization
MAX_LAG = 6  # frames (~0.2s at 25–30 fps)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = config.ROOT / "models" / "face_landmarker.task"


def get_landmarker():
    """Build a MediaPipe Tasks FaceLandmarker in VIDEO mode (mediapipe>=0.10)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face_landmarker model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,  # per-frame; no timestamp state
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts), mp


def mouth_openness_series(video_path: str, landmarker, mp) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            vals.append(np.nan)
        else:
            lm = res.face_landmarks[0]
            gap = abs(lm[UPPER_LIP].y - lm[LOWER_LIP].y)
            face_h = abs(lm[FOREHEAD].y - lm[CHIN].y) + 1e-6
            vals.append(gap / face_h)
    cap.release()
    return np.array(vals, dtype=np.float32), fps


def audio_envelope(audio_path: str, n_frames: int, fps: float) -> np.ndarray:
    wav, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    hop = max(int(sr / fps), 1)
    rms = librosa.feature.rms(y=wav, frame_length=hop * 2, hop_length=hop)[0]
    # resample RMS to exactly n_frames
    if len(rms) < 2:
        return np.zeros(n_frames, dtype=np.float32)
    idx = np.linspace(0, len(rms) - 1, n_frames)
    return np.interp(idx, np.arange(len(rms)), rms).astype(np.float32)


def sync_features(mouth: np.ndarray, env: np.ndarray) -> tuple[float, int, int]:
    """Return (max_corr, best_lag, n_valid_frames). Robust to missing-face frames."""
    valid = ~np.isnan(mouth)
    n_valid = int(valid.sum())
    if n_valid < 8:
        return 0.0, 0, n_valid
    m = mouth.copy()
    m[~valid] = np.nanmean(m[valid])
    m = (m - m.mean()) / (m.std() + 1e-6)
    e = (env - env.mean()) / (env.std() + 1e-6)
    best_corr, best_lag = -1.0, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        a, b = (m[lag:], e[:len(e) - lag]) if lag >= 0 else (m[:lag], e[-lag:])
        k = min(len(a), len(b))
        if k < 8:
            continue
        c = float(np.corrcoef(a[:k], b[:k])[0, 1])
        if not np.isnan(c) and c > best_corr:
            best_corr, best_lag = c, lag
    return best_corr, best_lag, n_valid


def select(df, per_category, limit):
    if per_category:
        df = df.groupby("type", group_keys=False).head(per_category)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--splits-dir", default=str(config.SPLITS_DIR / "clean"))
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    df = pd.read_csv(splits_dir / f"{args.split}.csv")
    df = select(df, args.per_category, args.limit)
    print(f"Extracting visual sync features for {len(df)} clips "
          f"(split={args.split}, dir={splits_dir})")

    landmarker, mp = get_landmarker()
    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        try:
            mouth, fps = mouth_openness_series(r.clip_path, landmarker, mp)
            if len(mouth) < 8:
                corr, lag, nv = 0.0, 0, 0
            else:
                wav = wav_path_for(r)
                audio_src = wav if Path(wav).exists() else r.clip_path
                env = audio_envelope(audio_src, len(mouth), fps)
                corr, lag, nv = sync_features(mouth, env)
        except Exception as e:  # noqa: BLE001
            corr, lag, nv = 0.0, 0, 0
            tqdm.write(f"  skip {Path(r.clip_path).name}: {e}")
        rows.append(dict(clip_path=r.clip_path, label=r.label, type=r.type,
                         max_corr=corr, best_lag=lag, n_frames=nv))
    landmarker.close()

    out = splits_dir / f"visual_{args.split}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    res = pd.DataFrame(rows)
    print(f"\nWrote {out}")
    print("Mean sync corr by category (higher = better lip-sync):")
    print(res.groupby("type")["max_corr"].mean().to_string())
    print("\nMean sync corr by label (0=real audio, 1=fake audio):")
    print(res.groupby("label")["max_corr"].mean().to_string())


if __name__ == "__main__":
    main()
