import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
META_CSV = DATA_ROOT / "meta_data.csv"


AUDIO_DIR = ROOT / "audio"          # extracted 16 kHz wavs
SPLITS_DIR = ROOT / "splits"        # index.csv, train/val/test.csv
CKPT_DIR = ROOT / "checkpoints"

for _d in (AUDIO_DIR, SPLITS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

# Label convention: 0 = real speech, 1 = fake speech.
# Audio is REAL in categories A & C, FAKE in B & D.
REAL_AUDIO_TYPES = {"RealVideo-RealAudio", "FakeVideo-RealAudio"}
FAKE_AUDIO_TYPES = {"RealVideo-FakeAudio", "FakeVideo-FakeAudio"}


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if pkgs.exists():
        candidates += list(pkgs.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        "ffmpeg not found."
    )


FFMPEG = find_ffmpeg()
