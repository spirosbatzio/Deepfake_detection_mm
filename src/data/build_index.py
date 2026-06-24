from __future__ import annotations

import re
import pandas as pd

from src import config

SPEAKER_RE = re.compile(r"id\d+", re.IGNORECASE)


def label_for_type(clip_type: str) -> int:
    """0 = real audio (A, C), 1 = fake audio (B, D)."""
    if clip_type in config.REAL_AUDIO_TYPES:
        return 0
    if clip_type in config.FAKE_AUDIO_TYPES:
        return 1
    raise ValueError(f"Unknown type folder: {clip_type!r}")


def build_index() -> pd.DataFrame:
    rows = []
    for mp4 in config.DATA_ROOT.rglob("*.mp4"):
        rel = mp4.relative_to(config.DATA_ROOT)
        parts = rel.parts  # (type, race, gender, id#####, file.mp4)
        if len(parts) < 2:
            continue
        clip_type = parts[0]
        if clip_type not in config.REAL_AUDIO_TYPES | config.FAKE_AUDIO_TYPES:
            continue  # skip stray files (README.txt, csv, etc.)


        speaker = next((p for p in parts if SPEAKER_RE.fullmatch(p)), None)
        if speaker is None:
            continue

        rows.append(
            dict(
                clip_path=str(mp4),
                type=clip_type,
                race=parts[1] if len(parts) > 2 else "",
                gender=parts[2] if len(parts) > 3 else "",
                speaker_id=speaker,
                filename=mp4.name,
                label=label_for_type(clip_type),
            )
        )

    df = pd.DataFrame(rows).sort_values(["type", "speaker_id", "filename"])
    return df.reset_index(drop=True)


def main() -> None:
    df = build_index()
    out = config.SPLITS_DIR / "index.csv"
    df.to_csv(out, index=False)

    print(f"Wrote {out}  ({len(df)} clips)\n")
    print("Clips per category:")
    print(df["type"].value_counts().to_string(), "\n")
    print("Label balance (0=real speech, 1=fake speech):")
    print(df["label"].value_counts().to_string(), "\n")
    print(f"Unique speakers: {df['speaker_id'].nunique()}")
    print("Unique speakers per label:")
    print(df.groupby("label")["speaker_id"].nunique().to_string())


if __name__ == "__main__":
    main()
