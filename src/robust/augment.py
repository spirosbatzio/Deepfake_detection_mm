from __future__ import annotations

import numpy as np

from src import config

RNG = np.random.default_rng()


def add_noise(wav, snr_db_range=(8, 30)):
    snr = RNG.uniform(*snr_db_range)
    sig_p = np.mean(wav ** 2) + 1e-9
    noise_p = sig_p / (10 ** (snr / 10))
    return wav + RNG.normal(0, np.sqrt(noise_p), size=wav.shape).astype(np.float32)


def random_gain(wav, db_range=(-6, 6)):
    return wav * float(10 ** (RNG.uniform(*db_range) / 20))


def mulaw_codec(wav, quant_range=(64, 256)):
    """mu-law companding + coarse requantization — mimics lossy-codec artifacts."""
    mu = float(RNG.integers(*quant_range)) - 1
    x = np.clip(wav, -1, 1)
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((comp + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)).astype(np.float32)


def band_limit(wav, low_sr_range=(4000, 12000)):
    """Downsample to a lower rate and back — band-limiting like phone/codec audio."""
    target = int(RNG.integers(*low_sr_range))
    n = len(wav)
    m = max(int(n * target / config.SAMPLE_RATE), 1)
    down = np.interp(np.linspace(0, n - 1, m), np.arange(n), wav)
    return np.interp(np.linspace(0, m - 1, n), np.arange(m), down).astype(np.float32)


def reverb(wav, decay_range=(0.2, 0.6), ir_ms_range=(50, 200)):
    """Convolve with a synthetic exponentially-decaying impulse response."""
    ir_len = int(config.SAMPLE_RATE * RNG.uniform(*ir_ms_range) / 1000)
    t = np.arange(ir_len)
    ir = np.exp(-t / (ir_len * RNG.uniform(*decay_range))).astype(np.float32)
    ir[0] = 1.0
    out = np.convolve(wav, ir / ir.sum(), mode="full")[: len(wav)]
    return out.astype(np.float32)


TRANSFORMS = [
    (add_noise, 0.5),
    (random_gain, 0.5),
    (mulaw_codec, 0.4),
    (band_limit, 0.4),
    (reverb, 0.3),
]


def augment(wav: np.ndarray) -> np.ndarray:
    """Apply each transform independently with its probability; re-normalize."""
    for fn, p in TRANSFORMS:
        if RNG.random() < p:
            wav = fn(wav)
    peak = np.max(np.abs(wav)) + 1e-9
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32)
