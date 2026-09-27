"""Synthetic EEG-like signals for definition hardening."""

from __future__ import annotations

import numpy as np


DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


def make_sine_epochs(
    *,
    n_epochs: int = 96,
    n_channels: int = 8,
    sfreq: float = 128.0,
    duration: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create controlled multi-band epochs with class-dependent band weights."""

    rng = np.random.default_rng(seed)
    n_times = int(round(sfreq * duration))
    t = np.arange(n_times) / sfreq
    freqs = np.array([2.0, 6.0, 10.0, 20.0, 40.0])
    labels = np.arange(n_epochs) % 3
    x = np.zeros((n_epochs, n_channels, n_times), dtype=np.float64)

    for i in range(n_epochs):
        weights = np.array([0.5, 0.6, 0.7, 0.7, 0.35])
        if labels[i] == 0:
            weights[2] += 1.0
        elif labels[i] == 1:
            weights[3] += 1.0
        else:
            weights[1] += 0.8
            weights[4] += 0.4
        for ch in range(n_channels):
            channel_gain = rng.lognormal(mean=0.0, sigma=0.15)
            phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
            for b, freq in enumerate(freqs):
                x[i, ch] += (
                    channel_gain
                    * weights[b]
                    * np.sin(2 * np.pi * freq * t + phase[b])
                )
        x[i] += 0.05 * rng.normal(size=(n_channels, n_times))

    x -= x.mean(axis=-1, keepdims=True)
    return x, labels.astype(int), 1.0 / sfreq


def fft_band_components(
    epoch: np.ndarray,
    *,
    sfreq: float,
    bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Split one epoch into ideal FFT bands.

    This is for synthetic definition-hardening and smoke tests. Real EEG
    experiments should later use a documented FIR/IIR preprocessing choice and
    report the pre-map band cosine matrix.
    """

    epoch = np.asarray(epoch, dtype=np.float64)
    freqs = np.fft.rfftfreq(epoch.shape[-1], d=1.0 / sfreq)
    spectrum = np.fft.rfft(epoch, axis=-1)
    components = []
    names = []
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        filtered = np.zeros_like(spectrum)
        filtered[..., mask] = spectrum[..., mask]
        components.append(np.fft.irfft(filtered, n=epoch.shape[-1], axis=-1))
        names.append(name)
    return np.asarray(components), tuple(names)
