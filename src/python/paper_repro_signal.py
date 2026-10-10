"""Isolated NR-PSS method-level reproduction, not a 3GPP channel conformance tool."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from numpy.lib.stride_tricks import sliding_window_view

NFFT = 256
FS = 3_840_000
SCS = 15_000
CP = 18
FRAME = 384
TDL_A_DELAYS = np.array([0, .3819, .4025, .5868, .4610, .5375, .6708,
    .5750, .7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119,
    3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
TDL_A_DB = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5,
    -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
    -18.3, -18.9, -16.6, -19.9, -29.7])


def pss(nid2: int) -> np.ndarray:
    if nid2 not in (0, 1, 2):
        raise ValueError("N_ID2 must be 0, 1, or 2")
    x = np.zeros(127, dtype=np.int8)
    x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for n in range(120):
        x[n + 7] = (x[n + 4] + x[n]) & 1
    return (1 - 2 * x[(np.arange(127) + 43 * nid2) % 127]).astype(np.float32)


def pss_symbol(nid2: int) -> np.ndarray:
    grid = np.zeros(NFFT, dtype=np.complex64)
    grid[NFFT // 2 - 64:NFFT // 2 + 63] = pss(nid2)
    useful = np.fft.ifft(np.fft.ifftshift(grid))
    useful = useful / np.sqrt(np.mean(np.abs(useful) ** 2))
    return np.r_[useful[-CP:], useful].astype(np.complex64)


def make_scene(seed: int, nid2: int, offset: int, cfo_sc: float, snr_db: float,
               profile: str = "TDL-A", delay_spread_ns: float = 100,
               speed_kmh: float = 0) -> tuple[np.ndarray, dict]:
    if not 0 <= offset <= FRAME - (NFFT + CP) - 8:
        raise ValueError("PSS offset outside observation window")
    rng = np.random.default_rng(seed)
    tx = np.zeros(FRAME, dtype=np.complex128)
    tx[offset:offset + NFFT + CP] = pss_symbol(nid2)
    if profile == "TDL-A":
        # Fractional-sample linear interpolation is an explicit approximation.
        delays = TDL_A_DELAYS * delay_spread_ns * 1e-9 * FS
        powers = 10 ** (TDL_A_DB / 10)
        gains = (rng.standard_normal(len(powers)) + 1j * rng.standard_normal(len(powers))) * np.sqrt(powers / 2)
        gains /= np.sqrt(np.sum(powers))
        signal = np.zeros(FRAME, dtype=np.complex128)
        doppler_max = (speed_kmh / 3.6) * 3e9 / 299_792_458
        for delay, gain in zip(delays, gains):
            samples = np.arange(FRAME) - delay
            shifted = (np.interp(samples, np.arange(FRAME), tx.real, left=0, right=0)
                + 1j * np.interp(samples, np.arange(FRAME), tx.imag, left=0, right=0))
            fd = doppler_max * rng.uniform(-1, 1)
            signal += gain * shifted * np.exp(2j * np.pi * fd * np.arange(FRAME) / FS)
    elif profile == "AWGN":
        signal = tx
    else:
        raise ValueError("profile must be AWGN or TDL-A")
    signal *= np.exp(2j * np.pi * cfo_sc * SCS * np.arange(FRAME) / FS)
    active_power = np.mean(np.abs(signal[offset:offset + NFFT + CP]) ** 2)
    sigma = np.sqrt(active_power / (2 * 10 ** (snr_db / 10)))
    rx = signal + sigma * (rng.standard_normal(FRAME) + 1j * rng.standard_normal(FRAME))
    return rx.astype(np.complex64), dict(seed=seed, nid2=nid2, offset=offset,
        cfo_sc=float(cfo_sc), snr_db=float(snr_db), profile=profile,
        delay_spread_ns=float(delay_spread_ns), speed_kmh=float(speed_kmh))


def _correlate(rx: np.ndarray, template: np.ndarray) -> np.ndarray:
    return fftconvolve(rx, np.conj(template[::-1]), mode="valid")


def grid_search(rx: np.ndarray, cfo_grid: np.ndarray) -> dict:
    if len(rx) != FRAME or len(cfo_grid) == 0:
        raise ValueError("invalid observation or CFO grid")
    sample = np.arange(FRAME)
    best = (-1., 0, 0, 0.)
    for cfo in cfo_grid:
        corrected = rx * np.exp(-2j * np.pi * float(cfo) * SCS * sample / FS)
        for nid2 in range(3):
            z = _correlate(corrected, pss_symbol(nid2))
            i = int(np.argmax(np.abs(z)))
            score = float(np.abs(z[i]))
            if score > best[0]:
                best = score, i, nid2, float(cfo)
    return dict(score=best[0], offset=best[1], nid2=best[2], cfo_sc=best[3])


def corr_features(rx: np.ndarray, combined: bool = False) -> np.ndarray:
    if len(rx) != FRAME:
        raise ValueError("expected 384 samples")
    cross, half_auto = correlation_maps(rx)
    cross_abs = np.abs(cross)
    cross_abs /= max(float(cross_abs.max()), 1e-8)
    if combined:
        auto_abs = np.abs(half_auto)
        auto_abs /= max(float(auto_abs.max()), 1e-8)
        channel0 = (cross_abs + auto_abs) / 2
        channel1 = auto_abs * np.angle(half_auto) / np.pi
    else:
        channel0 = cross_abs
        channel1 = cross_abs * np.angle(cross) / np.pi
    feature = np.zeros((3, 128, 2), np.float32)
    feature[:, :cross.shape[1], 0] = channel0
    feature[:, :cross.shape[1], 1] = channel1
    return feature.reshape(4, 96, 2)


def correlation_maps(rx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equations (23)/(24): full PSS correlation and half-symbol product sum."""
    if len(rx) != FRAME:
        raise ValueError("expected 384 samples")
    windows = sliding_window_view(rx[CP:], NFFT)
    cross = np.empty((3, len(windows)), np.complex64)
    half_auto = np.empty_like(cross)
    for nid2 in range(3):
        template = pss_symbol(nid2)[CP:]
        q = windows * np.conj(template)
        cross[nid2] = np.sum(q, axis=1)
        half_auto[nid2] = np.sum(q[:, :NFFT // 2] * np.conj(q[:, NFFT // 2:]), axis=1)
    return cross, half_auto


def half_auto_search(rx: np.ndarray, cfo_options: np.ndarray) -> dict:
    """Paper-equation classical score; CFO options resolve its 2-SCS phase alias."""
    if len(cfo_options) == 0:
        raise ValueError("empty CFO options")
    _, half_auto = correlation_maps(rx)
    nid2, offset = np.unravel_index(np.argmax(np.abs(half_auto)), half_auto.shape)
    phase = np.angle(half_auto[nid2, offset])
    wrapped = np.angle(np.exp(1j * (phase + np.pi * np.asarray(cfo_options))))
    cfo = float(cfo_options[np.argmin(np.abs(wrapped))])
    return dict(nid2=int(nid2), offset=int(offset), cfo_sc=cfo,
                score=float(np.abs(half_auto[nid2, offset])))
