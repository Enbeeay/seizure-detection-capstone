"""
Advanced EEG window features for seizure / preictal modeling (connectivity & geometry).

Why these features (ablation-friendly complements to time/spectral baselines):
- PLV captures *nonlinear phase synchronization* between channels (phase locking), which
  linear measures like correlation cannot fully characterize; it is often sensitive to
  preictal network dynamics in scalp and intracranial EEG.
- Riemannian / covariance-manifold features treat each window's channel covariance as a
  symmetric positive-definite (SPD) object and embed it in a geometry better matched to
  SPD matrices than raw Euclidean vectorization of the matrix entries.

Optional dependency: install ``pyriemann`` for the ``tangent_space`` Riemannian mode using
the library's reference-based tangent map; without it, both modes use stable NumPy/SciPy
fallbacks (see ``riemannian_mode`` below).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.linalg import logm
from scipy.signal import butter, hilbert, sosfiltfilt

logger = logging.getLogger(__name__)

try:
    from pyriemann.tangentspace import tangent_space as pyriemann_tangent_space
except ImportError:  # pragma: no cover - optional
    pyriemann_tangent_space = None

# (name, low_hz, high_hz) — low_gamma included when Nyquist allows (see band validity).
PLV_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 45.0),
)

PlvFeatureMode = Literal["summary", "full_matrix"]
RiemannianMode = Literal["tangent_space", "log_euclidean"]


@dataclass(frozen=True)
class AdvancedFeatureConfig:
    """Toggles for PLV / Riemannian blocks (use with ``extract_combined_features``)."""

    use_plv: bool = False
    use_riemannian: bool = False
    use_existing_features: bool = True
    plv_feature_mode: PlvFeatureMode = "summary"
    riemannian_mode: RiemannianMode = "tangent_space"
    covariance_regularization: float = 1e-6
    include_low_gamma_plv: bool = True
    plv_filter_order: int = 4


def _nyquist(sfreq: float) -> float:
    return 0.5 * float(sfreq)


def band_usable_for_filter(low_hz: float, high_hz: float, sfreq: float, order: int = 4) -> bool:
    """Return False if band edges are too close to 0 or Nyquist for a stable bandpass."""
    nyq = _nyquist(sfreq)
    if low_hz <= 0 or high_hz <= low_hz or nyq <= 0:
        return False
    # Leave margin for digital bandpass (Butterworth).
    margin = 0.05 * nyq
    if high_hz >= nyq - margin or low_hz >= nyq - margin:
        return False
    return True


@lru_cache(maxsize=256)
def _bandpass_sos(low_hz: float, high_hz: float, sfreq: float, order: int):
    """Design bandpass as second-order sections (cached per band / rate / order)."""
    return butter(order, [low_hz, high_hz], btype="band", fs=sfreq, output="sos")


def _bandpass_filter(data_1d: np.ndarray, low_hz: float, high_hz: float, sfreq: float, order: int) -> np.ndarray:
    sos = _bandpass_sos(low_hz, high_hz, sfreq, order)
    return sosfiltfilt(sos, data_1d)


def _pairwise_plv_matrix(phases: np.ndarray) -> np.ndarray:
    """
    phases: (n_channels, n_samples), real instantaneous phase in radians.
    PLV_ij = | E_t[ exp(i (phi_i - phi_j)) ] | in [0, 1].
    """
    z = np.exp(1j * np.asarray(phases, dtype=np.complex128))
    # (C, 1, T) * conj(1, C, T) -> mean over T gives cross-correlation of analytic phases.
    prod = z[:, np.newaxis, :] * np.conj(z[np.newaxis, :, :])
    plv = np.abs(np.mean(prod, axis=2))
    np.fill_diagonal(plv, 0.0)
    return np.clip(plv.astype(np.float64), 0.0, 1.0)


def _upper_triangle_values(mat: np.ndarray, *, exclude_diagonal: bool) -> np.ndarray:
    n = mat.shape[0]
    if exclude_diagonal:
        iu = np.triu_indices(n, k=1)
    else:
        iu = np.triu_indices(n, k=0)
    return mat[iu].astype(np.float64)


def plv_feature_dimension(n_channels: int, mode: PlvFeatureMode, *, n_bands: int) -> int:
    if mode == "summary":
        return n_bands * 3
    n_pairs = max(0, n_channels * (n_channels - 1) // 2)
    return n_bands * n_pairs


def riemannian_feature_dimension(n_channels: int) -> int:
    """Upper-triangular embedding of symmetric tangent / log matrix: n(n+1)/2."""
    return n_channels * (n_channels + 1) // 2


def extract_plv_features(
    window_data: np.ndarray,
    sfreq: float,
    *,
    feature_mode: PlvFeatureMode = "summary",
    include_low_gamma: bool = True,
    filter_order: int = 4,
    warned_bands: set[str] | None = None,
) -> np.ndarray:
    """
    Phase Locking Value (PLV) features for one window.

    window_data: (n_channels, n_samples)
    Phase via band-limited analytic signal: bandpass + Hilbert transform.
    """
    x = np.asarray(window_data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("window_data must be 2D (channels, samples)")
    n_ch, n_t = x.shape
    warned_bands = warned_bands if warned_bands is not None else set()

    bands = list(PLV_BANDS)
    if not include_low_gamma:
        bands = [b for b in bands if b[0] != "low_gamma"]

    n_bands_eff = len(bands)
    if n_ch < 2:
        dim = plv_feature_dimension(n_ch, feature_mode, n_bands=n_bands_eff)
        return np.zeros(dim, dtype=np.float32)

    feats: list[float] = []
    for name, low_hz, high_hz in bands:
        if not band_usable_for_filter(low_hz, high_hz, sfreq, filter_order):
            if name not in warned_bands:
                logger.warning(
                    "Skipping PLV band %s (%.2f–%.2f Hz): incompatible with sfreq=%.2f Hz",
                    name,
                    low_hz,
                    high_hz,
                    sfreq,
                )
                warned_bands.add(name)
            if feature_mode == "summary":
                feats.extend([0.0, 0.0, 0.0])
            else:
                feats.extend([0.0] * (n_ch * (n_ch - 1) // 2))
            continue

        phases = np.empty_like(x, dtype=np.float64)
        for c in range(n_ch):
            band_sig = _bandpass_filter(x[c], low_hz, high_hz, sfreq, filter_order)
            phases[c, :] = np.angle(hilbert(band_sig))

        plv_mat = _pairwise_plv_matrix(phases)
        if feature_mode == "summary":
            ut = _upper_triangle_values(plv_mat, exclude_diagonal=True)
            if ut.size == 0:
                feats.extend([0.0, 0.0, 0.0])
            else:
                feats.extend([float(np.mean(ut)), float(np.std(ut)), float(np.max(ut))])
        else:
            feats.extend(_upper_triangle_values(plv_mat, exclude_diagonal=True).tolist())

    return np.asarray(feats, dtype=np.float32)


def _empirical_covariance(window_data: np.ndarray, reg: float) -> np.ndarray:
    """Sample covariance of (channels, samples), symmetrized + diagonal regularization."""
    x = np.asarray(window_data, dtype=np.float64)
    x = x - np.mean(x, axis=1, keepdims=True)
    n = x.shape[1]
    denom = max(n - 1, 1)
    c = (x @ x.T) / denom
    c = 0.5 * (c + c.T)
    if not np.allclose(c, c.T, atol=1e-9):
        raise ValueError("Covariance must be symmetric after construction")
    n_ch = c.shape[0]
    c = c + float(reg) * np.eye(n_ch, dtype=np.float64)
    return c


def _symmetric_upper_flat(sym_mat: np.ndarray) -> np.ndarray:
    n = sym_mat.shape[0]
    iu = np.triu_indices(n, k=0)
    return np.real(sym_mat[iu]).astype(np.float64)


def extract_riemannian_features(
    window_data: np.ndarray,
    *,
    mode: RiemannianMode = "tangent_space",
    covariance_regularization: float = 1e-6,
) -> np.ndarray:
    """
    Map window covariance to a fixed-size vector.

    - log_euclidean: matrix logarithm log(C) (symmetric), upper triangle (incl. diagonal).
    - tangent_space: if pyriemann is installed, Riemannian log-map relative to I via
      ``pyriemann.tangentspace.tangent_space``; otherwise same numeric path as log-Euclidean
      at identity (documented fallback; install pyriemann for the library metric).
    """
    x = np.asarray(window_data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("window_data must be 2D (channels, samples)")
    n_ch = x.shape[0]
    c = _empirical_covariance(x, covariance_regularization)

    if mode == "log_euclidean":
        lmat = logm(c)
        lmat = 0.5 * (lmat + lmat.conj().T)
        return _symmetric_upper_flat(lmat).astype(np.float32)

    if mode == "tangent_space" and pyriemann_tangent_space is not None:
        cref = np.eye(n_ch, dtype=np.float64)
        tvec = pyriemann_tangent_space(c.astype(np.float64), cref, metric="riemann")
        return np.asarray(np.real(tvec).ravel(), dtype=np.float32)

    # Fallback tangent_space without pyriemann: log-Euclidean at identity (stable SPD log).
    lmat = logm(c)
    lmat = 0.5 * (lmat + lmat.conj().T)
    return _symmetric_upper_flat(lmat).astype(np.float32)


def extract_combined_features(
    window_data: np.ndarray,
    sfreq: float,
    config: AdvancedFeatureConfig,
    *,
    warned_plv_bands: set[str] | None = None,
) -> np.ndarray:
    """Concatenate enabled advanced feature blocks (PLV then Riemannian)."""
    parts: list[np.ndarray] = []
    if config.use_plv:
        parts.append(
            extract_plv_features(
                window_data,
                sfreq,
                feature_mode=config.plv_feature_mode,
                include_low_gamma=config.include_low_gamma_plv,
                filter_order=config.plv_filter_order,
                warned_bands=warned_plv_bands,
            )
        )
    if config.use_riemannian:
        parts.append(
            extract_riemannian_features(
                window_data,
                mode=config.riemannian_mode,
                covariance_regularization=config.covariance_regularization,
            )
        )
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def advanced_feature_dimension(n_channels: int, config: AdvancedFeatureConfig) -> int:
    bands = list(PLV_BANDS)
    if not config.include_low_gamma_plv:
        bands = [b for b in bands if b[0] != "low_gamma"]
    n_b = len(bands)
    d = 0
    if config.use_plv:
        d += plv_feature_dimension(n_channels, config.plv_feature_mode, n_bands=n_b)
    if config.use_riemannian:
        d += riemannian_feature_dimension(n_channels)
    return d


def validate_feature_matrix(vectors: np.ndarray, *, name: str = "features") -> None:
    if not np.isfinite(vectors).all():
        bad = int(np.size(vectors) - np.sum(np.isfinite(vectors)))
        raise ValueError(f"{name}: found {bad} non-finite values")
    print(f"[validate] {name}: shape={vectors.shape} dtype={vectors.dtype}")


def sanity_check_window(
    window_data_ct: np.ndarray,
    sfreq: float,
    config: AdvancedFeatureConfig,
) -> None:
    """
    Diagnostics on one window: shapes, finiteness, PLV in [0,1], covariance symmetry.
    window_data_ct: (channels, samples)
    """
    if config.use_plv:
        plv = extract_plv_features(
            window_data_ct,
            sfreq,
            feature_mode=config.plv_feature_mode,
            include_low_gamma=config.include_low_gamma_plv,
            filter_order=config.plv_filter_order,
        )
        validate_feature_matrix(plv[np.newaxis, :], name="PLV row")
        x = np.asarray(window_data_ct, dtype=np.float64)
        if x.shape[0] >= 2 and band_usable_for_filter(8.0, 13.0, sfreq, config.plv_filter_order):
            phases = np.empty_like(x)
            for c in range(x.shape[0]):
                band_sig = _bandpass_filter(x[c], 8.0, 13.0, sfreq, config.plv_filter_order)
                phases[c, :] = np.angle(hilbert(band_sig))
            pm = _pairwise_plv_matrix(phases)
            if (pm < -1e-6).any() or (pm > 1.0 + 1e-4).any():
                raise AssertionError(f"PLV out of [0,1]: min={pm.min()}, max={pm.max()}")

    if config.use_riemannian:
        c = _empirical_covariance(np.asarray(window_data_ct, dtype=np.float64), config.covariance_regularization)
        if not np.allclose(c, c.T):
            raise AssertionError("Covariance not symmetric")
        rv = extract_riemannian_features(
            window_data_ct,
            mode=config.riemannian_mode,
            covariance_regularization=config.covariance_regularization,
        )
        validate_feature_matrix(rv[np.newaxis, :], name="Riemannian row")


def demo_single_clip() -> None:
    """Synthetic (channels, samples) demo through PLV + Riemannian extractors."""
    rng = np.random.default_rng(0)
    n_ch, n_t = 6, 1000
    sfreq = 100.0
    window = rng.standard_normal((n_ch, n_t)).astype(np.float32)
    cfg = AdvancedFeatureConfig(
        use_plv=True,
        use_riemannian=True,
        plv_feature_mode="summary",
        riemannian_mode="log_euclidean",
    )
    plv = extract_plv_features(window, sfreq, feature_mode="summary")
    riem = extract_riemannian_features(window, mode="log_euclidean")
    comb = extract_combined_features(window, sfreq, cfg)
    print("demo PLV dim", plv.shape, "Riemannian dim", riem.shape, "combined", comb.shape)
    assert comb.shape[0] == plv.shape[0] + riem.shape[0]
    assert np.isfinite(comb).all()
    sanity_check_window(window, sfreq, cfg)
    print("demo OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_single_clip()
