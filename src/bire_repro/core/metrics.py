"""Canonical area-weighted metrics and explicitly labelled legacy diagnostics."""

from __future__ import annotations

import numpy as np


def latitude_weights(lat_deg: np.ndarray, nx: int | None = None) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(lat_deg, dtype="f8")))
    weights = np.maximum(weights, 0.0)
    if nx is not None:
        weights = np.repeat(weights[:, None], nx, axis=1)
    return weights


def _broadcast_weights(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None):
    weights = np.asarray(weights, dtype="f8")
    if mask is not None:
        weights = weights * np.asarray(mask, dtype=bool)
    while weights.ndim < values.ndim:
        weights = weights[None, ...]
    return np.broadcast_to(weights, values.shape)


def weighted_rmse(
    prediction: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    error_sq = np.square(np.asarray(prediction) - np.asarray(truth))
    weight = _broadcast_weights(error_sq, weights, mask)
    finite = np.isfinite(error_sq)
    numerator = np.sum(np.where(finite, error_sq * weight, 0.0), axis=(-2, -1))
    denominator = np.sum(np.where(finite, weight, 0.0), axis=(-2, -1))
    return np.sqrt(numerator / denominator)


def legacy_unweighted_mse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Archived script's quantity, incorrectly named RMSE there."""
    return np.nanmean(np.square(np.asarray(prediction) - np.asarray(truth)), axis=(-2, -1))


def anomaly_correlation(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    pred_anomaly = np.asarray(prediction) - np.asarray(climatology)
    truth_anomaly = np.asarray(truth) - np.asarray(climatology)
    weight = _broadcast_weights(pred_anomaly, weights, mask)
    valid = np.isfinite(pred_anomaly) & np.isfinite(truth_anomaly)
    weighted = np.where(valid, weight, 0.0)
    numerator = np.sum(weighted * pred_anomaly * truth_anomaly, axis=(-2, -1))
    pred_norm = np.sum(weighted * np.square(pred_anomaly), axis=(-2, -1))
    truth_norm = np.sum(weighted * np.square(truth_anomaly), axis=(-2, -1))
    denominator = np.sqrt(pred_norm * truth_norm)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def speed(state: np.ndarray) -> np.ndarray:
    """Surface speed from canonical channel order U_surface, V_surface."""
    state = np.asarray(state)
    return np.hypot(state[..., 0, :, :], state[..., 2, :, :])


def ssh_anomaly(phihyd_surface: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    ssh = np.asarray(phihyd_surface) / gravity
    return ssh - np.nanmean(ssh, axis=(-2, -1), keepdims=True)


def percentile_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    return (
        np.nanmean(values, axis=0),
        np.nanpercentile(values, 10, axis=0),
        np.nanpercentile(values, 90, axis=0),
    )
