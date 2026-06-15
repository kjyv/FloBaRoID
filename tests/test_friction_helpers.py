"""Tests for the Coulomb friction sign velocity/series helpers."""

import numpy as np
import pytest

from identification.helpers import getFrictionSignSeries, getFrictionSignVelocities

OPT = {"frictionVelocityCutoff": 3.0, "frictionSignThreshold": 0.02}


@pytest.fixture
def noisy_samples():
    """Synthetic samples dict: slow sine velocity (0.5 Hz) plus measurement noise."""
    rng = np.random.default_rng(0)
    freq = 200.0
    t = np.arange(2000) / freq
    v_true = 0.3 * np.sin(2 * np.pi * 0.5 * t)[:, np.newaxis]
    noisy = v_true + rng.normal(0, 0.0166, v_true.shape)
    samples = {
        "velocities": noisy.copy(),
        "velocities_raw": noisy.copy(),
        "frequency": np.float64(freq),
    }
    return samples, v_true


def test_sign_velocities_filtering_reduces_noise(noisy_samples):
    """The filtered sign velocities are closer to the true velocity than the raw measurement."""
    samples, v_true = noisy_samples
    v_sign = getFrictionSignVelocities(samples, OPT)
    err_filtered = np.sqrt(np.mean((v_sign - v_true) ** 2))
    err_raw = np.sqrt(np.mean((samples["velocities_raw"] - v_true) ** 2))
    assert err_filtered < err_raw / 2


def test_sign_velocities_cached(noisy_samples):
    """Repeated calls return the cached array from the samples dict."""
    samples, _ = noisy_samples
    first = getFrictionSignVelocities(samples, OPT)
    assert "velocities_for_sign" in samples
    assert getFrictionSignVelocities(samples, OPT) is first


def test_sign_velocities_fallback_without_raw():
    """Without raw velocities or frequency, the pipeline velocities are used unchanged."""
    samples = {"velocities": np.ones((10, 2))}
    v_sign = getFrictionSignVelocities(samples, OPT)
    assert v_sign is samples["velocities"]


def test_sign_velocities_fallback_cutoff_above_nyquist(noisy_samples):
    """A cutoff at or above Nyquist disables the filter (pipeline velocities used)."""
    samples, _ = noisy_samples
    opt = dict(OPT, frictionVelocityCutoff=1000.0)
    v_sign = getFrictionSignVelocities(samples, opt)
    assert v_sign is samples["velocities"]


def test_sign_series_is_tanh_of_sign_velocities(noisy_samples):
    """The sign series equals tanh(v_sign / threshold) and stays within [-1, 1]."""
    samples, _ = noisy_samples
    series = getFrictionSignSeries(samples, OPT)
    v_sign = samples["velocities_for_sign"]
    assert np.allclose(series, np.tanh(v_sign / OPT["frictionSignThreshold"]))
    assert np.all(np.abs(series) <= 1.0)


def test_sign_series_reduces_chatter(noisy_samples):
    """The filtered sign series chatters far less than tanh of the raw velocities."""
    samples, _ = noisy_samples
    series = getFrictionSignSeries(samples, OPT)
    raw_series = np.tanh(samples["velocities_raw"] / OPT["frictionSignThreshold"])
    chatter = np.sqrt(np.mean(np.diff(series, axis=0) ** 2))
    chatter_raw = np.sqrt(np.mean(np.diff(raw_series, axis=0) ** 2))
    assert chatter < chatter_raw / 3


def test_sign_series_cached(noisy_samples):
    """Repeated calls return the cached series from the samples dict."""
    samples, _ = noisy_samples
    first = getFrictionSignSeries(samples, OPT)
    assert "friction_sign_series" in samples
    assert getFrictionSignSeries(samples, OPT) is first
