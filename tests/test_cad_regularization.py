"""Tests for the observability-weighted CAD regularization in the SDP identification."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MEAS = "data/THREELINK/SIM/measurements_opt1_fb.npz"


def _identify(monkeypatch, mode):
    """Run identification on the threeLink sim data with the given cadRegularizationMode."""
    monkeypatch.chdir(PROJECT_ROOT)
    import yaml
    from idyntree import bindings as iDynTree

    from identifier import Identification

    with open("configs/threeLinks.yaml") as f:
        config = yaml.safe_load(f)
    config["urdf"] = "model/threeLinks.urdf"
    config["jointNames"] = iDynTree.StringVector([])
    iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"])
    config["num_dofs"] = len(config["jointNames"])
    config["verbose"] = 0
    config["cadRegularizationMode"] = mode
    idf = Identification(config, config["urdf"], None, [[MEAS]], None, None)
    idf.estimateParameters()
    return idf


def test_geometric_mode_runs_and_stays_physical(monkeypatch):
    """Geometric (log-det Bregman divergence) mode completes, yields a finite and
    physically-plausible std-param vector, and produces a decomposition differing from
    the uniform Euclidean pull.

    The geometric prior pulls each link's pseudo-inertia toward CAD on the SPD manifold,
    so it keeps masses positive and the decomposition close to CAD; like the
    observability mode it trades a little torque-fit for CAD-closeness on the tiny
    threeLink, so fit-invariance is not asserted.
    """
    idf_u = _identify(monkeypatch, "uniform")
    idf_g = _identify(monkeypatch, "geometric")

    xstd_u = np.asarray(idf_u.model.xStd, dtype=float)
    xstd_g = np.asarray(idf_g.model.xStd, dtype=float)

    assert xstd_g.shape == xstd_u.shape
    assert np.all(np.isfinite(xstd_g))
    # link masses (every 10th std param) stay positive — the divergence diverges at zero
    # mass, so it must not produce a degenerate link
    assert np.all(xstd_g[0 : idf_g.model.num_model_params : 10] > -1e-9)
    # the mode is actually active: the standard-parameter decomposition differs
    assert not np.allclose(xstd_g, xstd_u)


def test_observability_mode_runs_and_changes_decomposition(monkeypatch):
    """Observability mode completes, yields a finite physically-plausible std-param vector,
    and produces a decomposition that differs from the uniform mode.

    Note: the mode trades a little torque-fit for staying near CAD in weakly-determined
    directions, and how much it shifts the fit is system-dependent (negligible on the
    well-conditioned walkman, noticeable on the tiny threeLink). So this does not assert
    fit-invariance — only that the run is well-behaved and the mode is active.
    """
    idf_u = _identify(monkeypatch, "uniform")
    idf_o = _identify(monkeypatch, "observability")

    xstd_u = np.asarray(idf_u.model.xStd, dtype=float)
    xstd_o = np.asarray(idf_o.model.xStd, dtype=float)

    assert xstd_o.shape == xstd_u.shape
    assert np.all(np.isfinite(xstd_o))
    # link masses (every 10th std param) stay positive — basic physical-plausibility guard
    assert np.all(xstd_o[0 : idf_o.model.num_model_params : 10] > -1e-9)
    # the mode is actually active: the standard-parameter decomposition differs
    assert not np.allclose(xstd_o, xstd_u)
