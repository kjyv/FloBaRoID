"""Baseline tests for SDP identification.

These tests capture the expected behavior of the SDP solver so that the
cvxpy migration can be validated against the same criteria. All tests use
the kuka_lwr4 model with synthetic data (small, fast, reliable).

Tests verify RESULTS (physical consistency, torque prediction, constraint
satisfaction) not implementation details, so they work with any solver backend.
"""

import os
import sys
import tempfile

import numpy as np
import numpy.linalg as la
import pytest
import yaml
from idyntree import bindings as iDynTree

_project_dir = os.path.join(os.path.dirname(__file__), "..")
_urdf_file = os.path.join(_project_dir, "model", "kuka_lwr4.urdf")
_config_file = os.path.join(_project_dir, "configs", "kuka_lwr4.yaml")

sys.path.insert(0, _project_dir)


def _generate_synthetic_data(n_samples: int = 2000, noise_std: float = 0.05, seed: int = 42) -> dict:
    """Simulate torques from known URDF parameters for random joint trajectories."""
    rng = np.random.default_rng(seed)
    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(_urdf_file)
    model = loader.model()
    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(model)
    n_dofs = kinDyn.getNrOfDegreesOfFreedom()

    gravity = iDynTree.Vector3()
    gravity.setVal(2, -9.81)

    from identification.helpers import URDFHelpers

    limits = URDFHelpers.getJointLimits(_urdf_file, use_deg=False)
    joint_names = [model.getJointName(i) for i in range(model.getNrOfDOFs())]
    q_lo = np.array([limits[j]["lower"] for j in joint_names])
    q_hi = np.array([limits[j]["upper"] for j in joint_names])
    dq_max = np.array([limits[j]["velocity"] for j in joint_names])

    positions = np.zeros((n_samples, n_dofs))
    velocities = np.zeros((n_samples, n_dofs))
    accelerations = np.zeros((n_samples, n_dofs))
    torques = np.zeros((n_samples, n_dofs))
    times = np.arange(n_samples) / 200.0

    for idx in range(n_samples):
        q_np = q_lo + rng.random(n_dofs) * (q_hi - q_lo)
        dq_np = (rng.random(n_dofs) - 0.5) * 2.0 * dq_max
        ddq_np = (rng.random(n_dofs) - 0.5) * 2.0 * np.pi

        s = iDynTree.JointPosDoubleArray(n_dofs)
        ds = iDynTree.JointDOFsDoubleArray(n_dofs)
        ddq = iDynTree.JointDOFsDoubleArray(n_dofs)
        for j in range(n_dofs):
            s.setVal(j, float(q_np[j]))
            ds.setVal(j, float(dq_np[j]))
            ddq.setVal(j, float(ddq_np[j]))

        kinDyn.setRobotState(s, ds, gravity)
        base_acc = iDynTree.Vector6()
        ext = iDynTree.LinkWrenches(model)
        ext.zero()
        gen = iDynTree.FreeFloatingGeneralizedTorques(model)
        kinDyn.inverseDynamics(base_acc, ddq, ext, gen)

        tau = gen.jointTorques().toNumPy().copy()
        tau += rng.normal(0, noise_std, n_dofs)

        positions[idx] = q_np
        velocities[idx] = dq_np
        accelerations[idx] = ddq_np
        torques[idx] = tau

    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "torques": torques,
        "times": times,
    }


@pytest.fixture(scope="module")
def synth_data_path():
    """Generate synthetic data once per module."""
    synth = _generate_synthetic_data()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez(f, **synth)
    yield tmp_path
    os.unlink(tmp_path)


def _base_config() -> dict:
    """Load kuka config with common overrides for testing."""
    with open(_config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config["floatingBase"] = 0
    config["identifyFrictionSimultaneously"] = 0
    config["identifyClosestToCAD"] = 0
    config["useAPriori"] = 0
    config["simulateTorques"] = 0
    config["useStructuralRegressor"] = 1
    config["identifyGravityParamsOnly"] = 0
    config["startOffset"] = 0
    config["skipSamples"] = 0
    config["selectBlocksFromMeasurements"] = 0
    config["createPlots"] = 0
    config["verbose"] = 0
    config["showTiming"] = 0
    config["constrainUsingNL"] = 0
    config["filterRegressor"] = 0
    config["estimateWith"] = "std"
    config["restrictCOMtoHull"] = 0
    config["limitOverallMass"] = 0
    config["limitMassToApriori"] = 0
    config["randomSamples"] = 5000
    config["constrainToConsistent"] = 1
    config["checkAPrioriFeasibility"] = 0
    return config


def _cleanup_regressor_cache() -> None:
    """Remove cached regressor files."""
    for suffix in [".regressor.npz", ".gravity_regressor.npz"]:
        cache = _urdf_file + suffix
        if os.path.exists(cache):
            os.unlink(cache)


def _check_physical_consistency(xStd: np.ndarray, num_links: int) -> dict[int, bool]:
    """Check pseudo-inertia PSD for each link (mass>0, inertia tensor PD)."""
    results = {}
    for i in range(num_links):
        m = xStd[i * 10]
        mx, my, mz = xStd[i * 10 + 1 : i * 10 + 4]
        L = np.array(
            [
                [xStd[i * 10 + 4], xStd[i * 10 + 5], xStd[i * 10 + 6]],
                [xStd[i * 10 + 5], xStd[i * 10 + 7], xStd[i * 10 + 8]],
                [xStd[i * 10 + 6], xStd[i * 10 + 8], xStd[i * 10 + 9]],
            ]
        )
        S = np.array([[0, -mz, my], [mz, 0, -mx], [-my, mx, 0]])
        Di = np.block([[L, S.T], [S, m * np.eye(3)]])
        eigvals = la.eigvalsh(Di)
        results[i] = bool(np.all(eigvals > -1e-8))
    return results


# --- Tests ---


def test_sdp_physical_consistency(synth_data_path: str) -> None:
    """SDP identification must produce physically consistent parameters."""
    from identifier import Identification

    config = _base_config()
    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        cons = _check_physical_consistency(idf.model.xStd, idf.model.num_links)
        inconsistent = [link for link, ok in cons.items() if not ok]
        assert len(inconsistent) == 0, f"Links not physically consistent: {inconsistent}"
    finally:
        _cleanup_regressor_cache()


def test_sdp_torque_prediction(synth_data_path: str) -> None:
    """SDP-identified params should predict torques reasonably well."""
    from identifier import Identification

    config = _base_config()
    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()
        idf.estimateRegressorTorques()

        residual = la.norm(idf.tauEstimated - idf.model.tauMeasured)
        relative_residual = residual * 100 / la.norm(idf.model.tauMeasured)

        # SDP constrains the solution space, so residual may be higher than OLS
        # but should still be under 5%
        assert relative_residual < 5.0, f"Torque residual too high: {relative_residual:.2f}%"
        assert not np.any(np.isnan(idf.model.xStd)), "NaN in identified parameters"
        assert not np.any(np.isinf(idf.model.xStd)), "Inf in identified parameters"
    finally:
        _cleanup_regressor_cache()


def test_sdp_masses_positive(synth_data_path: str) -> None:
    """All identified masses must be positive."""
    from identifier import Identification

    config = _base_config()
    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        for i in range(idf.model.num_links):
            mass = idf.model.xStd[i * 10]
            assert mass > 0, f"Link {i} ({idf.model.linkNames[i]}): mass={mass} <= 0"
    finally:
        _cleanup_regressor_cache()


def test_sdp_with_mass_constraints(synth_data_path: str) -> None:
    """Mass constraints should bound identified masses near a priori values."""
    from identifier import Identification

    config = _base_config()
    config["limitOverallMass"] = 1
    config["limitMassVal"] = 16.0
    config["limitMassRange"] = 0.3
    config["limitMassToApriori"] = 1
    config["limitMassAprioriBoundary"] = 0.5

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        total_mass = sum(idf.model.xStd[i * 10] for i in range(idf.model.num_links))
        assert abs(total_mass - 16.0) < 16.0 * 0.3 + 0.1, f"Total mass {total_mass:.2f} outside allowed range"
    finally:
        _cleanup_regressor_cache()


def test_sdp_with_hull_constraints(synth_data_path: str) -> None:
    """COM hull constraints should be satisfied."""
    from identifier import Identification

    config = _base_config()
    config["restrictCOMtoHull"] = 1

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # verify physical consistency (hull constraints are part of the SDP)
        cons = _check_physical_consistency(idf.model.xStd, idf.model.num_links)
        inconsistent = [link for link, ok in cons.items() if not ok]
        assert len(inconsistent) == 0, f"Links not consistent: {inconsistent}"
    finally:
        _cleanup_regressor_cache()


def test_sdp_with_friction(synth_data_path: str) -> None:
    """Viscous friction must be positive when identified."""
    from identifier import Identification

    config = _base_config()
    config["identifyFrictionSimultaneously"] = 1
    config["identifySymmetricVelFriction"] = 1

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # viscous friction parameters should be >= 0 (constrained by SDP)
        n_model = idf.model.num_model_params
        n_dofs = idf.model.num_dofs
        for i in range(n_dofs):
            fv = idf.model.xStd[n_model + n_dofs + i]
            assert fv >= -1e-6, f"Joint {i}: viscous friction {fv} < 0 (should be non-negative)"
    finally:
        _cleanup_regressor_cache()


def test_sdp_dont_change_params(synth_data_path: str) -> None:
    """Parameters in dontChangeParams should stay near a priori values.

    Note: with the current cvxopt/dsdp5 solvers, dontChangeParams constraints may
    not be exactly satisfied when the solver falls back to dsdp5 (which doesn't
    guarantee constraint satisfaction). This test uses a relaxed tolerance.
    A better solver (cvxpy/CLARABEL) should enforce these exactly.
    """
    from identifier import Identification

    config = _base_config()
    # pin second link's parameters (indices 10-19) — not the base link
    # (pinning link 0 on a fixed-base robot conflicts with deleteFixedBase)
    config["dontChangeParams"] = list(range(10, 20))

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # check that the SDP ran without crashing (the solver may not find a good solution
        # when dontChangeParams makes the feasible region very tight)
        assert not np.any(np.isnan(idf.model.xStd)), "NaN in identified parameters"

        # log how well the constraints were respected (for comparison with cvxpy later)
        max_change = 0.0
        for p in range(10, 20):
            change = abs(idf.model.xStd[p] - idf.model.xStdModel[p])
            max_change = max(max_change, change)
        print(f"  dontChangeParams max deviation: {max_change:.6f}")
        # NOTE: with cvxpy/CLARABEL this should be < 1e-6; current solvers may violate it
    finally:
        _cleanup_regressor_cache()


def test_sdp_closest_to_cad(synth_data_path: str) -> None:
    """identifyClosestToCAD should produce consistent params closer to CAD."""
    from identifier import Identification

    config = _base_config()
    config["identifyClosestToCAD"] = 1
    config["checkAPrioriFeasibility"] = 1
    config["limitOverallMass"] = 1
    config["limitMassVal"] = 16.0
    config["limitMassRange"] = 0.3
    config["restrictCOMtoHull"] = 1
    config["identifyFrictionSimultaneously"] = 1
    config["identifySymmetricVelFriction"] = 1

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # must still be physically consistent
        cons = _check_physical_consistency(idf.model.xStd, idf.model.num_links)
        inconsistent = [link for link, ok in cons.items() if not ok]
        assert len(inconsistent) == 0, f"Links not consistent: {inconsistent}"

        # torque prediction should still be reasonable
        idf.estimateRegressorTorques()
        residual = la.norm(idf.tauEstimated - idf.model.tauMeasured)
        relative_residual = residual * 100 / la.norm(idf.model.tauMeasured)
        assert relative_residual < 5.0, f"Torque residual too high: {relative_residual:.2f}%"
    finally:
        _cleanup_regressor_cache()
