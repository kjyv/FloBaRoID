#!/usr/bin/env python3
"""Integration tests: generate synthetic measurement data from a known model,
run the identification pipeline (OLS and SDP), and verify that identified
parameters are close to the ground truth, torque prediction error is low,
and (for SDP) parameters are physically consistent."""

import os
import sys
import tempfile

import numpy as np
import numpy.linalg as la
import pytest
import yaml
from idyntree import bindings as iDynTree

# paths relative to project root
_project_dir = os.path.join(os.path.dirname(__file__), "..")
_urdf_file = os.path.join(_project_dir, "model", "kuka_lwr4.urdf")
_config_file = os.path.join(_project_dir, "configs", "kuka_lwr4.yaml")

sys.path.insert(0, _project_dir)


def _generate_synthetic_data(urdf_file, n_samples=2000, noise_std=0.05, seed=42):
    """Simulate torques from known URDF parameters for random joint trajectories.

    Returns a dict matching the measurement npz format.
    """
    rng = np.random.default_rng(seed)

    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(urdf_file)
    model = loader.model()
    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(model)
    n_dofs = kinDyn.getNrOfDegreesOfFreedom()

    gravity = iDynTree.Vector3()
    gravity.setVal(2, -9.81)

    from identification.helpers import URDFHelpers

    limits = URDFHelpers.getJointLimits(urdf_file, use_deg=False)
    joint_names = [model.getJointName(i) for i in range(model.getNrOfDOFs())]
    q_lo = np.array([limits[j]["lower"] for j in joint_names])
    q_hi = np.array([limits[j]["upper"] for j in joint_names])
    dq_max = np.array([limits[j]["velocity"] for j in joint_names])

    positions = np.zeros((n_samples, n_dofs))
    velocities = np.zeros((n_samples, n_dofs))
    accelerations = np.zeros((n_samples, n_dofs))
    torques = np.zeros((n_samples, n_dofs))
    dt = 1.0 / 200.0
    times = np.arange(n_samples) * dt

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
    """Generate synthetic data once and provide path to temp npz file."""
    synth = _generate_synthetic_data(_urdf_file, n_samples=2000, noise_std=0.05)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
        np.savez(f, **synth)
    yield tmp_path
    os.unlink(tmp_path)


def _base_config():
    """Load config and set common overrides for testing."""
    with open(_config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config["floatingBase"] = 0
    config["identifyFriction"] = 0
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
    config["useRBDL"] = 0
    config["constrainUsingNL"] = 0
    config["filterRegressor"] = 0
    config["estimateWith"] = "std"
    config["restrictCOMtoHull"] = 0
    config["limitOverallMass"] = 0
    config["limitMassToApriori"] = 0
    config["randomSamples"] = 5000
    return config


def _cleanup_regressor_cache():
    """Remove cached regressor files so tests don't interfere with each other."""
    for suffix in [".regressor.npz", ".gravity_regressor.npz"]:
        cache = _urdf_file + suffix
        if os.path.exists(cache):
            os.unlink(cache)


def test_identification_ols(synth_data_path):
    """Test unconstrained OLS identification: base params and torque prediction."""
    from identifier import Identification

    config = _base_config()
    config["constrainToConsistent"] = 0

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # compare base parameters
        base_error = la.norm(idf.model.xBase - idf.model.xBaseModel)
        relative_base_error = base_error / la.norm(idf.model.xBaseModel)
        print(f"OLS base parameter error: {relative_base_error:.4%}")

        # check torque prediction
        idf.estimateRegressorTorques()
        residual = la.norm(idf.tauEstimated - idf.model.tauMeasured)
        relative_residual = residual * 100 / la.norm(idf.model.tauMeasured)
        print(f"OLS torque residual: {relative_residual:.4f}%")

        assert relative_base_error < 0.05, f"Base params too far from ground truth: {relative_base_error:.2%}"
        assert relative_residual < 1.0, f"Torque residual too high: {relative_residual:.4f}%"
    finally:
        _cleanup_regressor_cache()


def test_identification_sdp(synth_data_path):
    """Test SDP-constrained identification: physical consistency and torque prediction."""
    from identifier import Identification

    config = _base_config()
    config["constrainToConsistent"] = 1
    config["identifyClosestToCAD"] = 1
    config["identifyFriction"] = 1
    config["identifySymmetricVelFriction"] = 1
    config["limitOverallMass"] = 1
    config["limitMassVal"] = 16.0
    config["limitMassRange"] = 0.3
    config["limitMassToApriori"] = 1
    config["limitMassAprioriBoundary"] = 0.5
    config["restrictCOMtoHull"] = 1

    try:
        idf = Identification(config, _urdf_file, None, [[synth_data_path]], None, None)
        idf.estimateParameters()

        # check physical consistency of identified standard parameters
        # the SDP enforces positive mass and positive-definite inertia tensor;
        # the triangle inequality is a stricter condition not enforced by the SDP
        # (the regular pipeline also uses NoTriangle, see showTriangleConsistency config option)
        cons = idf.paramHelpers.checkPhysicalConsistencyNoTriangle(idf.model.xStd)
        inconsistent = [link for link, ok in cons.items() if not ok]
        print(f"SDP physical consistency (positive mass + PD inertia): {cons}")
        assert len(inconsistent) == 0, f"Identified parameters not physically consistent for links: {inconsistent}"

        # check torque prediction (SDP may have slightly higher residual than OLS
        # since it constrains the solution space)
        idf.estimateRegressorTorques()
        residual = la.norm(idf.tauEstimated - idf.model.tauMeasured)
        relative_residual = residual * 100 / la.norm(idf.model.tauMeasured)
        print(f"SDP torque residual: {relative_residual:.4f}%")

        assert relative_residual < 5.0, f"Torque residual too high: {relative_residual:.4f}%"

        # base params should still be reasonable (SDP trades accuracy for consistency)
        base_error = la.norm(idf.model.xBase - idf.model.xBaseModel)
        relative_base_error = base_error / la.norm(idf.model.xBaseModel)
        print(f"SDP base parameter error: {relative_base_error:.4%}")

    finally:
        _cleanup_regressor_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
