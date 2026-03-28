"""Tests for suspended base dynamics simulation and related utilities."""

import numpy as np


def test_angular_velocity_to_rpy_rates_inverse() -> None:
    """angular_velocity_to_rpy_rates should be the inverse of rpy_to_angular_velocity."""
    from excitation.simulationEffects import (
        angular_velocity_to_rpy_rates,
        rpy_to_angular_velocity,
    )

    rng = np.random.default_rng(42)
    # test at various RPY values (avoiding pitch near ±90° where E is singular)
    for _ in range(50):
        rpy = rng.uniform([-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
        rpy_dot = rng.uniform(-1.0, 1.0, size=3)

        omega = rpy_to_angular_velocity(rpy, rpy_dot)
        rpy_dot_recovered = angular_velocity_to_rpy_rates(rpy, omega)
        np.testing.assert_allclose(rpy_dot_recovered, rpy_dot, atol=1e-12)


def test_angular_velocity_to_rpy_rates_zero() -> None:
    """Zero angular velocity should give zero RPY rates."""
    from excitation.simulationEffects import angular_velocity_to_rpy_rates

    rpy = np.array([0.1, 0.2, 0.3])
    omega = np.zeros(3)
    rpy_dot = angular_velocity_to_rpy_rates(rpy, omega)
    np.testing.assert_allclose(rpy_dot, np.zeros(3), atol=1e-15)


def test_static_equilibrium_base_stays_small() -> None:
    """With zero joint velocities and accelerations, base swing should be small.

    At q=0 the robot's COM may not be directly below the attachment point, so
    gravity will create a small swing torque. Over a short window (0.25s) the
    resulting drift should be bounded.
    """
    from excitation.suspendedDynamics import simulate_suspended_base_motion

    urdf = "model/walkman.urdf"
    n_dofs = 29
    num_samples = 50
    freq = 200.0
    times = np.arange(num_samples) / freq

    positions = np.zeros((num_samples, n_dofs))
    velocities = np.zeros((num_samples, n_dofs))
    accelerations = np.zeros((num_samples, n_dofs))

    base_rpy, base_vel, base_acc, _base_pos = simulate_suspended_base_motion(
        urdf,
        positions,
        velocities,
        accelerations,
        times,
    )

    assert base_rpy.shape == (num_samples, 3)
    assert base_vel.shape == (num_samples, 6)
    assert base_acc.shape == (num_samples, 6)

    # base swing should stay small over 0.25s (< 5° ≈ 0.087 rad)
    max_swing = np.max(np.abs(base_rpy))
    assert max_swing < 0.1, f"Base drifted too much in static config: {np.degrees(max_swing):.2f}°"


def test_joint_motion_produces_base_swing() -> None:
    """Moving joints should cause the base to swing (nonzero base RPY)."""
    from excitation.suspendedDynamics import simulate_suspended_base_motion

    urdf = "model/walkman.urdf"
    n_dofs = 29
    num_samples = 200
    freq = 200.0
    times = np.arange(num_samples) / freq

    positions = np.zeros((num_samples, n_dofs))
    velocities = np.zeros((num_samples, n_dofs))
    accelerations = np.zeros((num_samples, n_dofs))

    # apply a sinusoidal acceleration to a hip joint (joint 4 = RHipSag typically)
    w = 2.0 * np.pi * 1.0  # 1 Hz
    accelerations[:, 4] = 5.0 * np.sin(w * times)
    velocities[:, 4] = -5.0 / w * np.cos(w * times) + 5.0 / w
    positions[:, 4] = -5.0 / (w**2) * np.sin(w * times) + 5.0 / w * times

    base_rpy, base_vel, base_acc, _base_pos = simulate_suspended_base_motion(
        urdf,
        positions,
        velocities,
        accelerations,
        times,
    )

    # base should have moved — RPY should be nonzero
    max_swing = np.max(np.abs(base_rpy))
    assert max_swing > 1e-4, f"Expected nonzero base swing, got max RPY = {max_swing}"

    # but should stay within reasonable bounds (not diverging)
    assert max_swing < 1.0, f"Base swing too large: {np.degrees(max_swing):.1f}°"
