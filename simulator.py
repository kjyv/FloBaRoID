#!/usr/bin/env python

"""Simulate realistic measurement data from optimized trajectories.

Computes inverse dynamics torques from a trajectory file and adds configurable
real-world effects (friction, noise, backlash, etc.) to produce simulated
measurements in the same format as real robot data. This allows benchmarking
trajectory optimization and identification without a physical robot.

Usage:
  uv run simulator.py --config configs/myrobot.yaml --model model/myrobot.urdf \
      --trajectory model/myrobot.urdf.trajectory.npz --filename measurements.npz
"""

import argparse
import os
import sys
from typing import Any

import numpy as np
import yaml
from idyntree import bindings as iDynTree

from excitation.simulationEffects import (
    JointProperties,
    add_backlash,
    add_cable_forces,
    add_encoder_quantization,
    add_friction,
    add_gravity_compensation_residual,
    add_joint_elasticity,
    add_sensor_noise,
    add_structural_deflection,
    add_temperature_friction_drift,
    add_timing_jitter,
    add_torque_quantization,
    add_torque_ripple,
)
from identification.data import Data
from identification.model import Model

parser = argparse.ArgumentParser(description="Simulate realistic measurements from a trajectory file.")
parser.add_argument("--config", required=True, type=str, help="use options from given config file")
parser.add_argument("--model", required=True, type=str, help="the URDF model file")
parser.add_argument("--trajectory", type=str, help="trajectory .npz file (default: <model>.trajectory.npz)")
parser.add_argument("--filename", type=str, help="output measurements file (default: trajectory file)")
args = parser.parse_args()

with open(args.config) as stream:
    try:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)

config["urdf"] = args.model
config["jointNames"] = iDynTree.StringVector([])
if not iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"]):
    sys.exit()
config["num_dofs"] = len(config["jointNames"])


def load_trajectory_data(traj_file: str) -> dict[str, np.ndarray]:
    """Load trajectory kinematics saved by trajectory.py.

    Returns a dict with keys: times, positions, velocities, accelerations.
    """
    tf = np.load(traj_file, encoding="latin1", allow_pickle=True)
    required_keys = {"positions", "velocities", "accelerations", "times"}
    missing = required_keys - set(tf.files)
    if missing:
        print(f"Error: {traj_file} is missing sampled trajectory data: {', '.join(sorted(missing))}")
        print("Regenerate it with trajectory.py (which now saves sampled data).")
        sys.exit(1)

    return {
        "times": tf["times"],
        "positions": tf["positions"],
        "velocities": tf["velocities"],
        "accelerations": tf["accelerations"],
    }


def main() -> None:
    """Simulate realistic measurements from a trajectory file."""
    traj_file = args.trajectory or (config["urdf"] + ".trajectory.npz")
    output_file = args.filename or (config["urdf"] + ".measurements.npz")

    num_dofs = config["num_dofs"]
    freq = config["excitationFrequency"]
    floating_base = config.get("floatingBase", 0)
    seed = config.get("simulateRandomSeed", 42)
    rng = np.random.default_rng(seed)

    print(f"Model: {config['urdf']}, DOFs: {num_dofs}, floating-base: {floating_base}")

    # Load trajectory kinematics (sampled by trajectory.py, includes transitions/static postures/stops)
    print(f"Loading trajectory from {traj_file}")
    try:
        traj_data = load_trajectory_data(traj_file)
    except (FileNotFoundError, OSError):
        print(f"Trajectory file not found: {traj_file}")
        print("Generate one first with: uv run trajectory.py --config <config> --model <model>")
        sys.exit(1)

    times = traj_data["times"]
    positions = traj_data["positions"]
    velocities = traj_data["velocities"]
    accelerations = traj_data["accelerations"]
    num_samples = len(times)
    torque_col_offset = 6 if floating_base else 0

    # Use base motion from trajectory file if available, otherwise zeros.
    base_rpy = traj_data.get("base_rpy", np.zeros((num_samples, 3)))
    base_velocity = traj_data.get("base_velocity", np.zeros((num_samples, 6)))
    base_acceleration = traj_data.get("base_acceleration", np.zeros((num_samples, 6)))
    base_position: np.ndarray | None = None

    # For suspended floating base, compute the base motion from the ball joint dynamics
    if floating_base and config.get("floatingBaseAttachment") == "suspended":
        from excitation.suspendedDynamics import simulate_suspended_base_motion

        print("Simulating suspended base dynamics...")
        base_rpy, base_velocity, base_acceleration, base_position = simulate_suspended_base_motion(
            config["urdf"],
            positions,
            velocities,
            accelerations,
            times,
            attachment_frame=config.get("floatingBaseAttachmentFrame", "crane_ft"),
            damping=config.get("suspendedDamping", 2000.0),
        )

    # Compute torques via regressor (same code path as optimizer/simulateTrajectory)
    print(f"Computing inverse dynamics for {num_samples} samples...")
    sim_data: dict[str, Any] = {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "torques": np.zeros((num_samples, num_dofs + (6 if floating_base else 0))),
        "times": times,
        "measured_frequency": freq,
        "base_rpy": base_rpy,
        "base_velocity": base_velocity,
        "base_acceleration": base_acceleration,
        "contacts": np.array({}),
    }
    model = Model(config, config["urdf"])
    data = Data(config)
    old_skip, old_offset, old_sim = config["skipSamples"], config["startOffset"], config["simulateTorques"]
    config["skipSamples"] = 0
    config["startOffset"] = 0
    config["simulateTorques"] = True
    data.init_from_data(sim_data)
    model.computeRegressors(data)
    torques = data.samples["torques"].copy()
    config["skipSamples"], config["startOffset"], config["simulateTorques"] = old_skip, old_offset, old_sim

    # Build joint properties from URDF, override from config
    joint_names = list(config["jointNames"])
    jp = JointProperties.from_urdf(config["urdf"], joint_names)

    # sensor/actuator properties
    jp.control_rate = config.get("simulateControlRate", jp.control_rate)
    jp.torque_sensor_error = config.get("simulateTorqueSensorError", jp.torque_sensor_error)
    jp.torque_sensor_filter = config.get("simulateTorqueSensorFilter", jp.torque_sensor_filter)
    jp.position_filter = config.get("simulatePositionFilter", jp.position_filter)

    # scenario-dependent properties
    jp.thermal_warmup_time = config.get("simulateThermalWarmupTime", jp.thermal_warmup_time)
    jp.thermal_reduction = config.get("simulateThermalReduction", jp.thermal_reduction)
    jp.grav_comp_error_frac = config.get("simulateGravCompError", jp.grav_comp_error_frac)
    # recompute grav_comp_error array if fraction was overridden from config
    cum_mass = np.cumsum(jp.link_mass[::-1])[::-1]
    cum_max = cum_mass.max()
    jp.grav_comp_error = jp.grav_comp_error_frac * (cum_mass / cum_max if cum_max > 0 else np.ones_like(cum_mass))

    # friction model constants
    jp.stribeck_velocity = config.get("simulateStribeckVelocity", jp.stribeck_velocity)
    jp.friction_sign_threshold = config.get("simulateFrictionSignThreshold", jp.friction_sign_threshold)

    # cable stiffness scaling (0 = effectively disable cable forces)
    cable_scale = config.get("simulateCableStiffnessScale", jp.cable_stiffness_scale)
    jp.cable_stiffness = jp.cable_stiffness * cable_scale

    # Add effects (all configurable via config file)
    print("Adding simulated effects...")

    # always-on effects (part of physics)
    elastic = add_joint_elasticity(torques, accelerations, freq, jp, torque_col_offset)
    torques += elastic

    ripple = add_torque_ripple(num_samples, positions, jp, torque_col_offset)
    torques += ripple

    # configurable effects
    if config.get("simulateFriction", 1):
        friction = add_friction(torques, velocities, jp, torque_col_offset)
        torques += friction

    if config.get("simulateThermalDrift", 1):
        thermal = add_temperature_friction_drift(torques, velocities, times, jp, torque_col_offset)
        torques += thermal

    if config.get("simulateCableForces", 1):
        cable = add_cable_forces(torques, positions, jp, torque_col_offset, rng=rng)
        torques += cable

    if config.get("simulateGravityCompResidual", 1):
        grav_residual = add_gravity_compensation_residual(torques, positions, jp, torque_col_offset)
        torques += grav_residual

    if config.get("simulateTorqueQuantization", 1):
        torques = add_torque_quantization(torques, jp, torque_col_offset)

    if config.get("simulateStructuralDeflection", 1):
        positions = add_structural_deflection(positions, torques, jp, torque_col_offset)

    if config.get("simulateBacklash", 1):
        positions = add_backlash(positions, velocities, jp)

    if config.get("simulateEncoderQuantization", 1):
        positions = add_encoder_quantization(positions, jp)

    if config.get("simulateTimingJitter", 1):
        times = add_timing_jitter(times, freq, rng, jp=jp)

    # Add sensor noise
    (
        positions_noisy,
        velocities_noisy,
        torques_noisy,
        base_rpy_noisy,
        base_velocity_noisy,
        base_acceleration_noisy,
    ) = add_sensor_noise(
        positions,
        velocities,
        torques,
        freq,
        rng,
        jp=jp,
        base_rpy=base_rpy,
        base_velocity=base_velocity,
        base_acceleration=base_acceleration,
    )

    # Save in the format expected by identifier.py
    bv = np.zeros((num_samples, 6))
    ba = np.zeros((num_samples, 6))
    br = np.zeros((num_samples, 3))
    bp = np.zeros((num_samples, 3))
    if floating_base:
        if base_rpy_noisy is None or base_velocity_noisy is None or base_acceleration_noisy is None:
            raise RuntimeError("floating-base mode requires base sensor data")
        bv = base_velocity_noisy
        ba = base_acceleration_noisy
        br = base_rpy_noisy
        if base_position is not None:
            bp = base_position

    measurement_keys = {
        "positions",
        "positions_raw",
        "velocities",
        "velocities_raw",
        "accelerations",
        "torques",
        "torques_raw",
        "target_positions",
        "target_velocities",
        "target_accelerations",
        "times",
        "frequency",
        "contacts",
        "base_velocity",
        "base_acceleration",
        "base_rpy",
        "base_position",
    }

    # preserve any existing data in the output file (e.g. trajectory parameters)
    save_data: dict[str, Any] = {}
    if os.path.exists(output_file):
        existing = np.load(output_file, allow_pickle=True)
        colliding_keys = set(existing.files) & measurement_keys
        if colliding_keys:
            print(f"Warning: {output_file} already contains: {', '.join(sorted(colliding_keys))}")
            answer = input("Overwrite existing measurement data? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return
        for k in existing.files:
            save_data[k] = existing[k]

    # *_raw means "unprocessed measurement" (matching the excite.py/csv2npz.py real-data
    # semantics), so it carries the sensor noise. The clean reference signals are
    # available as target_*.
    save_data.update(
        positions=positions_noisy,
        positions_raw=positions_noisy,
        velocities=velocities_noisy,
        velocities_raw=velocities_noisy,
        accelerations=accelerations,
        torques=torques_noisy,
        torques_raw=torques_noisy,
        target_positions=positions,
        target_velocities=velocities,
        target_accelerations=accelerations,
        times=times,
        frequency=np.float64(freq),
        contacts=np.array({}),
        base_velocity=bv,
        base_acceleration=ba,
        base_rpy=br,
        base_position=bp,
    )
    np.savez(output_file, **save_data)

    # Print summary
    print(f"\nSaved {num_samples} samples to {output_file}")
    print(f"  Joint pos range: [{positions.min():.3f}, {positions.max():.3f}] rad")

    if floating_base:
        print("\n  Base wrench ranges:")
        wrench_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        for i in range(6):
            wrench = torques[:, i]
            print(f"    {wrench_labels[i]}: [{wrench.min():+.3f}, {wrench.max():+.3f}]")

    print("\n  Per-joint torque ranges:")
    for j in range(num_dofs):
        col = torque_col_offset + j
        tau_min = torques[:, col].min()
        tau_max = torques[:, col].max()
        tau_absmax = max(abs(tau_min), abs(tau_max))
        print(f"    joint {j}: [{tau_min:+.3f}, {tau_max:+.3f}] Nm (max |tau| = {tau_absmax:.3f})")

    print(f"\n  Overall torques range: [{torques.min():.3f}, {torques.max():.3f}] Nm")
    noise_power = np.sqrt(np.mean((torques_noisy - torques) ** 2))
    print(f"  Torque noise RMS: {noise_power:.4f} Nm")
    signal_power = np.sqrt(np.mean(torques**2))
    if noise_power > 0:
        print(f"  SNR: {signal_power / noise_power:.1f}")


if __name__ == "__main__":
    main()
