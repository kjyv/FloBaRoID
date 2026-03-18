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
import sys

import numpy as np
import yaml
from idyntree import bindings as iDynTree
from scipy.signal import butter, sosfiltfilt  # used by generate_static_posture_data

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
    add_sudden_stops,
    add_temperature_friction_drift,
    add_timing_jitter,
    add_torque_quantization,
    add_torque_ripple,
    rpy_to_angular_velocity,
)
from excitation.trajectoryGenerator import PulsedTrajectory

parser = argparse.ArgumentParser(description="Simulate realistic measurements from a trajectory file.")
parser.add_argument("--config", required=True, type=str, help="use options from given config file")
parser.add_argument("--model", required=True, type=str, help="the URDF model file")
parser.add_argument("--trajectory", type=str, help="trajectory .npz file (default: <model>.trajectory.npz)")
parser.add_argument("--filename", type=str, default="measurements.npz", help="output measurements file")
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


def load_trajectory(
    traj_file: str, num_dofs: int, freq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a PulsedTrajectory from .npz and compute position/velocity/acceleration arrays."""
    tf = np.load(traj_file, encoding="latin1", allow_pickle=True)
    use_deg = bool(tf["use_deg"])
    traj = PulsedTrajectory(num_dofs, use_deg=use_deg)
    traj.initWithParams(tf["a"], tf["b"], tf["q"], tf["nf"], tf["wf"])

    num_samples = int(traj.getPeriodLength() * freq)
    times = np.arange(num_samples) / freq
    positions = np.empty((num_samples, num_dofs))
    velocities = np.empty((num_samples, num_dofs))
    accelerations = np.empty((num_samples, num_dofs))

    for d in range(num_dofs):
        osc = traj.oscillators[d]
        l_arr = np.arange(1, osc.nf + 1)
        wlt = osc.w_f * np.outer(times, l_arr)
        sin_wlt = np.sin(wlt)
        cos_wlt = np.cos(wlt)

        a_coeff = np.array(osc.a) / (osc.w_f * l_arr)
        b_coeff = np.array(osc.b) / (osc.w_f * l_arr)
        positions[:, d] = sin_wlt @ a_coeff - cos_wlt @ b_coeff + osc.nf * osc.q0

        a_arr = np.array(osc.a)
        b_arr = np.array(osc.b)
        velocities[:, d] = cos_wlt @ a_arr + sin_wlt @ b_arr

        wl = osc.w_f * l_arr
        accelerations[:, d] = -sin_wlt @ (a_arr * wl) + cos_wlt @ (b_arr * wl)

    if use_deg:
        positions = np.deg2rad(positions)
        velocities = np.deg2rad(velocities)
        accelerations = np.deg2rad(accelerations)

    return times, positions, velocities, accelerations


def compute_inverse_dynamics_fixed_base(
    model: iDynTree.Model,
    kinDyn: iDynTree.KinDynComputations,
    num_dofs: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
) -> np.ndarray:
    """Compute joint torques via fixed-base inverse dynamics."""
    num_samples = positions.shape[0]
    torques = np.zeros((num_samples, num_dofs))

    gravity = iDynTree.Vector3()
    gravity.setVal(0, 0.0)
    gravity.setVal(1, 0.0)
    gravity.setVal(2, -9.81)

    s = iDynTree.JointPosDoubleArray(num_dofs)
    ds = iDynTree.JointDOFsDoubleArray(num_dofs)

    for t in range(num_samples):
        for j in range(num_dofs):
            s.setVal(j, positions[t, j])
            ds.setVal(j, velocities[t, j])
        kinDyn.setRobotState(s, ds, gravity)

        ddq = iDynTree.JointDOFsDoubleArray(num_dofs)
        for j in range(num_dofs):
            ddq.setVal(j, accelerations[t, j])

        ext_wrenches = iDynTree.LinkWrenches(model)
        gen_torques = iDynTree.FreeFloatingGeneralizedTorques(model)
        base_acc = iDynTree.Vector6()
        kinDyn.inverseDynamics(base_acc, ddq, ext_wrenches, gen_torques)
        torques[t, :] = gen_torques.jointTorques().toNumPy()

    return torques


def compute_inverse_dynamics_floating_base(
    model: iDynTree.Model,
    kinDyn: iDynTree.KinDynComputations,
    num_dofs: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    base_rpy: np.ndarray,
    base_velocity: np.ndarray,
    base_acceleration: np.ndarray,
) -> np.ndarray:
    """Compute torques (base wrench + joint torques) via floating-base inverse dynamics."""
    num_samples = positions.shape[0]
    torques = np.zeros((num_samples, num_dofs + 6))

    gravity = iDynTree.Vector3()
    gravity.setVal(0, 0.0)
    gravity.setVal(1, 0.0)
    gravity.setVal(2, -9.81)

    for t in range(num_samples):
        s = iDynTree.JointPosDoubleArray(num_dofs)
        ds = iDynTree.JointDOFsDoubleArray(num_dofs)
        ddq = iDynTree.JointDOFsDoubleArray(num_dofs)
        for j in range(num_dofs):
            s.setVal(j, positions[t, j])
            ds.setVal(j, velocities[t, j])
            ddq.setVal(j, accelerations[t, j])

        rot = iDynTree.Rotation.RPY(base_rpy[t, 0], base_rpy[t, 1], base_rpy[t, 2])
        pos_idt = iDynTree.Position.Zero()
        # Convention: RPY describes base_R_world, invert to get world_T_base
        # (must match identification/model.py line 249)
        world_T_base = iDynTree.Transform(rot, pos_idt).inverse()

        base_vel_twist = iDynTree.Twist()
        for i in range(6):
            base_vel_twist.setVal(i, base_velocity[t, i])

        kinDyn.setRobotState(world_T_base, s, base_vel_twist, ds, gravity)

        base_acc = iDynTree.Vector6()
        for i in range(6):
            base_acc.setVal(i, base_acceleration[t, i])

        ext_wrenches = iDynTree.LinkWrenches(model)
        gen_torques = iDynTree.FreeFloatingGeneralizedTorques(model)
        kinDyn.inverseDynamics(base_acc, ddq, ext_wrenches, gen_torques)

        torques[t, 0:6] = gen_torques.baseWrench().toNumPy()
        torques[t, 6:] = gen_torques.jointTorques().toNumPy()

    return torques


def generate_static_posture_data(
    model: iDynTree.Model,
    kinDyn: iDynTree.KinDynComputations,
    num_dofs: int,
    postures: list[list[float]],
    samples_per_posture: int,
    freq: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate static posture measurement data (gravity torques only).

    Returns (times, positions, velocities, torques_clean, torques_noisy).
    """
    total_samples = len(postures) * samples_per_posture

    positions = np.zeros((total_samples, num_dofs))
    velocities = np.zeros((total_samples, num_dofs))
    accelerations = np.zeros((total_samples, num_dofs))
    times = np.arange(total_samples) / freq

    for p_idx, posture in enumerate(postures):
        start = p_idx * samples_per_posture
        end = start + samples_per_posture
        for j in range(num_dofs):
            positions[start:end, j] = posture[j]

    torques = compute_inverse_dynamics_fixed_base(model, kinDyn, num_dofs, positions, velocities, accelerations)

    torque_noise = rng.normal(0, 0.02, torques.shape)
    torques_noisy = torques + torque_noise
    sos = butter(4, 40.0, btype="low", fs=freq, output="sos")
    for j in range(torques.shape[1]):
        torques_noisy[:, j] = sosfiltfilt(sos, torques_noisy[:, j])

    pos_noise = rng.normal(0, 1e-4, positions.shape)
    positions_noisy = positions + pos_noise

    return times, positions_noisy, velocities, torques, torques_noisy


def main() -> None:
    """Simulate realistic measurements from a trajectory file."""
    traj_file = args.trajectory or (config["urdf"] + ".trajectory.npz")
    output_file = args.filename

    num_dofs = config["num_dofs"]
    freq = config["excitationFrequency"]
    floating_base = config.get("floatingBase", 0)
    seed = config.get("simulateRandomSeed", 42)
    rng = np.random.default_rng(seed)

    # Load model
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(config["urdf"]):
        print(f"Failed to load {config['urdf']}")
        sys.exit(1)
    model = loader.model()

    kinDyn = iDynTree.KinDynComputations()
    if not kinDyn.loadRobotModel(model):
        print("Failed to load robot model into KinDynComputations")
        sys.exit(1)

    print(f"Model: {config['urdf']}, DOFs: {num_dofs}, floating-base: {floating_base}")

    # Load trajectory from file
    print(f"Loading trajectory from {traj_file}")
    try:
        times, positions, velocities, accelerations = load_trajectory(traj_file, num_dofs, freq)
    except (FileNotFoundError, OSError):
        print(f"Trajectory file not found: {traj_file}")
        print("Generate one first with: uv run trajectory.py --config <config> --model <model>")
        sys.exit(1)
    num_samples = len(times)
    dt = 1.0 / freq
    torque_col_offset = 6 if floating_base else 0

    # Generate base motion for floating-base
    base_rpy: np.ndarray | None = None
    base_velocity: np.ndarray | None = None
    base_acceleration: np.ndarray | None = None

    if floating_base:
        rpy_freqs = [0.15, 0.2, 0.1]
        rpy_amps = [0.3, 0.25, 0.2]
        base_rpy = np.zeros((num_samples, 3))
        base_rpy_dot = np.zeros((num_samples, 3))
        for i in range(3):
            w = 2.0 * np.pi * rpy_freqs[i]
            base_rpy[:, i] = rpy_amps[i] * np.sin(w * times)
            base_rpy_dot[:, i] = rpy_amps[i] * w * np.cos(w * times)

        base_velocity = np.zeros((num_samples, 6))
        base_acceleration = np.zeros((num_samples, 6))
        for t in range(num_samples):
            omega = rpy_to_angular_velocity(base_rpy[t], base_rpy_dot[t])
            base_velocity[t, 3:6] = omega
        for t in range(1, num_samples - 1):
            base_acceleration[t, 3:6] = (base_velocity[t + 1, 3:6] - base_velocity[t - 1, 3:6]) / (2 * dt)
        base_acceleration[0, 3:6] = (base_velocity[1, 3:6] - base_velocity[0, 3:6]) / dt
        base_acceleration[-1, 3:6] = (base_velocity[-1, 3:6] - base_velocity[-2, 3:6]) / dt

    # Insert sudden stops/restarts
    num_stops = config.get("simulateNumStops", 0)
    if num_stops > 0:
        print(f"Inserting {num_stops} sudden stop/restart segments...")
        positions, velocities, accelerations = add_sudden_stops(
            times,
            positions,
            velocities,
            accelerations,
            freq,
            num_stops=num_stops,
            rng=rng,
        )

    # Compute clean torques via inverse dynamics
    print(f"Computing inverse dynamics for {num_samples} samples...")
    if floating_base:
        if base_rpy is None or base_velocity is None or base_acceleration is None:
            raise RuntimeError("floating-base mode requires base_rpy, base_velocity, base_acceleration")
        torques = compute_inverse_dynamics_floating_base(
            model,
            kinDyn,
            num_dofs,
            positions,
            velocities,
            accelerations,
            base_rpy,
            base_velocity,
            base_acceleration,
        )
    else:
        torques = compute_inverse_dynamics_fixed_base(
            model,
            kinDyn,
            num_dofs,
            positions,
            velocities,
            accelerations,
        )

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

    # Add realistic effects (all configurable via config file)
    print("Adding realistic effects...")

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

    # Optionally append static posture data
    if config.get("simulateStaticPostures", 0) and not floating_base:
        postures_config = config.get("staticPostures", [])
        # filter postures to match DOF count
        valid_postures = [p[:num_dofs] for p in postures_config if len(p) >= num_dofs]
        if valid_postures:
            samples_per = config.get("simulateStaticSamplesPerPosture", 100)
            print(f"Generating {len(valid_postures)} static postures ({samples_per} samples each)...")
            s_times, s_pos, s_vel, s_torques, s_torques_noisy = generate_static_posture_data(
                model,
                kinDyn,
                num_dofs,
                valid_postures,
                samples_per,
                freq,
                rng,
            )
            s_times += times[-1] + 1.0 / freq

            times = np.concatenate([times, s_times])
            positions = np.concatenate([positions, s_pos])
            positions_noisy = np.concatenate([positions_noisy, s_pos])
            velocities = np.concatenate([velocities, s_vel])
            velocities_noisy = np.concatenate([velocities_noisy, s_vel])
            accelerations = np.concatenate([accelerations, np.zeros_like(s_vel)])
            torques = np.concatenate([torques, s_torques])
            torques_noisy = np.concatenate([torques_noisy, s_torques_noisy])
            num_samples = len(times)

            print("  Static torque ranges per joint:")
            for j in range(num_dofs):
                tau_absmax = np.nanmax(np.abs(s_torques[:, torque_col_offset + j]))
                print(f"    joint {j}: max |tau| = {tau_absmax:.3f} Nm")

    # Save in the format expected by identifier.py
    bv = np.zeros((num_samples, 6))
    ba = np.zeros((num_samples, 6))
    br = np.zeros((num_samples, 3))
    if floating_base:
        if base_rpy_noisy is None or base_velocity_noisy is None or base_acceleration_noisy is None:
            raise RuntimeError("floating-base mode requires base sensor data")
        bv = base_velocity_noisy
        ba = base_acceleration_noisy
        br = base_rpy_noisy

    np.savez(
        output_file,
        positions=positions_noisy,
        positions_raw=positions,
        velocities=velocities_noisy,
        velocities_raw=velocities,
        accelerations=accelerations,
        torques=torques_noisy,
        torques_raw=torques,
        target_positions=positions,
        target_velocities=velocities,
        target_accelerations=accelerations,
        times=times,
        frequency=np.float64(freq),
        contacts=np.array({}),
        base_velocity=bv,
        base_acceleration=ba,
        base_rpy=br,
    )

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
