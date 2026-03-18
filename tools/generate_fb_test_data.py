"""Generate realistic floating-base test data for the 3-link model.

Creates simulated measurement data with:
- Smooth sinusoidal joint trajectories with multiple harmonics
- Smooth sinusoidal base orientation (RPY)
- Base angular velocity derived analytically from RPY
- Base angular acceleration derived analytically
- Torques and base wrench computed via iDynTree inverse dynamics
- Realistic effects: sensor noise, joint elasticity vibrations, torque ripple
"""

import sys

sys.path.insert(0, ".")

import idyntree.bindings as iDynTree
import numpy as np
from scipy.signal import butter, sosfiltfilt


def rpy_to_angular_velocity(rpy: np.ndarray, rpy_dot: np.ndarray) -> np.ndarray:
    """Convert RPY rates to angular velocity in world frame.

    The relationship is: omega = E(rpy) * rpy_dot
    where E is the RPY rate to angular velocity mapping matrix.
    """
    roll, pitch, _yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    _cp, sp = np.cos(pitch), np.sin(pitch)

    # Mapping matrix E such that omega = E * [roll_dot, pitch_dot, yaw_dot]^T
    # (world-frame angular velocity from RPY Euler angle rates)
    E = np.array(
        [
            [1.0, 0.0, -sp],
            [0.0, cr, sr * _cp],
            [0.0, -sr, cr * _cp],
        ]
    )
    return E @ rpy_dot


def add_joint_elasticity(torques: np.ndarray, accelerations: np.ndarray, freq: float, num_dofs: int) -> np.ndarray:
    """Add damped oscillations to joint torques triggered by high jerk.

    Simulates elastic vibrations from joint flexibility — when acceleration
    changes rapidly (high jerk), the joint compliance causes ringing that
    decays over time.
    """
    dt = 1.0 / freq
    num_samples = torques.shape[0]

    # Compute jerk (rate of change of acceleration)
    jerk = np.diff(accelerations, axis=0) / dt
    jerk = np.vstack([jerk, jerk[-1:]])  # pad to same length

    # Per-joint elasticity parameters
    elastic_freqs = [25.0, 30.0][:num_dofs]  # natural frequency Hz (typical for light robot joints)
    damping_ratios = [0.08, 0.06][:num_dofs]  # underdamped
    elastic_gains = [0.003, 0.002][:num_dofs]  # how much jerk excites the vibration

    elastic_torques = np.zeros_like(torques)
    for j in range(num_dofs):
        wn = 2.0 * np.pi * elastic_freqs[j]
        zeta = damping_ratios[j]
        wd = wn * np.sqrt(1.0 - zeta**2)  # damped frequency

        # Convolve jerk with damped oscillation impulse response
        # h(t) = exp(-zeta*wn*t) * sin(wd*t) for t >= 0
        # Use a truncated impulse response (5 time constants)
        t_decay = 5.0 / (zeta * wn)
        n_impulse = min(int(t_decay * freq), num_samples)
        t_impulse = np.arange(n_impulse) * dt
        impulse = np.exp(-zeta * wn * t_impulse) * np.sin(wd * t_impulse)

        vibration = np.convolve(jerk[:, j], impulse, mode="full")[:num_samples]
        # Add to joint torque columns (offset by 6 for base wrench)
        elastic_torques[:, 6 + j] = elastic_gains[j] * vibration

    return elastic_torques


def add_torque_ripple(num_samples: int, num_dofs: int, positions: np.ndarray) -> np.ndarray:
    """Add position-dependent torque ripple (cogging torque).

    Real motors have small periodic torque disturbances that depend on
    rotor position, typically at multiples of the electrical frequency.
    """
    ripple = np.zeros((num_samples, num_dofs + 6))
    # Typical cogging has harmonics at 6x and 12x electrical frequency
    # Amplitude is small (fraction of a Nm for light robots)
    cogging_amps = [0.05, 0.03][:num_dofs]
    pole_pairs = [4, 4][:num_dofs]  # motor pole pairs

    for j in range(num_dofs):
        electrical_angle = positions[:, j] * pole_pairs[j]
        ripple[:, 6 + j] = cogging_amps[j] * (np.sin(6 * electrical_angle) + 0.3 * np.sin(12 * electrical_angle))
    return ripple


def add_sensor_noise(
    positions: np.ndarray,
    velocities: np.ndarray,
    torques: np.ndarray,
    base_rpy: np.ndarray,
    base_velocity: np.ndarray,
    base_acceleration: np.ndarray,
    freq: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add realistic sensor noise to all measurement channels.

    Position encoders: very low noise (high resolution)
    Velocity: moderate noise (often derived from position)
    Torque sensors: moderate noise
    IMU (base rpy, velocity, acceleration): moderate noise
    """
    # Position encoder noise (rad) — high resolution encoders
    pos_noise = rng.normal(0, 1e-4, positions.shape)
    positions_noisy = positions + pos_noise

    # Velocity noise — typically noisier than position (numerical differentiation or tachometer)
    vel_noise = rng.normal(0, 5e-3, velocities.shape)
    velocities_noisy = velocities + vel_noise

    # Torque sensor noise (Nm)
    torque_noise = rng.normal(0, 0.02, torques.shape)
    torques_noisy = torques + torque_noise

    # IMU noise on base RPY (rad)
    rpy_noise = rng.normal(0, 5e-4, base_rpy.shape)
    base_rpy_noisy = base_rpy + rpy_noise

    # IMU angular velocity noise (rad/s)
    base_vel_noise = rng.normal(0, 1e-3, base_velocity.shape)
    base_velocity_noisy = base_velocity + base_vel_noise

    # IMU acceleration noise (rad/s^2)
    base_acc_noise = rng.normal(0, 5e-3, base_acceleration.shape)
    base_acceleration_noisy = base_acceleration + base_acc_noise

    # Low-pass filter noisy signals (simulating on-board filtering, as a real system would do)
    sos = butter(4, 40.0, btype="low", fs=freq, output="sos")
    for j in range(positions.shape[1]):
        positions_noisy[:, j] = sosfiltfilt(sos, positions_noisy[:, j])
        velocities_noisy[:, j] = sosfiltfilt(sos, velocities_noisy[:, j])
    for j in range(torques.shape[1]):
        torques_noisy[:, j] = sosfiltfilt(sos, torques_noisy[:, j])
    for j in range(3):
        base_rpy_noisy[:, j] = sosfiltfilt(sos, base_rpy_noisy[:, j])
    for j in range(6):
        base_velocity_noisy[:, j] = sosfiltfilt(sos, base_velocity_noisy[:, j])
        base_acceleration_noisy[:, j] = sosfiltfilt(sos, base_acceleration_noisy[:, j])

    return (
        positions_noisy,
        velocities_noisy,
        torques_noisy,
        base_rpy_noisy,
        base_velocity_noisy,
        base_acceleration_noisy,
    )


def main() -> None:
    """Generate floating-base test data for identification."""
    rng = np.random.default_rng(42)

    # Load model
    urdf_file = "model/threeLinks.urdf"
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_file):
        print(f"Failed to load {urdf_file}")
        sys.exit(1)
    model = loader.model()

    kinDyn = iDynTree.KinDynComputations()
    if not kinDyn.loadRobotModel(model):
        print("Failed to load robot model into KinDynComputations")
        sys.exit(1)

    num_dofs = kinDyn.getNrOfDegreesOfFreedom()
    print(f"Model: {urdf_file}, DOFs: {num_dofs}")

    # Simulation parameters
    freq = 200.0  # Hz
    duration = 10.0  # seconds
    num_samples = int(duration * freq)
    dt = 1.0 / freq
    times = np.arange(num_samples) * dt

    gravity = iDynTree.Vector3()
    gravity.setVal(0, 0.0)
    gravity.setVal(1, 0.0)
    gravity.setVal(2, -9.81)

    # Generate smooth joint trajectories (multiple harmonics for better excitation)
    positions = np.zeros((num_samples, num_dofs))
    velocities = np.zeros((num_samples, num_dofs))
    accelerations = np.zeros((num_samples, num_dofs))

    # Use fourier series with multiple harmonics (like the real trajectory optimizer)
    joint_offsets = [0.0, 0.5]
    joint_harmonics: list[list[tuple[float, float]]] = [
        [(0.8, 0.3), (0.3, 0.7), (0.15, 1.1)],  # joint 0
        [(1.2, 0.5), (0.4, 1.0), (0.2, 1.5)],  # joint 1
    ]
    for j in range(num_dofs):
        for amp, f in joint_harmonics[j]:
            w = 2.0 * np.pi * f
            # Mix sine and cosine for asymmetric trajectories
            phase = rng.uniform(0, 2.0 * np.pi)
            positions[:, j] += amp * np.sin(w * times + phase)
            velocities[:, j] += amp * w * np.cos(w * times + phase)
            accelerations[:, j] += -amp * w**2 * np.sin(w * times + phase)
        positions[:, j] += joint_offsets[j]

    # Generate smooth base orientation trajectory (sinusoidal RPY)
    # Keep angles moderate to avoid gimbal lock
    rpy_freqs = [0.15, 0.2, 0.1]  # Hz for roll, pitch, yaw
    rpy_amps = [0.3, 0.25, 0.2]  # rad amplitude

    base_rpy = np.zeros((num_samples, 3))
    base_rpy_dot = np.zeros((num_samples, 3))

    for i in range(3):
        w = 2.0 * np.pi * rpy_freqs[i]
        base_rpy[:, i] = rpy_amps[i] * np.sin(w * times)
        base_rpy_dot[:, i] = rpy_amps[i] * w * np.cos(w * times)

    # Convert RPY rates to angular velocity and acceleration
    base_velocity = np.zeros((num_samples, 6))  # [lin_vel(3), ang_vel(3)]
    base_acceleration = np.zeros((num_samples, 6))  # [lin_acc(3), ang_acc(3)]

    for t in range(num_samples):
        omega = rpy_to_angular_velocity(base_rpy[t], base_rpy_dot[t])
        base_velocity[t, 3:6] = omega

    # Numerical differentiation for angular acceleration (central differences)
    for t in range(1, num_samples - 1):
        base_acceleration[t, 3:6] = (base_velocity[t + 1, 3:6] - base_velocity[t - 1, 3:6]) / (2 * dt)
    # Forward/backward for boundaries
    base_acceleration[0, 3:6] = (base_velocity[1, 3:6] - base_velocity[0, 3:6]) / dt
    base_acceleration[-1, 3:6] = (base_velocity[-1, 3:6] - base_velocity[-2, 3:6]) / dt

    # Compute clean torques via inverse dynamics
    torques = np.zeros((num_samples, num_dofs + 6))  # 6 base wrench + num_dofs joint torques

    print(f"Computing inverse dynamics for {num_samples} samples...")
    for t in range(num_samples):
        # Set joint state
        s = iDynTree.JointPosDoubleArray(num_dofs)
        ds = iDynTree.JointDOFsDoubleArray(num_dofs)
        ddq = iDynTree.JointDOFsDoubleArray(num_dofs)
        for j in range(num_dofs):
            s.setVal(j, positions[t, j])
            ds.setVal(j, velocities[t, j])
            ddq.setVal(j, accelerations[t, j])

        # Set base state
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

        # Inverse dynamics
        ext_wrenches = iDynTree.LinkWrenches(model)
        gen_torques = iDynTree.FreeFloatingGeneralizedTorques(model)
        kinDyn.inverseDynamics(base_acc, ddq, ext_wrenches, gen_torques)

        torques[t, 0:6] = gen_torques.baseWrench().toNumPy()
        torques[t, 6:] = gen_torques.jointTorques().toNumPy()

    # Add realistic effects to torque measurements
    print("Adding realistic effects...")

    # Joint elasticity vibrations (triggered by high jerk)
    elastic = add_joint_elasticity(torques, accelerations, freq, num_dofs)
    torques += elastic

    # Cogging torque ripple
    ripple = add_torque_ripple(num_samples, num_dofs, positions)
    torques += ripple

    # Add sensor noise and filter (positions, velocities, torques, IMU)
    (
        positions_noisy,
        velocities_noisy,
        torques_noisy,
        base_rpy_noisy,
        base_velocity_noisy,
        base_acceleration_noisy,
    ) = add_sensor_noise(positions, velocities, torques, base_rpy, base_velocity, base_acceleration, freq, rng)

    # Save (no contacts — clean test of floating-base inverse dynamics)
    output_file = "data/THREELINK/SIM/measurements_opt1_fb.npz"
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
        base_velocity=base_velocity_noisy,
        base_acceleration=base_acceleration_noisy,
        base_rpy=base_rpy_noisy,
        contacts=np.array({}),
        times=times,
        frequency=np.float64(freq),
    )

    print(f"Saved {num_samples} samples to {output_file}")
    print(f"  Joint pos range: [{positions.min():.3f}, {positions.max():.3f}] rad")
    print(f"  Base RPY range: [{base_rpy.min():.3f}, {base_rpy.max():.3f}] rad")
    print(f"  Base ang vel range: [{base_velocity[:, 3:].min():.3f}, {base_velocity[:, 3:].max():.3f}] rad/s")
    print(f"  Torques range: [{torques.min():.3f}, {torques.max():.3f}] Nm")
    noise_power = np.sqrt(np.mean((torques_noisy - torques) ** 2))
    print(f"  Torque noise RMS: {noise_power:.4f} Nm")
    print(f"  SNR: {np.sqrt(np.mean(torques**2)) / noise_power:.1f}")


if __name__ == "__main__":
    main()
