"""Realistic measurement effects for robot dynamics simulation.

Contains functions that add real-world effects (friction, noise, backlash, etc.)
to ideal inverse dynamics torques and positions, producing simulated measurements
that closely resemble what a real robot would produce.

Used by simulator.py to transform optimized trajectories into realistic
measurement data for identification benchmarking.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class JointProperties:
    """Per-joint physical properties used by simulation effects.

    Derived from the URDF model and optionally overridden by config.
    All arrays are indexed by joint index [0..num_dofs).
    """

    num_dofs: int

    # from URDF <dynamics> tag
    viscous_friction: np.ndarray  # Fv (Nm·s/rad) — URDF "damping"
    coulomb_friction: np.ndarray  # Fc (Nm) — URDF "friction"

    # from URDF <limit> tag
    torque_limit: np.ndarray  # max effort (Nm)
    velocity_limit: np.ndarray  # max velocity (rad/s)

    # from URDF <inertial> — mass of the link driven by each joint
    link_mass: np.ndarray  # (kg)

    # scalar sensor/actuator properties (from config, with generic defaults)
    control_rate: float = 1000.0  # joint-level control loop rate (Hz)
    torque_sensor_error: float = 0.01  # torque sensor noise as fraction of torque limit
    torque_sensor_filter: float = 200.0  # torque sensor on-board low-pass cutoff (Hz)
    position_filter: float = 40.0  # position/velocity data low-pass cutoff (Hz)

    # thermal drift parameters (scenario-dependent — real warmup takes 10-30 minutes)
    thermal_warmup_time: float = 0.0  # time in seconds robot was running before trajectory (0 = cold start)
    thermal_reduction: float = 0.12  # max friction reduction when fully warm (fraction)

    # gravity compensation residual (depends on whether controller has grav comp enabled)
    grav_comp_error_frac: float = 0.08  # error as fraction of gravity torque (0 = perfect compensation)

    # friction model constants
    stribeck_velocity: float = 0.05  # Stribeck velocity (rad/s) — speed at which stiction decays
    friction_sign_threshold: float = 0.02  # smoothing threshold for sign(vel) (rad/s)

    # cable routing (highly robot-specific — varies even between same model)
    cable_stiffness_scale: float = 1.0  # multiplier for derived cable stiffness (0 = disable)

    # derived / config-overridable per-joint arrays (set by from_urdf with sensible defaults)
    stiction: np.ndarray = field(default_factory=lambda: np.array([]))  # Stribeck breakaway excess (Nm)
    backlash: np.ndarray = field(default_factory=lambda: np.array([]))  # gear backlash half-width (rad)
    encoder_bits: np.ndarray = field(default_factory=lambda: np.array([]))  # encoder resolution (bits)
    compliance: np.ndarray = field(default_factory=lambda: np.array([]))  # structural compliance (rad/Nm)
    cable_stiffness: np.ndarray = field(default_factory=lambda: np.array([]))  # cable routing stiffness (Nm/rad)
    elasticity_freq: np.ndarray = field(default_factory=lambda: np.array([]))  # joint elasticity natural freq (Hz)
    elasticity_damping: np.ndarray = field(default_factory=lambda: np.array([]))  # elasticity damping ratio
    elasticity_gain: np.ndarray = field(default_factory=lambda: np.array([]))  # jerk-to-vibration gain
    cogging_amplitude: np.ndarray = field(default_factory=lambda: np.array([]))  # cogging torque amplitude (Nm)
    torque_quant_bits: np.ndarray = field(default_factory=lambda: np.array([]))  # PWM resolution (bits)
    thermal_tau: np.ndarray = field(default_factory=lambda: np.array([]))  # thermal time constant (s)
    grav_comp_error: np.ndarray = field(default_factory=lambda: np.array([]))  # per-joint gravity comp error

    @staticmethod
    def from_urdf(urdf_file: str, joint_names: list[str]) -> "JointProperties":
        """Build JointProperties from a URDF file.

        Reads friction, limits, and masses from the URDF and derives reasonable
        defaults for all other parameters proportional to the joint's size/capacity.
        """
        import xml.etree.ElementTree as ET

        from identification.helpers import URDFHelpers

        num_dofs = len(joint_names)
        limits = URDFHelpers.getJointLimits(urdf_file, use_deg=False)
        friction = URDFHelpers.getJointFriction(urdf_file)

        # read URDF link masses (via iDynTree)
        from idyntree import bindings as iDynTree

        loader = iDynTree.ModelLoader()
        loader.loadModelFromFile(urdf_file)
        idyn_model = loader.model()
        # map joint name → child link mass
        link_masses = np.zeros(num_dofs)
        for i, jname in enumerate(joint_names):
            joint_idx = idyn_model.getJointIndex(jname)
            if joint_idx >= 0:
                joint = idyn_model.getJoint(joint_idx)
                child_link_idx = joint.getSecondAttachedLink()
                link = idyn_model.getLink(child_link_idx)
                link_masses[i] = link.getInertia().getMass()

        # parse <transmission> elements for gear ratio and rotor inertia
        tree = ET.parse(urdf_file)
        gear_ratios = np.ones(num_dofs)
        rotor_inertias = np.zeros(num_dofs)
        for trans in tree.findall("transmission"):
            joint_el = trans.find("joint")
            if joint_el is None:
                continue
            jname = joint_el.attrib.get("name", "")
            if jname not in joint_names:
                continue
            j_idx = joint_names.index(jname)
            actuator_el = trans.find("actuator")
            if actuator_el is not None:
                mr_el = actuator_el.find("mechanicalReduction")
                if mr_el is not None and mr_el.text:
                    gear_ratios[j_idx] = float(mr_el.text)
                ri_el = actuator_el.find("rotor_inertia")
                if ri_el is not None and ri_el.text:
                    rotor_inertias[j_idx] = float(ri_el.text)

        # extract URDF values into arrays (with safe defaults for missing values)
        fv = np.array([friction.get(j, {}).get("f_velocity", 1.0) for j in joint_names])
        fc = np.array([friction.get(j, {}).get("f_constant", 0.0) for j in joint_names])
        tau_max = np.array([limits.get(j, {}).get("torque", 50.0) for j in joint_names])
        vel_max = np.array([limits.get(j, {}).get("velocity", 3.0) for j in joint_names])

        props = JointProperties(
            num_dofs=num_dofs,
            viscous_friction=fv,
            coulomb_friction=fc,
            torque_limit=tau_max,
            velocity_limit=vel_max,
            link_mass=link_masses,
        )

        # --- derive remaining parameters from URDF values ---
        # helper to safely normalize arrays (avoid division by zero if all values are 0)
        def _norm(arr: np.ndarray) -> np.ndarray:
            m = arr.max()
            return arr / m if m > 0 else np.ones_like(arr)

        # stiction: proportional to Coulomb friction (or ~0.3% of torque limit if Fc=0)
        props.stiction = np.where(fc > 0, fc * 0.6, tau_max * 0.003)

        # backlash: scales with gear ratio (higher ratio = slightly more clearance).
        # ~0.5 arcmin base + proportional to gear ratio
        arcmin_to_rad = np.pi / (180.0 * 60.0)
        props.backlash = (0.5 + 0.01 * gear_ratios) * arcmin_to_rad

        # encoder bits: motor-side encoder resolution scaled by gear ratio gives
        # effective output-side resolution. Typical motor encoders are 13-17 bit.
        # Higher gear ratio = better effective resolution at output.
        base_motor_bits = 13.0 + 3.0 * _norm(tau_max)  # bigger motor = better encoder
        props.encoder_bits = base_motor_bits + np.log2(np.clip(gear_ratios, 1, None))

        # structural compliance: inversely proportional to torque capacity
        # (stiffer joints for higher torques)
        tau_min = tau_max.min() if tau_max.min() > 0 else 1.0
        props.compliance = 1e-4 / (tau_max / tau_min)

        # cable stiffness: proportional to cumulative mass outboard of each joint
        # (proximal joints carry more cables from all distal links)
        cum_mass = np.cumsum(link_masses[::-1])[::-1]
        props.cable_stiffness = 0.02 + 0.15 * _norm(cum_mass)

        # elasticity: natural frequency from reflected rotor inertia and link inertia
        # higher gear ratio = stiffer coupling but also more reflected inertia
        reflected_inertia = rotor_inertias * gear_ratios**2
        total_inertia = link_masses * 0.01 + reflected_inertia  # rough link inertia + reflected motor
        # higher total inertia → lower natural frequency
        props.elasticity_freq = 20.0 + 15.0 * (1.0 - total_inertia / (total_inertia.max() + 1e-10))
        props.elasticity_damping = np.full(num_dofs, 0.07)
        # gain: higher compliance = more vibration per jerk
        props.elasticity_gain = 0.001 + 0.002 * _norm(props.compliance)

        # cogging: proportional to torque limit / gear ratio (cogging is motor-side,
        # appears reduced at output by gear ratio)
        props.cogging_amplitude = tau_max / (gear_ratios + 1.0) * 0.005

        # torque quantization: current driver resolution over motor torque range
        # motor torque range = output torque limit / gear ratio
        motor_tau = tau_max / np.clip(gear_ratios, 1, None)
        props.torque_quant_bits = np.clip(11 + 3 * _norm(motor_tau), 11, 16).astype(float)

        # thermal: bigger motors have more thermal mass → longer time constant
        # real gearbox/motor warmup takes 10-30 minutes, not seconds
        # rotor inertia is a proxy for motor size; fall back to link mass if no rotor data
        if rotor_inertias.max() > 0:
            motor_size = _norm(rotor_inertias)
        else:
            motor_size = _norm(link_masses)
        props.thermal_tau = 300.0 + 900.0 * motor_size  # 5-20 minutes

        # gravity compensation error: proportional to cumulative mass beyond this joint
        # (heavier arm = more gravity torque to compensate, thus larger absolute error)
        props.grav_comp_error = props.grav_comp_error_frac * _norm(cum_mass)

        return props


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


def add_joint_elasticity(
    torques: np.ndarray,
    accelerations: np.ndarray,
    freq: float,
    jp: JointProperties,
    torque_col_offset: int = 6,
) -> np.ndarray:
    """Add damped oscillations to joint torques triggered by high jerk.

    Simulates elastic vibrations from joint flexibility — when acceleration
    changes rapidly (high jerk), the joint compliance causes ringing that
    decays over time. The vibration is modeled by convolving jerk with a
    damped oscillation impulse response: h(t) = exp(-zeta*wn*t) * sin(wd*t).

    Natural frequency, damping ratio, and gain are per-joint from JointProperties
    (derived from URDF link masses — lighter links have higher natural frequencies).
    """
    dt = 1.0 / freq
    num_samples = torques.shape[0]

    jerk = np.diff(accelerations, axis=0) / dt
    jerk = np.vstack([jerk, jerk[-1:]])

    elastic_torques = np.zeros_like(torques)
    for j in range(jp.num_dofs):
        wn = 2.0 * np.pi * jp.elasticity_freq[j]
        zeta = jp.elasticity_damping[j]
        gain = jp.elasticity_gain[j]
        wd = wn * np.sqrt(1.0 - zeta**2)

        t_decay = 5.0 / (zeta * wn)
        n_impulse = min(int(t_decay * freq), num_samples)
        t_impulse = np.arange(n_impulse) * dt
        impulse = np.exp(-zeta * wn * t_impulse) * np.sin(wd * t_impulse)

        vibration = np.convolve(jerk[:, j], impulse, mode="full")[:num_samples]
        elastic_torques[:, torque_col_offset + j] = gain * vibration

    return elastic_torques


def add_torque_ripple(
    num_samples: int,
    positions: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 6,
) -> np.ndarray:
    """Add position-dependent torque ripple (cogging torque).

    Real motors have small periodic torque disturbances that depend on
    rotor position, caused by interaction between permanent magnets and
    stator teeth. The ripple occurs at multiples of the electrical
    frequency (6x and 12x are dominant harmonics for typical 3-phase motors).

    Amplitude (cogging_amplitude in JointProperties) is derived from the
    URDF torque limit and gear ratio — cogging is a motor-side effect,
    reduced at the output by the gear ratio.

    Limitations: the actual cogging profile depends on motor geometry and
    can vary significantly between motor types. The sinusoidal approximation
    captures the dominant effect but not the exact waveform shape. The number
    of pole pairs (hardcoded at 4) affects the ripple frequency.
    """
    num_torque_cols = torque_col_offset + jp.num_dofs
    ripple = np.zeros((num_samples, num_torque_cols))

    for j in range(jp.num_dofs):
        amp = jp.cogging_amplitude[j]
        # 4 pole pairs typical for servo motors
        electrical_angle = positions[:, j] * 4
        ripple[:, torque_col_offset + j] = amp * (np.sin(6 * electrical_angle) + 0.3 * np.sin(12 * electrical_angle))
    return ripple


def add_sensor_noise(
    positions: np.ndarray,
    velocities: np.ndarray,
    torques: np.ndarray,
    freq: float,
    rng: np.random.Generator,
    jp: JointProperties | None = None,
    base_rpy: np.ndarray | None = None,
    base_velocity: np.ndarray | None = None,
    base_acceleration: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Add realistic sensor noise to all measurement channels.

    Position encoders: noise derived from encoder resolution in JointProperties
    Velocity: typically noisier (derived from position differentiation at control rate)
    Torque sensors: per-joint noise scaled as fraction of torque limit (typical: 0.5%)
    IMU (base rpy, velocity, acceleration): moderate noise (only if provided)

    Torque signals are low-pass filtered at a high cutoff (typical on-board torque
    sensor filtering). Position/velocity signals are filtered at a lower cutoff
    matching typical data transmission filtering.
    """
    num_dofs = positions.shape[1]
    torque_col_offset = torques.shape[1] - num_dofs

    # position noise: one encoder count std (from effective encoder resolution)
    positions_noisy = positions.copy()
    for j in range(num_dofs):
        if jp is not None:
            encoder_resolution = 2.0 * np.pi / (2 ** jp.encoder_bits[j])
        else:
            encoder_resolution = 1e-4  # fallback: ~17-bit encoder
        pos_noise = rng.normal(0, encoder_resolution, positions.shape[0])
        positions_noisy[:, j] += pos_noise

    # velocity noise: derived from position differentiation at the control loop rate
    vel_noise_std = 5e-3  # fallback
    if jp is not None:
        # noise from differentiating encoder at control rate (~10% of theoretical worst case)
        encoder_res_avg = 2.0 * np.pi / (2 ** np.mean(jp.encoder_bits))
        vel_noise_std = encoder_res_avg * jp.control_rate * 0.1
    vel_noise = rng.normal(0, vel_noise_std, velocities.shape)
    velocities_noisy = velocities + vel_noise

    # torque noise: fraction of torque limit per joint (from torque_sensor_error)
    torques_noisy = torques.copy()
    torque_error_frac = jp.torque_sensor_error if jp is not None else 0.01
    for j in range(num_dofs):
        if jp is not None:
            torque_noise_std = jp.torque_limit[j] * torque_error_frac
        else:
            torque_noise_std = 0.1  # fallback
        col = torque_col_offset + j
        torques_noisy[:, col] += rng.normal(0, torque_noise_std, torques.shape[0])

    # on-board filtering: torque sensors have a high-frequency low-pass,
    # position/velocity signals are filtered at a lower cutoff for data transmission
    nyquist = freq / 2.0
    torque_filter_hz = jp.torque_sensor_filter if jp is not None else 200.0
    if torque_filter_hz < nyquist:
        sos_torque = butter(4, torque_filter_hz, btype="low", fs=freq, output="sos")
        for j in range(torques.shape[1]):
            torques_noisy[:, j] = sosfiltfilt(sos_torque, torques_noisy[:, j])

    pos_vel_cutoff = min(jp.position_filter if jp is not None else 40.0, nyquist * 0.8)
    sos_pos = butter(4, pos_vel_cutoff, btype="low", fs=freq, output="sos")
    for j in range(num_dofs):
        positions_noisy[:, j] = sosfiltfilt(sos_pos, positions_noisy[:, j])
        velocities_noisy[:, j] = sosfiltfilt(sos_pos, velocities_noisy[:, j])

    # IMU noise (floating-base only)
    base_rpy_noisy: np.ndarray | None = None
    base_velocity_noisy: np.ndarray | None = None
    base_acceleration_noisy: np.ndarray | None = None
    if base_rpy is not None:
        rpy_noise = rng.normal(0, 5e-4, base_rpy.shape)
        base_rpy_noisy = base_rpy + rpy_noise
        for j in range(3):
            base_rpy_noisy[:, j] = sosfiltfilt(sos_pos, base_rpy_noisy[:, j])
    if base_velocity is not None:
        base_vel_noise = rng.normal(0, 1e-3, base_velocity.shape)
        base_velocity_noisy = base_velocity + base_vel_noise
        for j in range(6):
            base_velocity_noisy[:, j] = sosfiltfilt(sos_pos, base_velocity_noisy[:, j])
    if base_acceleration is not None:
        base_acc_noise = rng.normal(0, 5e-3, base_acceleration.shape)
        base_acceleration_noisy = base_acceleration + base_acc_noise
        for j in range(6):
            base_acceleration_noisy[:, j] = sosfiltfilt(sos_pos, base_acceleration_noisy[:, j])

    return (
        positions_noisy,
        velocities_noisy,
        torques_noisy,
        base_rpy_noisy,
        base_velocity_noisy,
        base_acceleration_noisy,
    )


def add_sudden_stops(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    freq: float,
    num_stops: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert sudden deceleration/stop/restart segments into the trajectory.

    Simulates realistic robot behavior where the controller commands a rapid
    stop (e.g., due to a waypoint, safety check, or mode switch) followed
    by a fast restart. Uses a cosine velocity profile for smooth but rapid
    deceleration (0.3s), a brief hold at zero (0.2s), and acceleration back
    to the original velocity (0.3s).

    The high jerk during transitions excites joint elasticity, backlash,
    and stiction effects that smooth Fourier trajectories cannot produce.
    Positions are reintegrated after modifying velocity to keep them consistent.

    Limitations: the stop/restart profile is idealized — real emergency stops
    can be much harder, and the restart velocity may not match the original.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    num_samples = len(times)
    dt = 1.0 / freq

    margin = int(0.15 * num_samples)
    stop_samples = rng.choice(range(margin, num_samples - margin), size=num_stops, replace=False)
    stop_samples.sort()

    decel_duration = 0.3
    hold_duration = 0.2
    accel_duration = 0.3

    decel_samples = int(decel_duration * freq)
    hold_samples = int(hold_duration * freq)
    accel_samples = int(accel_duration * freq)
    total_stop_samples = decel_samples + hold_samples + accel_samples

    for stop_idx in stop_samples:
        end_idx = min(stop_idx + total_stop_samples, num_samples)
        if end_idx - stop_idx < total_stop_samples // 2:
            continue

        vel_at_stop = velocities[stop_idx].copy()

        for t in range(stop_idx, end_idx):
            phase_t = t - stop_idx
            if phase_t < decel_samples:
                s = 0.5 * (1.0 + np.cos(np.pi * phase_t / decel_samples))
                velocities[t] = vel_at_stop * s
                accelerations[t] = vel_at_stop * (
                    -0.5 * np.pi / decel_duration * np.sin(np.pi * phase_t / decel_samples)
                )
            elif phase_t < decel_samples + hold_samples:
                velocities[t] = 0.0
                accelerations[t] = 0.0
            else:
                restart_t = phase_t - decel_samples - hold_samples
                s = 0.5 * (1.0 - np.cos(np.pi * restart_t / accel_samples))
                velocities[t] = vel_at_stop * s
                accelerations[t] = vel_at_stop * (
                    0.5 * np.pi / accel_duration * np.sin(np.pi * restart_t / accel_samples)
                )

        for t in range(stop_idx + 1, num_samples):
            positions[t] = positions[t - 1] + velocities[t] * dt

    return positions, velocities, accelerations


def add_friction(
    torques: np.ndarray,
    velocities: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
) -> np.ndarray:
    """Add realistic joint friction to torques.

    Models three friction components found in real geared robot joints:
    1. Viscous friction: Fv * vel — proportional to velocity, dominates at higher speeds.
       Fv is read from the URDF <dynamics damping="..."/> attribute.
    2. Coulomb friction: Fc * sign(vel) — constant torque opposing motion direction.
       Fc is read from the URDF <dynamics friction="..."/> attribute.
    3. Stribeck stiction: Fs * exp(-|vel|/vs) * sign(vel) — extra breakaway torque
       near zero velocity that decays exponentially as speed increases. Fs is derived
       proportional to Fc (or torque limit if Fc=0) in JointProperties.

    The sign function is smoothed with tanh(vel/threshold) to avoid discontinuity
    at zero velocity.

    Limitations: real friction is more complex — it depends on temperature, load,
    direction history, and lubrication state. The linear Fv + Fc model is what the
    identifier can capture; the Stribeck term creates residuals that test robustness.
    The Stribeck velocity and sign smoothing threshold are configurable via
    JointProperties (simulateStribeckVelocity, simulateFrictionSignThreshold in config).
    """
    stribeck_vel = jp.stribeck_velocity
    sign_threshold = jp.friction_sign_threshold

    friction_torques = np.zeros_like(torques)
    for j in range(jp.num_dofs):
        fv = jp.viscous_friction[j]
        fc = jp.coulomb_friction[j]
        fs = jp.stiction[j]
        vel = velocities[:, j]

        viscous = fv * vel
        smooth_sign = np.tanh(vel / sign_threshold)
        stribeck_decay = np.exp(-np.abs(vel) / stribeck_vel)
        coulomb_stiction = (fc + fs * stribeck_decay) * smooth_sign

        friction_torques[:, torque_col_offset + j] = viscous + coulomb_stiction

    return friction_torques


def add_backlash(
    positions: np.ndarray,
    velocities: np.ndarray,
    jp: JointProperties,
) -> np.ndarray:
    """Add gear backlash effect to measured positions.

    When velocity reverses direction, there's a dead zone where the motor
    shaft turns but the link-side encoder doesn't register movement. This
    is caused by clearance between gear teeth.

    Implementation tracks a backlash offset that accumulates within
    ±half_width as the joint moves. When the offset saturates, motion
    passes through. The half-width (from JointProperties) is derived from
    the gear ratio in the URDF transmission element.

    Limitations: real backlash is load-dependent — under high torque,
    gear compliance takes up the clearance and backlash effectively
    disappears. This model uses a simple position dead zone which
    represents the worst case (unloaded gear reversal).
    """
    positions_with_backlash = positions.copy()
    for j in range(jp.num_dofs):
        half_width = jp.backlash[j]
        offset = 0.0
        for t in range(1, len(positions)):
            delta = positions[t, j] - positions[t - 1, j]
            offset += delta
            offset = np.clip(offset, -half_width, half_width)
            positions_with_backlash[t, j] = positions[t, j] - offset

    return positions_with_backlash


def add_encoder_quantization(
    positions: np.ndarray,
    jp: JointProperties,
) -> np.ndarray:
    """Quantize positions to simulate finite encoder resolution.

    Real position encoders have finite resolution (typically 13-19 bit
    per motor revolution). The effective output-side resolution is much
    higher when a gear reduction is present: effective_bits = motor_bits +
    log2(gear_ratio). The encoder_bits field in JointProperties is the
    effective output-side resolution derived from the URDF gear ratio.

    Implementation rounds positions to the nearest encoder count.

    Limitations: real encoders may have non-uniform quantization steps,
    interpolation errors, and eccentricity. This model assumes ideal
    uniform quantization which is the dominant effect.
    """
    positions_quantized = positions.copy()
    for j in range(jp.num_dofs):
        bits = int(jp.encoder_bits[j])
        resolution = 2.0 * np.pi / (2**bits)
        positions_quantized[:, j] = np.round(positions[:, j] / resolution) * resolution

    return positions_quantized


def add_timing_jitter(
    times: np.ndarray,
    freq: float,
    rng: np.random.Generator,
    jp: JointProperties | None = None,
) -> np.ndarray:
    """Add realistic timing jitter to sample timestamps.

    Real control loops have small variations in loop timing due to
    OS scheduling, communication latency, and interrupt handling.
    The jitter is derived from the joint-level control loop rate
    rather than the data sampling rate — the control loop is the primary
    timing source, and data is downsampled from it.
    """
    # jitter std is ~1% of the control period (not the data period)
    control_rate = jp.control_rate if jp is not None else 1000.0
    control_dt = 1.0 / control_rate
    jitter_std = 0.01 * control_dt
    jitter = rng.normal(0, jitter_std, len(times))
    jitter[0] = 0.0
    times_jittered = times + jitter
    # ensure monotonically increasing
    times_jittered = np.maximum.accumulate(times_jittered)
    return times_jittered


def add_temperature_friction_drift(
    torques: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
) -> np.ndarray:
    """Simulate temperature-dependent friction drift.

    As joints warm up during operation, lubricant viscosity decreases and
    friction drops. Joints that move more (higher RMS velocity) warm up faster.
    Modeled as an exponential approach to a reduced friction level, with
    per-joint thermal time constants (typically 5-20 minutes for geared joints).

    thermal_warmup_time offsets the time axis so the robot doesn't start from
    cold — e.g., if set to 600s, the robot is assumed to have been running
    for 10 minutes before the trajectory begins. The friction reduction at
    time t is: reduction * vel_scale * (1 - exp(-(t + warmup) / tau_thermal)).

    Limitations: this is a highly simplified thermal model. Real thermal behavior
    depends on ambient temperature, cooling, duty cycle history, and lubrication
    type. The effect magnitude (thermal_reduction, typically ~12%) and time
    constants are difficult to determine without dedicated thermal measurements.
    For short trajectories (<1 min) with a pre-warmed robot, the drift is small.
    """
    drift_torques = np.zeros_like(torques)
    t_offset = jp.thermal_warmup_time  # how long robot was already running before trajectory
    for j in range(jp.num_dofs):
        tau_thermal = jp.thermal_tau[j]
        reduction = jp.thermal_reduction
        # amplitude is the viscous friction that gets reduced by temperature
        fric_amp = jp.viscous_friction[j] * reduction

        vel_rms = np.sqrt(np.cumsum(velocities[:, j] ** 2) / (np.arange(len(times)) + 1))
        vel_scale = vel_rms / (np.max(np.abs(velocities[:, j])) + 1e-10)
        # offset time so warmup doesn't start from zero
        effective_time = times + t_offset
        warmup = 1.0 - reduction * vel_scale * (1.0 - np.exp(-effective_time / tau_thermal))
        sign = np.sign(velocities[:, j])
        drift_torques[:, torque_col_offset + j] = -fric_amp * (1.0 - warmup) * sign

    return drift_torques


def add_cable_forces(
    torques: np.ndarray,
    positions: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate configuration-dependent cable/hose routing forces.

    Real robots have cables, pneumatic hoses, and sometimes optical fibers
    routed along the arm that exert small configuration-dependent forces on
    joints. Modeled as a nonlinear spring: F = -k * deflection * (1 + 0.3 * deflection²),
    where deflection is the distance from a random rest angle per joint.
    Stiffness scales with cumulative mass outboard of each joint (proximal
    joints carry more cables) and can be scaled via cable_stiffness_scale.

    Limitations: this is one of the most uncertain effects. The actual cable
    routing, stiffness, and rest configuration vary between individual robots
    of the same model, change when cables are re-routed, and depend on
    cable age and temperature. The random rest angles make the effect
    non-deterministic (seeded by the rng). Consider disabling this effect
    (cable_stiffness_scale=0) unless you have reason to believe cable forces
    are significant for your robot.
    """
    if rng is None:
        rng = np.random.default_rng(99)

    # rest angles are random per joint (cable routing is robot-specific)
    rest_angles = rng.uniform(-0.5, 0.5, jp.num_dofs)

    cable_torques = np.zeros_like(torques)
    for j in range(jp.num_dofs):
        k = jp.cable_stiffness[j]
        q0 = rest_angles[j]
        deflection = positions[:, j] - q0
        cable_torques[:, torque_col_offset + j] = -k * deflection * (1.0 + 0.3 * deflection**2)

    return cable_torques


def add_gravity_compensation_residual(
    torques: np.ndarray,
    positions: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
) -> np.ndarray:
    """Simulate imperfect gravity compensation residual.

    Some robot controllers apply built-in gravity compensation based on a
    nominal dynamics model. If the real mass/COM parameters differ from the
    nominal model, the measured torques contain a residual gravity-dependent
    error. This is modeled as: residual = error_frac * g * cum_mass * lever * sin(q),
    where cum_mass is the total mass outboard of each joint and lever is an
    approximate average distance (~15 cm).

    This effect only applies when torques are derived from motor current
    (where the controller's gravity comp is subtracted). For robots with
    joint torque sensors, the sensors measure actual physical torques directly
    and this effect should be disabled (simulateGravCompError=0 in config).

    Limitations: the sinusoidal position dependence is a crude approximation
    of the true gravity torque, which depends on the full kinematic chain
    configuration. The error fraction depends on how well the controller's
    internal model matches reality.
    """
    residual = np.zeros_like(torques)
    for j in range(jp.num_dofs):
        error_frac = jp.grav_comp_error[j]
        # approximate gravity torque amplitude from cumulative mass beyond this joint
        cum_mass = np.sum(jp.link_mass[j:])
        grav_amp = cum_mass * 9.81 * 0.15  # ~15cm average lever arm
        grav_approx = grav_amp * np.sin(positions[:, j])
        residual[:, torque_col_offset + j] = error_frac * grav_approx

    return residual


def add_structural_deflection(
    positions: np.ndarray,
    torques: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
) -> np.ndarray:
    """Simulate load-dependent structural deflection.

    Under load, the robot structure (links, bearings) deflects slightly.
    The encoder reads the motor-side position, but the actual link-side
    position differs by a small amount proportional to the joint torque
    (linear spring model). Compliance is inversely proportional to joint
    torque capacity — stiffer joints for higher torques.
    """
    positions_deflected = positions.copy()
    for j in range(jp.num_dofs):
        compliance = jp.compliance[j]
        joint_torque = torques[:, torque_col_offset + j]
        positions_deflected[:, j] += compliance * joint_torque

    return positions_deflected


def add_torque_quantization(
    torques: np.ndarray,
    jp: JointProperties,
    torque_col_offset: int = 0,
) -> np.ndarray:
    """Simulate motor current / PWM discretization in torque commands.

    The motor driver converts a continuous torque command to discrete
    PWM duty cycles, creating small step-like quantization in the
    actual applied torque. Resolution is 11-16 bits over the joint's
    torque range (from URDF effort limit).
    """
    torques_quantized = torques.copy()
    for j in range(jp.num_dofs):
        bits = int(jp.torque_quant_bits[j])
        t_range = jp.torque_limit[j]
        resolution = 2.0 * t_range / (2**bits)
        col = torque_col_offset + j
        torques_quantized[:, col] = np.round(torques[:, col] / resolution) * resolution

    return torques_quantized
