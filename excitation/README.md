
# Excitation: Trajectory Generation and Optimization

## Overview

The excitation module generates optimal trajectories for robot parameter identification. The goal is
to find joint motions that maximize the information content of the resulting measurement data, so that
the inertial parameters (mass, center of mass, inertia tensor per link) can be identified as
accurately as possible from inverse dynamics.

The approach follows Swevers et al. (1997): joint trajectories are parameterized as truncated Fourier
series, and the Fourier coefficients are optimized to improve the quality of the regressor matrix
`Y` in the linear relation `Y · φ = τ` (regressor × parameters = torques).

## Regressor Quality Criterion

The optimizer minimizes a **regularized D-optimality** criterion:

```
f = -log(det(Y^T Y + δI))
```

This is equivalent to maximizing the determinant of the information matrix, which simultaneously
improves all identifiable parameter directions. The regularization `δ = ε · λ_max` ensures numerical
stability even when the regressor is structurally rank-deficient (common for full humanoids where
some parameters are inherently unobservable).

Previous versions used the condition number `cond(Y) = σ_max / σ_min`, but this is numerically
unstable for ill-conditioned regressors (the analytical gradient involves `1/σ_min` which amplifies
floating-point noise). D-optimality avoids this issue entirely.

## Optimization Pipeline

The trajectory optimization uses a two-stage approach:

1. **Global exploration** (optional): Optuna with TPE sampler explores the parameter space broadly,
   evaluating many random Fourier coefficient combinations to find promising regions.

2. **Local refinement**: IPOPT (interior-point method) refines the best solution using analytical
   gradients. The gradient computation chains:
   - **Phase 1**: SVD-based D-optimality gradient weights `R = -2·scale·U·diag(s/(s²+δ))·V^T`
   - **Phase A**: Regressor sensitivities `dY/d(q,dq,ddq)` via finite differences of the iDynTree
     regressor (parallelized across CPU cores)
   - **Phase B**: Trajectory Jacobians `d(q,dq,ddq)/d(α)` — analytical Fourier derivatives chained
     with the regressor sensitivities via the chain rule
   - **Phase C**: Penalty gradients for torque utilization, position range, and balance

The analytical gradient is passed directly to IPOPT, eliminating ~182 finite-difference probes per
iteration that pyOptSparse would otherwise need. Each IPOPT iteration takes ~40s instead of ~3-5 min.

### Penalty Terms

In addition to the D-optimality criterion, the objective includes soft penalties:

- **Torque balance** (CoV): penalizes uneven torque utilization across joints
- **Torque magnitude**: penalizes low overall torque utilization (poor signal-to-noise)
- **Position range**: penalizes joints that don't use their available range of motion

Hard constraints enforce joint position limits, velocity limits, torque limits, minimum torque
utilization, and collision avoidance.

## Bounded Trajectories

For robots with many DOFs, the `trajectoryBounded: 1` mode uses a tanh-based mapping that
guarantees joint positions stay within URDF limits:

```
q(t) = q_center + q_range · tanh(Σ aₖ sin(k·ω·t) + bₖ cos(k·ω·t))
```

The center offset is configurable per joint via `trajectoryOscillationCenters`, and the range
automatically shrinks to fit within URDF limits regardless of the offset.

## Observability Analysis

After optimization, the trajectory file includes an observability analysis based on the SVD of the
regressor. Parameters whose corresponding singular values fall below a threshold are flagged as
unobservable — they cannot be reliably identified from this trajectory. The identifier automatically
constrains these to their a priori values during SDP-based identification.

## Collision Checking

The optimizer checks for self-collisions at sampled trajectory points. Collision geometry can be
configured via `collisionMode`:

- `box`: axis-aligned bounding boxes (fastest, least accurate)
- `convex`: convex hulls computed from collision/visual meshes (default, good tradeoff)
- `full`: raw triangle meshes via BVH (most accurate, slowest)
- `capsule`: analytical capsule primitives (fast, with analytical gradients)

Per-link overrides (`fullMeshLinks`) allow using full meshes for specific concave links (e.g.,
protective shells) while using convex hulls everywhere else.

## Running

### Optimize a trajectory

```bash
uv run trajectory.py --config configs/kuka_lwr4.yaml --model model/kuka_lwr4.urdf
```

This saves the optimized trajectory parameters to `<model>.trajectory.npz` by default (override
with `--filename`). The resulting trajectory file can then be passed to `excite.py` via
`--trajectory`.

For the full humanoid:

```bash
uv run trajectory.py --config configs/walkman_full.yaml --model model/walkman_measured.urdf
```

### Simulate measurements

```bash
uv run simulator.py --config configs/walkman_full.yaml --model model/walkman_measured.urdf \
    --trajectory model/walkman_measured.urdf.trajectory.npz --filename measurements.npz
```

### Visualize a trajectory

```bash
uv run visualizer.py --config configs/kuka_lwr4.yaml --model model/kuka_lwr4.urdf \
    --trajectory measurements.npz
```

## Configuration

Key trajectory optimization options (see sample configs for all options):

| Option | Description |
|--------|-------------|
| `trajectoryBounded` | Use tanh-bounded trajectories (recommended for >6 DOF) |
| `trajectoryNf` | Fourier harmonics per joint: `{joint_name: nf}` |
| `trajectoryOscillationCenters` | Preferred center offsets: `{joint_name: degrees}` |
| `trajectoryCenterFreedom` | Optimizer freedom around preferred centers (degrees) |
| `trajectoryPulseMin/Max` | Angular frequency range (rad/s), determines trajectory duration |
| `useAnalyticalGradients` | Enable analytical D-optimality gradient for IPOPT |
| `doptRegularization` | D-optimality regularization strength (default 1e-4) |
| `collisionMode` | Collision geometry: `box`, `convex`, `full`, or `capsule` |
| `collisionCheckStep` | Check every Nth sample for collisions |
| `ignoreCollisionBetweenGroups` | Skip collision checks between link groups |

## Measurements Data File Structure

The measurements retrieved from excitation are saved in a numpy `.npz` binary file archive which
includes multiple data streams. All data fields have the same number of samples S relative to the
time in field `times`. The same structure is expected by `identifier.py`.

| Field | Content |
|-------|---------|
| `positions` | Joint positions in rad, S × N_DOF |
| `positions_raw` | Unfiltered joint positions, S × N_DOF (optional) |
| `velocities` | Joint angular velocity in rad/s, S × N_DOF |
| `velocities_raw` | Unfiltered joint angular velocities (optional) |
| `accelerations` | Joint angular accelerations in rad/s², S × N_DOF |
| `torques` | Measured torques per joint in Nm, S × N_DOF |
| `torques_raw` | Unfiltered torques (optional) |
| `base_velocity`* | Linear (0-2) and angular (3-5) base velocity in m/s and rad/s, S × 6 |
| `base_acceleration`* | Proper base acceleration (without gravity), S × 6 |
| `base_rpy`* | Base orientation in roll-pitch-yaw, S × 3 |
| `contacts` | External contact wrenches, array of dicts `{frame_name: S × 6}` |
| `times` | Sample times in seconds, S × 1 |
| `frequency` | Sampling frequency in Hz |

*Only required for floating base dynamics.

All data is expected to be cleaned, low-pass filtered, and free of large measurement errors.
The noise should ideally be Gaussian with zero mean.

## Data Requirements

The sampling frequency should be sufficiently high (e.g. at least 100 Hz) to get reasonably good
position and velocity derivatives.

The number of samples should be high enough to contain sufficient information about the parameters.
It depends on how many parameters are to be identified and on the motion range of the robot. At
least 10 times the number of parameters is a good rule of thumb, more is always better. The higher
the sampling frequency, the less information there is in successive samples, so the number should be
higher at e.g. 1000 Hz. At the same time, more or less redundant samples can be skipped by setting
the `skipSamples` option to speed up identification.

## Excitation Modules

Existing modules that will send commands to a robot and record measurements can be used with slight
modifications or as template for other robot command interfaces.

### Generate excitation for Walk-Man

A Yarp GYM module is included that has to be built and started:

in robotCommunication/yarpGYM/:

```bash
mkdir build && cd build
cmake ../
make
```

Run `yarpserver --write` and then `gazebo`, load the robot.

Run `./excitation`

```bash
yarp write ... /excitation/switch:i
>> start
```

The control thread is now started and accepts commands.

Using `yarp write ... /excitation/command:i` it is then possible to manually set positions, e.g.:
```
(set_legs_refs 0 0 0 0 0 0 0 0 0 0 0 0) 0
```

To generate excitation trajectories and send them to the robot, set the option `exciteMethod` to
`'yarp'` and run:

```bash
uv run excite.py --config configs/walkman_full.yaml --model model/walkman_measured.urdf --filename measurements.npz
```

This will also read the resulting joint torque measurements and write them to the given file.

### Generate excitation for Kuka LWR4+

Using ROS package from https://github.com/CentroEPiaggio/kuka-lwr:

Start controllers, simulator and moveit (to directly use the hardware add options:
`use_lwr_sim:=false lwr_powered:=true`):

```bash
roslaunch single_lwr_launch single_lwr.launch load_moveit:=true
```

(Make sure that the gazebo plugin gets loaded in the world file and `joint_state_publisher` has a
high enough rate param of 100-200 Hz set in the launch file.)

To generate excitation trajectories and send them to the robot, set the option `exciteMethod` to
`'ros'` and run:

```bash
uv run excite.py --config configs/kuka_lwr4.yaml --model model/kuka_lwr4.urdf --filename measurements.npz
```

## References

- Swevers et al. (1997): "Optimal robot excitation and identification" — Fourier trajectory parameterization
- Ayusawa et al. (2017): "Generating persistently exciting trajectory based on condition number optimization" — analytical condition number gradient
- Sousa & Cortesão (2014): "Physical feasibility of robot base inertial parameter identification" — SDP-constrained identification
