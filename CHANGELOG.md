# Changelog

## 0.9.0

### Build system
- Migrated from `requirements.txt` to `pyproject.toml` + `uv`
- Requires Python >= 3.13 (up from 2.7 / 3.3)
- All dependencies pinned to exact versions

### iDynTree migration (0.11 → 15.0.0)

The old `DynamicsRegressorGenerator` and `DynamicsComputations` classes depended on
KDL, which was removed in iDynTree 2.0. Both are replaced by `KinDynComputations`
(plus `ModelLoader` / `Model` for model access).

**Regressor computation**: The old `generator.computeRegressor(Y, tau)` required
loading a regressor structure XML to specify which joints to include, then called
`setRobotState()` and `computeRegressor()` separately. The new
`kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq, Y)` computes the
regressor for all joints directly after `setRobotState()`. It always returns a
`(6 + n_dof) x (n_links * 10)` matrix (base wrench + joint torques vs. all
inertial parameters). For fixed-base use, the first 6 rows (base wrench) are
discarded.

**Frame convention change**: The old regressor expressed base wrench rows in the
base frame, requiring manual rotation to world frame. The new API uses the same
frame for both regressor and inverse dynamics, so the rotation workaround was
removed.

**State vectors**: `VectorDynSize.fromList()` replaced by `.FromPython()`.
Joint state vectors changed from `VectorDynSize` to `JointPosDoubleArray` /
`JointDOFsDoubleArray` (populated via `setVal()` loops). Gravity changed from
`Twist`/`SpatialAcc` (6D) to `Vector3` (3D). Base acceleration uses `Vector6`.

**Other changes**:
- `generator.getModelParameters()` → `model.getInertialParameters()`
- `generator.getNrOfFakeLinks()` removed (concept no longer exists)
- `generator.getDescriptionOfParameters()` → new `Model.getDescriptionOfParameters()` method
- `dynComp.getFrameJacobian()` → `kinDyn.getFrameFreeFloatingJacobian()`
- `dynComp.getLinkInertia(i)` → `model.getLink(i).getInertia()`
- `dynComp.inverseDynamics(torques, baseForce)` → `kinDyn.inverseDynamics(base_acc, ddq, ext_wrenches, gen_torques)` using `LinkWrenches` and `FreeFloatingGeneralizedTorques`
- `iDynTree.modelFromURDF()` → `ModelLoader.loadModelFromFile()`
- Migrated all tool scripts and tests

### Plotting (mpld3 → plotly)
- HTML and PDF output now use plotly + kaleido instead of matplotlib + mpld3
- Interactive output uses plotly `fig.show()`
- matplotlib only used for tikz export (moved to optional dependency)

### sympy / pylmi-sdp
- Upgraded sympy 1.0 → 1.12.1
- Removed custom `LMI_PD`/`LMI_PSD`/`lmi_to_coeffs` wrappers, using `lmi_sdp` directly

### Trajectory optimization (pyOpt → pyOptSparse)
- Replaced pyOpt (unmaintained since 2014) with pyOptSparse (actively maintained fork)
- pyOptSparse installed from git (not on PyPI)
- Adapted objective functions to dict-based API (`{name: value}` instead of flat arrays)
- Removed COBYLA support (not available in pyOptSparse; use SLSQP instead)
- Added integration tests for OLS and SDP-constrained identification

### Visualizer
- Camera rotation mode and shadow rendering for OpenGL visualizer
- Updated for current OpenGL version

### Cleanup
- Removed RBDL alternative for inverse dynamics
- Removed Python 2 compatibility (`__future__`, `builtins`, `future` package)
- Removed `distutils` usage (version checks for ancient library versions)
- Fixed `time.clock()` → `time.perf_counter()`
- Fixed `yaml.load()` to use `SafeLoader`
- Fixed `np.core.arrayprint` deprecation
- Added graceful fallback when `dsdp5` is not installed

## 0.8.0

### Visualization
- Added OpenGL-based model visualizer with mesh loading (STL/DAE via trimesh)
- Trajectory animation for optimized and measured trajectories

### Nonlinear constrained identification
- Added nonlinear optimization as alternative to SDP for physical consistency (IPOPT, SLSQP, PSQP, ALPSO, NSGA2)
- Traversaro (2016) physically consistent reparametrization (feasible parametrization space)
- MPI-based parallel optimization for global solvers

## 0.7.0

### Gravity-only identification
- Gravity-only parameter identification from static postures
- Static trajectory generation and optimization
- Separate gravity-term regressor computation and storage

### Friction identification
- Viscous and Coulomb friction identification (symmetric and asymmetric)
- Friction parameters read from URDF and used in simulation
- Positive viscous friction constraints in SDP formulation

### SDP and constraint improvements
- Regressor regularization for non-identifiable parameters
- symengine acceleration for SDP matrix construction
- Symmetry constraints between link parameters (e.g. left/right legs)
- Per-link mass constraints around a priori values

## 0.6.0

### Floating base
- Contact forces from force/torque sensor data integrated into dynamics equations
- Base velocity and acceleration computed from IMU measurements
- Contact forces included in SDP formulation
- Jacobian-based contact wrench computation at each sample

### Python 3 support
- Ported identification and excitation modules to Python 3

## 0.5.0

### Trajectory optimization
- Fourier-based parametric trajectory optimization (pyOpt)
- Condition number minimization of the observation matrix
- Global + local optimization (ALPSO → SLSQP)
- Joint and velocity limit enforcement from URDF
- YAML-based configuration files (replacing hardcoded options)
- Removal of near-zero-velocity samples to reduce friction effects

### SDP constrained identification
- SDP-based feasible standard parameter identification (Sousa, 2014)
- SDP-based feasible base parameter identification
- Closest-to-CAD feasible standard parameter recovery from feasible base solution
- Fallback to DSDP5 solver when CVXOPT fails
- COM bounding box constraints from STL mesh geometry
- A priori parameter constraints and mass limiting

## 0.4.0

### Output and analysis
- HTML report generation with interactive plots (matplotlib + mpld3)
- PDF and TikZ export for publication figures
- Per-joint torque estimation error plots
- Parameter comparison tables (CAD vs. identified vs. ground truth)
- Percentual parameter error display (Pham method)

### Data processing
- Block-wise data selection by sub-regressor condition number
- Butterworth filtering for positions, velocities, accelerations, and torques
- Symbolic base parameter dependency equations via sympy (Gautier's method)

## 0.3.0

### Estimation methods
- Weighted Least Squares (WLS) with torque variance weighting
- Essential parameter identification with model reduction
- Direct standard parameter estimation (bypassing base projection)
- A priori parameter weighting for improved standard parameter estimates

### Base parameter computation
- QR decomposition for base parameter projection
- Random regressor sampling for numeric rank estimation
- Basis projection matrix as alternative to permutation-based projection

### Validation
- Separate validation dataset support
- Torque prediction on held-out measurements

## 0.2.0

### Core identification pipeline
- Inverse dynamics regressor computation via iDynTree `DynamicsRegressorGenerator`
- Ordinary Least Squares (OLS) base parameter identification
- Physical consistency checking (positive definite inertia, triangle inequality)
- URDF parameter replacement with identified values
- Support for multiple measurement files

### Robot support
- KUKA LWR4+ with real hardware measurements (via ROS/MoveIt)
- WALKMAN humanoid (legs) with simulated and real data (via YARP)
- Simulated 3-link chain for testing
- CSV and NPZ data import with configurable sample skipping

### Excitation
- YARP-based joint excitation module for WALKMAN
- ROS/MoveIt excitation for KUKA LWR4+
- Sinusoidal trajectory generation

## 0.1.0

Initial commit: iDynTree integration, basic regressor computation, first
identification experiments.
