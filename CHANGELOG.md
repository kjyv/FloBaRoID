# Changelog

## 0.9.3

### Suspended base dynamics
- Added ball-joint suspended dynamics simulation for floating-base identification testing.
  The robot hangs from a configurable attachment frame (default: `crane_ft`), and base
  angular motion is computed from the Newton-Euler equations at each timestep.
- Integrated into both trajectory optimizer and simulator via `floatingBaseAttachment` config
- New config options: `floatingBaseAttachment` (`fixed`/`suspended`/`free`),
  `floatingBaseAttachmentFrame`, `suspendedDamping`
- Equilibrium finder locates the hanging-down RPY before integration starts
- World URDF (`model/world_walkman_suspended.urdf`) for crane visualization
- Base velocity computed analytically via iDynTree `getFrameVel` (avoids numerically
  unstable RPY finite differences)

### Simulator
- Fixed fabricated base motion: simulator no longer generates synthetic sinusoidal base
  RPY/velocity/acceleration; uses trajectory data as-is
- Simulator output now defaults to `<urdf>.measurements.npz` instead of overwriting the
  trajectory file

### Identification
- Measured base wrench is preserved when already present in the measurement data (e.g. from
  simulator or F/T sensor). Previously always overwritten with a priori model simulation.
- Fixed missing `contactForcesSum` in `rho2_norm_sqr` for `identifyFeasibleBaseParameters`
  and `identifyFeasibleStandardParametersDirect` (Schur complement consistency)
- Fixed `onlyUseDSDP` config option that was hardcoded to 0 (dead code)
- Added symmetry enforcement on pseudo-inertia matrix (BlockMatrix expansion can lose it)
- Added division-by-zero guard on COM computation for zero-mass links
- Fixed `checkFeasibility` crash on SymPy `Zero` scalar (from pinned parameters)
- Fixed `checkFeasibility` to use `eigvalsh` instead of `eig` for symmetric matrices
- `dontChangeLinks`: pin all parameters of named links to a priori (replaces manual
  parameter index lists)
- `sdpSafeMargin`: configurable eigenvalue lower bound for physical consistency constraints
- `sdpFeasTol`: configurable cvxopt feasibility tolerance for thin feasible regions
- Improved cvxopt error reporting (iterations, gap, primal/dual infeasibility)

### Trajectory optimization
- World collision links excluded for suspended dynamics (robot swings freely)
- Ctrl-C now cleanly terminates all Optuna worker processes and gradient pool
- `globalOptJobs: 0` now means auto-detect cores (consistent with `analyticalGradientJobs`)
- Clearer logging when IPOPT starts from global best vs default initial trajectory

### Visualizer
- Floating-base visualization: displays base RPY and position from measurement data
- World URDF links now properly positioned using joint chain transforms (was ignoring
  joint origins, only using visual origins)
- `getLinkGeometry` handles cylinder and sphere primitives (was box-only)

## 0.9.2

### Trajectory optimization
- Replaced condition number objective with regularized D-optimality (`-log(det(Y^T Y + δI))`)
  - Numerically stable at any condition number (condition number gradient was broken for cond > 1e8)
  - Regularization `δ = ε·λ_max` ensures analytical gradient accuracy to 3-4 significant digits
  - Works for structurally rank-deficient regressors (full humanoids with unobservable parameters)
- Analytical gradient passed directly to IPOPT, eliminating 182 FD objective evaluations per iteration
  (~15-28x fewer calls per iteration)
- Added observability analysis: SVD identifies unobservable parameters, saved to trajectory file,
  automatically constrains them to a priori values during identification
- Collision checking ~10x faster: reuse FCL objects across samples, exclude impossible pairs
- `ignoreCollisionBetweenGroups` config option for excluding cross-body collision pairs
- Bounded trajectory now guarantees positions stay within URDF limits regardless of center offset

### Configuration
- `collisionMode`: single option replacing `useCollisionMeshes`/`useConvexHullCollision`/`useCapsuleCollision`
  (values: `box`, `convex`, `full`, `capsule`)
- `fullMeshLinks`: replaces `bvhMeshLinks` — per-link override for full triangle mesh collision
- `trajectoryOscillationCenters`: named dict of preferred joint centers (replaces `trajectoryAngleRanges`)
- `trajectoryCenterFreedom`: how far optimizer can shift centers from preferred values
- `trajectoryNf`: named dict format (`{joint_name: nf}`) with validation
- `symmetryLinkPairs`: named link pairs replacing raw parameter index triples
- `ovrPosLimit`: named dict format (`{joint_name: [lo, hi]}`)
- `doptRegularization`: controls D-optimality gradient conditioning (default 1e-4)

### Identification
- Unobservable parameters from trajectory file automatically added to `dontChangeParams`
- Loads observability data from measurement files passed via `--measurements`
- Fixed missing `contactForcesSum` in `rho2_norm_sqr` for `identifyFeasibleBaseParameters`
  and `identifyFeasibleStandardParametersDirect` (Schur complement consistency)
- Fixed `onlyUseDSDP` config option that was hardcoded to 0 (dead code)
- Added symmetry enforcement on pseudo-inertia matrix (BlockMatrix expansion can lose it)
- Added division-by-zero guard on COM computation for zero-mass links

### Visualizer
- Handles old measurement files with fewer joints than current model
- Plotly.js served locally for offline HTML output

## 0.9.1

### Trajectory generation
- Added bounded trajectory generation to prevent running into joint angle limits
- Added trajectory simulator for generated trajectories
- Fixes for floating base trajectory generation

### Trajectory optimization
- Added Optuna optimizer with multi-core support
- Added analytical gradient computation for condition number optimization (Ayusawa et al., ICRA 2017), using SVD-based condition number sensitivities chained with analytical Fourier trajectory Jacobians
- Added capsule-based collision detection with analytical distance gradients, eliminating finite-difference perturbations for collision constraints (~93x faster collision gradients on the full Walkman)
- Added trajectory quality penalties
- Added mesh-based collision detection with configurable convex hull (`useConvexHullCollision`)
- Fixed playback rate in trajectory visualization
- New config options: `useCapsuleCollision`, `scaleCapsuleRadius`, `useConvexHullCollision`, `collisionMaxKinematicDistance`

### Visualizer
- Added capsule collision geometry display (cycle with M key: visual → collision → capsules → boxes)
- Collision checking now matches the selected geometry mode (capsule/mesh/visual)
- Collision mesh rendering now shows convex hulls matching what the optimizer uses
- Fixed inverted meshes for links with negative URDF scale (mirrored geometry)
- Fixed rendering of links that were incorrectly hidden by the collision ignore list
- Added collision checking toggle (C key) and moved render mode toggle to B key

### Identification
- Fixed missing sign for Coulomb friction identification
- Fixed mid-point p offset calculation

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
