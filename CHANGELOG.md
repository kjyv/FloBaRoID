# Changelog

## 0.9.0

### Build system
- Migrated from `requirements.txt` to `pyproject.toml` + `uv`
- Requires Python >= 3.13 (up from 2.7 / 3.3)
- All dependencies pinned to exact versions

### iDynTree migration (0.11 → 14.2.0)

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

### Cleanup
- Removed Python 2 compatibility (`__future__`, `builtins`, `future` package)
- Removed `distutils` usage (version checks for ancient library versions)
- Fixed `time.clock()` → `time.perf_counter()`
- Fixed `yaml.load()` to use `SafeLoader`
- Fixed `np.core.arrayprint` deprecation
- Added graceful fallback when `dsdp5` is not installed

## 0.1.0

Initial release (2017-2018).
