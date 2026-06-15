# Friction identification and floating-base excitation — design notes

Rationale for the friction-identification, excitation and collision-checking choices in
the floating-base pipeline. These are durable design notes; concrete benchmark numbers
from individual experiments are intentionally not recorded here.

## Two-step friction identification

Friction (Coulomb `Fc`, viscous `Fv`, constant offset) is identified after the inertial
parameters, by fitting the per-joint torque residual `tau_measured - tau_inertial` with
ordinary least squares on the regressor `[sign(v), v, 1]`. The inertial parameters are
identified from the friction-free base-wrench equations (Ayusawa base-link dynamics), so
the residual the friction OLS sees is dominated by friction rather than inertial error.

Design points:

- **Smoothed Coulomb sign term.** The sign of velocity is smoothed with
  `tanh(v / frictionSignThreshold)` to match the simulator's continuous friction model
  and to avoid a discontinuity at zero velocity. The term is centralized in
  `helpers.getFrictionSignSeries` so the regressor columns, the friction OLS and all
  torque predictions use exactly the same series.
- **Sign-velocity filtering (`frictionVelocityCutoff`).** Computing the steep `tanh`
  from noisy measured velocity makes the predicted Coulomb torque chatter by `±Fc`
  whenever a joint hovers within its velocity noise floor around zero. The velocity used
  for the sign is therefore taken from the raw measured velocity, low-pass filtered with a
  zero-phase filter at a cutoff just above the trajectory's velocity bandwidth. Because the
  filter is zero-phase it does not shift zero crossings, so a low cutoff removes noise
  without distorting the sign timing. A Schmitt-trigger hysteresis was evaluated as an
  alternative and rejected: holding a stale sign through a slow zero crossing produces a
  larger error than the zero-mean chatter it removes.
- **Velocity dead zone (`frictionVelocityDeadZone`).** Below the velocity noise floor the
  `tanh` is in its near-linear region and becomes collinear with the viscous `v` column,
  so `Fc` and `Fv` cannot be separated. Following Swevers et al., samples with
  `|v|` below the dead zone are excluded from the OLS per joint (with a fallback to all
  samples if too few, or only one motion direction, remain). Above roughly twice the sign
  threshold the `tanh` is saturated and maximally decorrelated from `v`.
- **Viscous regularization (`frictionFvRegularization`).** `Fv` is poorly determined for
  joints that barely move and tends to inflate, absorbing unmodeled effects. A Tikhonov
  row per joint pulls `Fv` toward the a priori URDF value. The data term's `Fv`
  information scales with the kept samples' velocity energy, so a fixed weight is
  automatically per-joint adaptive: negligible for well-excited joints, dominant for
  weakly-excited ones. On real hardware the weight should reflect how much the a priori
  friction values are trusted.

`identifier.py` prints friction parameter errors against the real model when
`--model_real` is supplied — parameter quality is the metric that matters; the fit NRMS
and parameter ranges are only proxies.

## Simulator measurement semantics

The simulator stores the noisy measurements under the `*_raw` keys (positions, velocities,
torques), matching the meaning those keys have for real data produced by `excite.py` /
`tools/csv2npz.py`. The clean, pre-noise reference signals are available separately as
`target_*`. Friction-sign filtering and any other "raw measurement" processing therefore
behave identically on simulated and real data.

## Excitation (trajectory optimization)

The trajectory optimizer maximizes a regularized D-optimality objective on the inertial
base regressor, plus soft costs for torque balance, torque magnitude and position-range
utilization, subject to joint, velocity, torque and collision constraints. Global search
is Optuna (constraint-aware), local refinement is IPOPT with analytical gradients.

- **Velocity excitation (`trajectoryTargetVelocity`).** The inertial D-optimality
  objective does not reward joint velocity, and in two-step friction mode there are no
  friction columns in the regressor, so friction excitation is otherwise invisible to the
  optimizer and weakly-moving joints stay slow. An optional soft cost penalizes joints
  whose peak velocity stays below the target, with an analytical gradient.
- **Feasibility navigation.** For a constrained humanoid the feasible region is small and
  most random/global candidates are infeasible. Two mechanisms help: `globalOptAmplitudeRepair`
  repairs an infeasible candidate in place by scaling its Fourier amplitudes toward zero
  until the constraints are satisfied (the trial then reports the repaired objective, so
  the sampler is biased toward repairable shapes), and `trajectorySeedSolutions` enqueues
  known-feasible previously-optimized trajectories as initial trials.
- **Sequential experiment design (`trajectoryPriorMeasurements`).** The accumulated
  information matrix `sum(Y^T Y)` of previously executed trajectories is added to the
  D-optimality objective, so a new trajectory maximizes the information it adds in the
  directions earlier trajectories left weak. Requires `useStructuralRegressor` so the base
  projection matches between the prior and per-iteration regressors.

## Multi-trajectory identification

Identifying from several measurement files (passed as repeated `--measurements`) has
always been supported; the files are concatenated. `useTrajectoryWeighting` weights each
file's base-wrench equations by the inverse of its residual noise (estimated from an OLS
pre-pass) instead of stacking all samples with equal weight, so a noisier session does not
dominate. This matters mainly for real data, where per-session noise genuinely differs;
for equal-noise simulated data the weights are near uniform.

Note that pooling helps only when an added trajectory brings more information than
trajectory-specific bias (from its own unmodeled effects). A trajectory that is poor on
its own can dilute a combined fit; gate on held-out validation before pooling.

## Collision checking for the suspended floating base

The robot is identified while suspended from a crane, and its base swings according to the
simulated suspended dynamics. Collision checking must reflect this:

- Robot links are placed at the **simulated (swung) base pose**, not the fixed mount pose,
  for every checked configuration — otherwise robot-vs-world clearance is evaluated at a
  pose the robot never holds.
- World geometry (crane, ground) is included for the suspended attachment, placed at its
  composed URDF joint-chain pose.
- The ramp-in/out transition segments are checked against representative swung poses,
  because the suspension keeps swinging well past the transition duration.
- `worldCollisionMargin` enforces a clearance to world geometry, because the box hulls
  under-approximate protruding parts (fingers) and `scaleCollisionHull` shrinks them, so
  visual contact can occur at a positive hull distance.
- `ignoreCollisionBetweenGroups` excludes whole groups of link pairs from checking. Use it
  only for pairs that are genuinely unreachable across the full configuration space. In
  particular do not assume the legs cannot reach the arms/hands: high-velocity suspended
  trajectories can bring a hand down onto a lower leg. The constraint count is derived
  from the actual checked-pair list, so changing the pairing configuration cannot
  desynchronize it.

The visualizer (`visualizer.py`) highlights both robot self-collisions and robot-vs-world
clearance violations using the same geometry and margin as the optimizer, so a trajectory
can be inspected before it is executed.

## Evaluation

- Compare identified parameters to the real model directly (parameter distance, friction
  errors) when a real model is available — torque-fit metrics are only proxies, and for
  floating-base robots torque data determines the base parameters, not all standard
  parameters.
- Report the torque NRMS on a **held-out** trajectory (`--validation`), not only on the
  training trajectory: the training-trajectory residual understates the true error because
  each trajectory leaves a different parameter subspace underdetermined.
