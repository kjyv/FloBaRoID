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
- **Viscous regularization (`frictionFvRegularization`, `frictionFvRegularizationRelative`).**
  `Fv` is poorly determined for joints that barely move and tends to inflate, absorbing
  unmodeled effects. A Tikhonov row per joint pulls `Fv` toward the a priori URDF value.
  Because the data term's `Fv` information is the kept samples' velocity energy `Σv²`, the
  per-joint solution is the weighted average `Fv = (Σv²·Fv_data + λ·Fv_apriori)/(Σv² + λ)`:
  `λ` is the velocity energy at which data and prior weigh equally, so the weight is
  automatically per-joint adaptive (data-driven where `Σv² ≫ λ`, a priori where `Σv² ≪ λ`).
  The absolute weight is in raw energy units and so depends on the trajectory's speed and
  sampling, which is awkward to set without ground truth. The preferred form is the
  unit-free `frictionFvRegularizationRelative` (`α`): the weight is set to
  `α · median(per-joint Σv²)`, a single value that transfers across robots and trajectories
  while preserving the adaptivity (a joint at the median excitation gets a 50/50 blend at
  `α = 1`). It is set without ground truth — by held-out validation and by raising it until
  no joint's `Fv` collapses to zero (the `Fc`/`Fv` split degenerates for a joint that moves
  at a narrow speed band, where the columns are collinear regardless of weighting). Scaling
  `λ` per joint instead (`α · Σv²_j`) would *not* work: it makes the data/prior ratio
  constant across joints and loses the adaptivity, trusting a barely-moving joint's noisy
  `Fv` as much as a well-excited one's.

`identifier.py` prints friction parameter errors against the real model when
`--model_real` is supplied — parameter quality is the metric that matters; the fit NRMS
and parameter ranges are only proxies.

## Two-step vs. simultaneous friction identification

There are two ways to identify friction, and which one applies is determined by whether a
friction-free measurement is available, not by preference.

- **Two-step (floating base).** Inertial parameters are identified from the base-wrench
  equations, which contain no joint friction at all (`useBaseWrenchForBaseParams`). The
  inertial estimate is therefore unbiased by any friction-model error, and the per-joint
  torque residual `tau_measured - tau_inertial` is fit afterwards (`postIdentifyFriction`)
  with the robust per-joint OLS above. Because each joint's friction is fit separately, the
  velocity dead zone, the `Fv` prior and the smoothed-sign handling all apply cleanly
  per joint.
- **Simultaneous (fixed base, or whenever no base wrench is measured).** Friction columns
  are appended to the standard-parameter regressor and inertial + friction parameters are
  solved jointly inside the SDP (`identifyFrictionSimultaneously`). The friction columns are
  block-diagonal per joint — `tanh`-smoothed sign for `Fc`, velocity for `Fv` (or `Fv±` when
  asymmetric), a constant offset, and an optional Stribeck `Fs * exp(-|v|/vs)` column — so a
  joint's friction only affects its own torque row. The inertial columns, however, couple
  all joints through the dynamics.

A fixed-base robot **cannot** use the clean two-step scheme: its only equations are the
joint-torque equations, and those always contain friction, so there is no friction-free
anchor to identify the inertial parameters from. That friction-free anchor is exactly what
the floating-base base wrench provides. This is why the two-step path is gated on
`floatingBase` and fixed-base robots identify friction simultaneously.

The trade-off of the simultaneous path is robustness near velocity reversals. The per-joint
velocity dead zone (which drops the near-zero-velocity samples where the friction model is
wrong — stiction, backlash) is *not* applied inside the simultaneous SDP, because dropping a
sample removes the whole multi-joint row block and the inertial columns couple the joints,
so a clean per-joint exclusion is not available there. The reversal-corrupted samples
therefore also bias the inertial estimate; the friction Tikhonov regularization
(`frictionRegularization`) only damps the `Fc`/`Fv` correlation, not this bias.

**Post-hoc residual refit on fixed base.** `postIdentifyFriction` is therefore allowed on a
fixed base too (not only on a floating base), provided friction was first identified
simultaneously so the inertial parameters are not friction-biased. After the SDP, the
per-joint residual refit (with its dead zone, `Fv` prior and smoothed sign) re-estimates
`Fc`/`Fv`/offset and overwrites the jointly-fit friction slots in the standard-parameter
vector; the inertial parameters are left unchanged. This is approximate — the inertials it
conditions on were themselves fit with the corrupted samples, so it improves the friction
parameters rather than removing the inertial bias — but it recovers the robust friction
handling for the fixed-base case and the result stays as standard `Fc`/`Fv` that the URDF
can hold (unlike a Stribeck term). The `Fv` prior matters more here than in the
floating-base two-step: after the dead zone removes the low-velocity samples, a joint that
moves at near-constant speed has collinear `Fc` and `Fv·v` columns, so too weak a prior
lets `Fv` collapse to zero with `Fc` absorbing the slope — use a prior strong enough to keep
`Fv` physical. A fuller alternative that would also de-bias the inertials is a per-sample
weighting inside the SDP (down-weighting each row by the slowest relevant joint's velocity,
rather than a per-joint mask), or iterating the refit; neither is implemented — the
per-sample weighting is tracked as a follow-up in `documentation/std_parameter_roadmap.md`
(section c).

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
