# FloBaRoID identification design notes

The durable design rationale behind the identification pipeline — the *why* of the
current standard-parameter, friction, excitation and collision-checking settings.
Concrete benchmark numbers from individual runs are intentionally kept out of this
file; those live in `analysis_findings.md`. Two parts follow: the standard-parameter
roadmap, then the friction/excitation/collision design notes.

## Toward better standard-parameter identification of floating-base robots

Status roadmap and design notes. The recurring question: simulated data from a "measured"
model *should* let us identify parameters close to the real ones starting from a CAD-only
model, yet the **standard** parameters often move *away* from the real values.

Legend: ✅ done · ⏸ deferred · ◻ open

### The core issue: base vs. standard parameters

From joint-torque (and base-wrench) data you can only identify the **base parameters** —
independent linear combinations of the standard parameters. Many physically-consistent
standard-parameter sets produce identical base parameters and identical torques. The
identification picks one consistent set; "distance to real" over *all* standard parameters
is then partly an arbitrary choice inside the data null space.

Consequences confirmed by experiment:
- The **base-parameter distance is the meaningful data-driven metric**, and it improves
  dramatically and CAD-independently. The torque model (which depends only on base params)
  generalizes — verified on held-out trajectories.
- **Standard-parameter accuracy is prior-dominated.** From torque data alone, identification
  cannot make std params *closer* to real than the prior in the null space — across several
  independent CAD priors, identifying std params moved them slightly *further* from real
  than the CAD started; the regularization only minimizes that drift.

### The structural constraint (decisive for what can help)

For a fixed kinematic structure and sensor set, **which** standard parameters are
identifiable is *structural* — it is **not** improvable by excitation. Excitation diversity
and multi-trajectory pooling make the identifiable (base) params more *accurate*; they do
**not** recover more standard parameters. Therefore standard-parameter quality has exactly
two levers:

  (a) **better priors** in the null space, or
  (b) **adding equations** to shrink the null space.

Everything below is grouped accordingly.

### (a) Better priors — null-space std accuracy

- ✅ **Observability-weighted CAD regularization** (`cadRegularizationMode: observability`):
  per-parameter pull toward CAD weighted by how poorly the data determines each parameter
  (ridge-regularized normal matrix of the reduced system). Well-determined params stay free,
  weakly-determined ones stay near CAD. Robust across independent CAD priors; reduces
  std-param drift and improves base-param distance. On the well-conditioned walkman it
  barely changes the torque fit or held-out validation; on small/poorly-conditioned
  systems the fit can shift more (it trades a little fit for CAD-closeness), so it is
  opt-in: default `uniform` preserved, enabled for walkman.
- ✅ **Geometric (log-det divergence) CAD prior** (`cadRegularizationMode: geometric`):
  pulls each link's 4×4 pseudo-inertia toward its CAD value with the log-det Bregman
  divergence `D(P‖P₀) = tr(P₀⁻¹P) − logdet(P) + logdet(P₀) − 4` (Lee et al. 2020), added
  to the SDP objective rather than as Euclidean residual rows. Unlike the
  uniform/observability Euclidean pull, the divergence is coordinate/frame invariant
  (it does not mix kg, kg·m, kg·m² in one L2 norm), convex in the inertial parameters,
  zero iff `P = P₀`, and diverges as a link approaches degeneracy — so it actively repels
  the zero-mass / flat-inertia null-space solutions the Euclidean pull leaves unpenalized
  (the robustness failure mode SDP identification is known for). Links with degenerate
  (non-PD) CAD pseudo-inertia, pinned (`dontChangeLinks`), or in gravity-only mode are
  skipped. Strength is tuned via `geometricRegularizationFactor`.
  **Numerically, it must be whitened to scale**: the divergence is evaluated on
  `Q = P₀^{-1/2} P P₀^{-1/2}` (≈ I at the a priori) as `tr(Q) − logdet(Q) − 4`
  — mathematically identical but O(1)-scaled regardless of each link's CAD conditioning.
  The raw `tr(P₀⁻¹P) − logdet(P)` form makes the conic solver fail on a large floating-base
  robot (small links have CAD pseudo-inertia spanning many orders of magnitude); the
  torque-residual block is also normalized to O(1) so it is commensurate with the
  divergence. With `geometricObservabilityWeighting: true` it composes with the
  observability lever (each link's divergence scaled by the mean observability weight of
  its 10 params) — but on uniformly-perturbed synthetic CAD this adds nothing (every link
  is equally untrustworthy), so it is expected to matter only with heterogeneous per-link
  CAD trust, the same regime as soft-trust priors. It does not recover *more* std params
  (structural limit, section b) and cannot beat the prior-domination ceiling — it changes
  *which* consistent null-space point is chosen toward a more physically plausible one.
  Measured to give the best base-parameter distance among the prior modes on the
  floating-base robot while not hurting the fit on the well-conditioned arm; see the
  geometric-prior findings in `analysis_findings.md`. Opt-in; default `uniform` preserved.
- ⏸ **Per-link soft-trust priors**: graded per-link trust in CAD (generalizing
  `dontChangeLinks` from hard pin to a finite weight). The only lever to preserve std
  accuracy on links whose CAD is trustworthy. Trust is prior knowledge the data cannot
  supply in the directions that matter, so it needs sparse manual per-link input (a few
  trusted/untrusted links; the rest a uniform default) — it cannot be fully automatic.
  Only pays off when CAD quality genuinely varies between links (a real-robot scenario;
  irrelevant for the current uniformly-perturbed synthetic CAD).
- ◻ **Geometry/density prior from meshes**: derive a plausible mass/inertia from mesh
  volume × assumed material density as a complementary prior. Semi-automatic from the meshes
  already loaded, but assumes near-uniform density (wrong for motor/battery-heavy links) —
  a sanity prior, not a trust oracle.

### (b) Add equations — recover *more* standard params (quantified: low ROI here)

A structural identifiability analysis of the 29-DOF suspended walkman (random-regressor
SVD over joint torques + base wrench) bounds how much this can help:
- Of ~420 real inertial parameters, only ~70 are individually identifiable; ~213
  base-parameter directions are determined, leaving a **~207-direction null space** — the
  recoverable gap. The worst-lumped links are the proximal/torso group (Waist, hip motors,
  torso, shoulders) and the terminal links (hands, head, wrists).
- Each added **6-axis F/T** sensor recovers only **~3** of those 207 directions, because the
  floating-base wrench (already measured) plus joint torques already capture most of what an
  extremity F/T would add — it only isolates the distal-from-proximal split. Roughly
  additive for disjoint placements: both wrists +6, both ankles +6, wrists+ankles +12,
  +knees+elbows ~+24. Closing the whole gap needs essentially an F/T at every joint.
- **Known payloads do not change the rank at all** — they only improve conditioning and let
  you identify the payload itself.

So for a high-DOF floating-base humanoid, adding equations is **low ROI for standard-parameter
recovery**: realistic instrumentation reaches ~10% of the gap. Worth it only if the sensors
are wanted anyway (e.g. foot F/T for contact/control, giving a small std-param bonus).
- ◻ **Extra sensors (link IMUs, force/torque)** — only meaningful if instrumenting many
  joints; a few extremity sensors barely move std-param recovery (see numbers above).
- ◻ **Known payloads / varied loading** — improves conditioning and identifies the payload,
  but does not break the structural null space.

### (c) Reduce unmodeled-effect bias (matters on real hardware; indirectly helps std)

On real data, unmodeled effects bias even the identifiable params, which propagates into the
std decomposition. Priority roughly by how pervasively each acts:
- ◻ **Cable forces, thermal/load-dependent drift, backlash** — act across the whole motion,
  so they bias identification pervasively. Model or filter them. Dominant real-robot error.
- ◻ **Stribeck/stiction friction** — lower priority. Real geared joints do have a
  stiction→kinetic drop near zero velocity that Coulomb+viscous cannot represent, so real
  data does *not* "handle it automatically". Its impact on inertial/std identification is
  small because good excitation keeps joints moving. The near-zero-velocity samples where
  Stribeck lives are excluded by the velocity dead zone *only in the floating-base two-step
  friction path*; the fixed-base simultaneous path (friction columns in the SDP regressor)
  does not apply a dead zone, so there those samples still bias the joint estimate (see the
  friction design notes for why the two paths differ).
  - The Stribeck term *is* usable as-is: with `vs` (`stribeckVelocity`) taken as a fixed
    constant from config, `Fs * exp(-|v|/vs) * sign(v)` is a linear regressor column and
    `Fs` is identified linearly per joint. Only *auto-identifying* `vs` would be nonlinear —
    that is the part that would need a redesign, not the term itself.
  - Quantified on real KUKA hardware data (simultaneous mode): the near-reversal residual
    (`|v|` < 0.1 rad/s) is ~1.8× the moving-sample residual, confirming the spikes are real
    and localized. Enabling Stribeck with a physical `vs ≈ 0.05` reduces the near-reversal
    residual by only ~7% and improves held-out validation marginally; the bulk of the
    reversal residual persists. It is therefore **not** a cure: most of the reversal residual
    is backlash/hysteresis, which a static rigid-body + friction regressor cannot represent.
    A caveat against over-tuning `vs`: at larger `vs` the `exp` term also lowers the
    *moving*-sample residual, i.e. it starts acting as a general extra fitting basis rather
    than physical stiction. `Fs` also cannot be written back to URDF (no field for it).
  - Worth enabling only if an accurate low-speed friction model is needed for control. The
    higher-leverage action for inertial/std quality is to keep the near-reversal samples from
    biasing the fit (extend the dead zone / sample weighting to the simultaneous path), not
    to fit the spike.
- ◻ **Per-sample (reversal) weighting in the simultaneous SDP** — follow-up to the
  fixed-base post-hoc friction refit, which de-biases only the friction and leaves the
  inertials as they were fit *with* the corrupted near-reversal samples. The idea is to
  weight each sample row of the standard regressor by a function of velocity, down-weighting
  the near-zero-velocity rows where stiction/backlash make the model wrong, so those samples
  bias *both* the inertial and friction estimates less, inside one weighted SDP. A row is a
  multi-joint torque equation, so the per-row weight has to aggregate across joints (e.g. by
  the slowest relevant joint's `|v|`) — which is exactly why a clean per-joint dead-zone mask
  is not available in this path. Unlike the post-hoc refit this also de-biases the inertials,
  and it could be iterated with the refit. Caveats: the weighting must be carried consistently
  through the base-parameter projection and the physical-consistency objective (weighted least
  squares on the regressor before the QR/SVD base reduction); down-weighting reduces the
  effective data and can worsen conditioning, so gate on held-out validation. It does **not**
  remove the need for the `Fv` prior — that is a velocity-diversity (identifiability) limit,
  not a bad-sample one.
- For algorithm development, validate first with only friction enabled (no
  backlash/cable/thermal) to isolate identification performance from model mismatch.

### (d) Method / workflow

- ◻ **Staged identification** (gravity params from slow/static poses → then dynamics):
  partly supported already (`identifyGravityParamsOnly` + the `*_static` configs); the open
  part is chaining them as a standard workflow. Improves conditioning and separates
  gravity-dependent params.
- ◻ **Essential-parameter identification** (`useEssentialParams`, Pham/Gautier): reduces to a
  well-determined subset rather than recovering more; evaluate if a robust reduced set is the
  goal.
- ◻ **Reproducibility**: a configurable Optuna seed (+ single-job mode) for reproducing a
  *specific* optimization run; keep the seed random for diverse multi-trajectory sets.

### Evaluation discipline (adopt as standard)

- Report **base-parameter distance** as the primary metric; standard-parameter distance only
  with the null-space caveat in mind.
- Report torque NRMS on a **held-out** trajectory (`--validation`), not only on the training
  trajectory — the training residual understates the true error.
- When a real model is available, report friction and parameter errors against it directly.

### Bottom line for standard-parameter fidelity

The identifiability analysis (section b) shows the data side is close to exhausted: ~207
std-param directions are unrecoverable from torque + base-wrench sensing, and practical
extra instrumentation reaches only ~10% of that. So **priors are essentially the only
practical lever** for getting standard parameters closer to real:
- ✅ observability-weighted CAD regularization (done) — keeps the decomposition sensible
  where the data is weak;
- ✅ geometric (log-det divergence) CAD prior (done) — a coordinate-invariant prior metric
  that repels degenerate (zero-mass) null-space solutions and composes with the
  observability weighting;
- ⏸ per-link soft-trust priors — the remaining lever, valuable only when CAD quality
  genuinely varies between links (real-robot knowledge).

Adding sensors/payloads/contacts is low ROI for std params on a high-DOF floating base, and
known payloads don't change the rank at all. The base parameters and torque model are
already strong and data-determined; chasing more std-param recovery via more equations is
not worthwhile here. If individual link parameters are genuinely needed, the realistic
options are trusting/pinning the links whose CAD is reliable, or instrumenting essentially
every joint — not a few extremity sensors.

## Friction identification and floating-base excitation

Rationale for the friction-identification, excitation and collision-checking choices in
the floating-base pipeline.

### Two-step friction identification

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

### Two-step vs. simultaneous friction identification

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
per-sample weighting is tracked as a follow-up in the standard-parameter roadmap above
(section c).

### Simulator measurement semantics

The simulator stores the noisy measurements under the `*_raw` keys (positions, velocities,
torques), matching the meaning those keys have for real data produced by `excite.py` /
`tools/csv2npz.py`. The clean, pre-noise reference signals are available separately as
`target_*`. Friction-sign filtering and any other "raw measurement" processing therefore
behave identically on simulated and real data.

### Excitation (trajectory optimization)

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

### Multi-trajectory identification

Identifying from several measurement files (passed as repeated `--measurements`) has
always been supported; the files are concatenated. `useTrajectoryWeighting` weights each
file's base-wrench equations by the inverse of its residual noise (estimated from an OLS
pre-pass) instead of stacking all samples with equal weight, so a noisier session does not
dominate. This matters mainly for real data, where per-session noise genuinely differs;
for equal-noise simulated data the weights are near uniform.

Note that pooling helps only when an added trajectory brings more information than
trajectory-specific bias (from its own unmodeled effects). A trajectory that is poor on
its own can dilute a combined fit; gate on held-out validation before pooling.

### Collision checking for the suspended floating base

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

