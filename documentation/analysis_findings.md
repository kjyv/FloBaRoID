# Analysis findings

Empirical results from specific identification runs, with concrete numbers. These are
findings from particular experiments, not durable documentation — the design rationale
lives in `design_notes.md`.

## Geometric (log-det divergence) CAD prior

Analysis of the `cadRegularizationMode: geometric` prior (Lee et al. 2020) against the
existing `uniform` and `observability` Euclidean CAD pulls. Context and the design
rationale are in `design_notes.md` (standard-parameter roadmap, section a). Numbers below
are from specific runs and are findings, not durable documentation.

### What the prior does

Instead of pulling standard parameters toward CAD with a Euclidean (weighted L2) residual,
it adds the log-det Bregman divergence between each link's 4×4 pseudo-inertia and its CAD
value to the SDP objective:

    D(P ‖ P0) = tr(P0^-1 P) − logdet(P) + logdet(P0) − 4

It is coordinate/frame invariant, convex, zero iff `P = P0`, and diverges as a link
approaches degeneracy — so it repels the zero-mass / flat-inertia solutions the Euclidean
pull leaves unpenalized.

### Numerical scaling was required to reach the floating-base robot

The naïve implementation solved on small/well-conditioned systems (threeLink, KUKA) but
**failed on the 29-DOF / 480-param walkman**: CLARABEL returned `solver_error` and SCS
reported a (false) unbounded ray, falling back to the a priori model. The objective is
provably bounded below (the divergence is ≥ 0), so this was numerical, not structural. Two
fixes made it solve:

- **Whitening.** Evaluate the divergence on `Q = P0^{-1/2} P P0^{-1/2}` (≈ I at the a
  priori) and use `tr(Q) − logdet(Q) − 4`. Mathematically identical (`logdet(P0)` cancels)
  but O(1)-scaled regardless of each link's CAD conditioning. The raw `tr(P0^-1 P)` form is
  catastrophic for small links whose CAD pseudo-inertia spans many orders of magnitude:
  `P0^-1` is then huge and the conic solver cannot equilibrate the resulting dynamic range.
- **Residual normalization.** The torque-residual block of the objective is ~2.1e7 on
  walkman while the divergence is O(1); normalizing the residual to O(1) (fit-preserving)
  makes the two terms commensurate.
- Pinned links (`dontChangeLinks`) are also skipped — they were adding degenerate
  (variable-fixed) `log_det` blocks for no benefit (42 → 30 links on walkman).

### Walkman (29 DOF, suspended), CAD = `walkman_apriori`, real = `walkman_measured`

Identification only (no trajectory optimization), `walkman_measured.urdf.measurements.npz`,
default `geometricRegularizationFactor = 1.0`. Distances are L2 to the real model over the
identified parameters.

| mode           | std-param distance | base-param distance |
|----------------|--------------------|---------------------|
| uniform        | 4.60               | 4.80                |
| observability  | 3.41               | 2.82                |
| geometric      | 3.30               | **2.25**            |
| geometric+obs  | 3.31               | 2.26                |

- The **geometric prior gives the best base-parameter distance** — the meaningful,
  data-driven metric (the roadmap's primary metric) — beating both Euclidean modes. It also
  gives the best standard-parameter distance.
- **Observability-weighting the geometric prior is a no-op here** (2.25 vs 2.26). On a
  uniformly-perturbed synthetic CAD every link is equally (un)trustworthy, so the per-link
  observability scaling has nothing to discriminate; whitening already makes each link's
  divergence scale-invariant. It is expected to matter only when CAD quality genuinely
  varies between links (a real-robot scenario), the same regime where per-link soft-trust
  priors pay off.

### KUKA LWR4 (7 DOF, fixed base), real hardware data

No ground-truth model; metric is held-out validation NRMS (train on `measurements_1`,
validate on `measurements_2`).

| mode           | held-out validation NRMS |
|----------------|--------------------------|
| uniform        | 0.211 %                  |
| observability  | 0.176 %                  |
| geometric      | 0.181 %                  |
| geometric+obs  | 0.183 %                  |

On a well-conditioned real robot the regularization choice barely affects torque
generalization; the geometric prior matches observability and slightly beats uniform, i.e.
it does not hurt the fit. This is consistent with the roadmap: the prior changes the
null-space decomposition, not the data-determined torque model.

### Takeaways

- The geometric prior is the better default *metric* for the null-space CAD pull: it
  improves base-parameter distance on the floating-base robot and does not hurt the fit on
  the well-conditioned arm.
- Whitening is mandatory for it to scale; the raw divergence form does not.
- Observability-weighting it adds nothing on uniformly-perturbed synthetic CAD; revisit it
  only with heterogeneous per-link CAD trust.

## Identifying from walking data via foot-contact wrenches

Can recorded walking data with foot force/torque (F/T) sensors be used for floating-base
inertial identification? The contact pipeline supports it, but the data is poor for
identification. Numbers below are from specific runs (real WALK-MAN walking logs) and are
findings, not durable documentation.

### Setup

Real WALK-MAN walking logs in `data/WALKMAN/` (`measurements_mpc_01.npz` train,
`measurements_mpc_02.npz` validate): 200 Hz, ~69 s / 13770 samples, 29 DOF, both-feet F/T
(`r_leg_ft`, `l_leg_ft`), full base state. CAD start `walkman_apriori.urdf`. There is **no
ground-truth parameter set** (real hardware), so accuracy is judged by held-out fit and
physical plausibility, not by distance to any model.

The floating base is closed through the measured foot wrenches:
`floatingBaseAttachment: free`, `addContacts: 1`. Each contact frame's 6D wrench is
projected through its free-floating Jacobian into the base+joint equations
(`model.py`), moved to the known side of the identification system.

### What works

The pipeline runs end-to-end on the real walking data: the free-floating base is closed by
the two foot wrenches, and the (geometric-prior) SDP solves. So **using contact forces /
walking data is an existing capability**, not a missing one.

### Why the data is poor for identification

- **Weak, uneven excitation.** Median per-joint peak velocity ~0.39 rad/s; joints move
  (>0.1 rad/s) a median of only ~4 % of the time; **9 of 29 joints barely move** (peak
  < 0.2 rad/s — arms/torso held during the gait). Base-parameter conditioning is **~4.3×10⁵**.
- **Held-out joint-torque fit is worse than predicting zero.** Joint-torque relative error
  on the held-out trajectory is **~122 %** (per-joint: legs ~161 %, arms ~139 %, trunk
  ~92 %). The rigid-body + friction model does not explain walking joint torques: they are
  dominated by quasi-static gravity holding and contact load transfer, not by the inertial
  dynamics the regressor identifies.
- **Physical red flags.** Identified total mass drifts to ~138.6 kg from the CAD ~128.0 kg
  (+8 %), and a link is driven to near-zero mass — the SDP is fitting bias/noise, not signal.

### Metric caveat (important)
The headline validation numbers the identifier prints are flattering and must not be taken
at face value here:
- For a floating base whose validation file has only joint torques, the 6 base-wrench rows
  are filled with the *estimated* base wrench, so they compare to themselves (zero error)
  and inflate the denominator. The reported "relative validation error" (~14 %) and
  limit-normalized "NRMS" (~7 %) are therefore optimistic.
- The honest metric is the **joint-torque-only relative error (~122 %)** above.

### Caveats / not yet ruled out

The foot-F/T **frame and sign conventions** were not independently audited. A systematic
contact-wrench sign/frame error would inflate the joint-torque residual, so the absolute
122 % may improve with a convention check. However, the conditioning (~4.3×10⁵) and the
excitation profile (a third of joints static) are independent of conventions and cap how
useful this data can be regardless.

### Recommendation

- Walking data is **not** a substitute for a dedicated excitation trajectory and should not
  be used standalone for inertial identification.
- Realistic uses: **validation in the operating regime**, or pooled as a supplementary file
  alongside an excitation run (multi-file inverse-noise weighting), where the excitation run
  carries the inertial information and the walking run anchors the contact/leg behavior.
- If walking identification is pursued, first audit the F/T frame/sign conventions, then
  restrict to the leg subchain and the parameters the gait actually excites.

## KUKA LWR4+ simulation validation (real vs simulated measurements)

Comparison of real measurements (data/KUKA/HW/measurements_1.npz, recorded 2017 at IIT Genova)
against ideal inverse dynamics computed from the same recorded positions using the URDF model
with friction (Fv*vel + Fc*sign(vel)).

### Torque Residuals (real - simulated)

| Joint | Residual RMS (Nm) | Real RMS (Nm) | Relative | Notes |
|-------|-------------------|---------------|----------|-------|
| 0 | 0.36 | 0.99 | 36% | Large bias (+0.85 Nm) |
| 1 | 1.55 | 25.20 | 6% | Good match, small bias |
| 2 | 0.49 | 3.05 | 16% | Reasonable |
| 3 | 0.98 | 10.67 | 9% | Good match |
| 4 | 0.63 | 0.81 | 77% | Large relative error, bias -0.72 Nm |
| 5 | 0.49 | 0.28 | 173% | Model overpredicts (Fv too high?) |
| 6 | 0.27 | 0.25 | 105% | Model overpredicts |

### Observations

1. **Proximal joints (1-3)**: rigid body model + URDF friction fits well (6-16% residual).
   The URDF inertial parameters are a good match for the IIT robot.

2. **Distal joints (4-6)**: large relative residuals. The small torque signals mean that
   unmodeled effects (stiction, cable forces, sensor bias) dominate over the rigid body torques.
   Joint 5 model overpredicts — likely the URDF Fv=0.3 is too high for this joint.

3. **Mean biases (0.1-0.9 Nm per joint)**: constant offsets that Coulomb friction identification
   should capture. The URDF Fc values (0.05-0.8 Nm) are in the right ballpark.

4. **Real data has zero high-frequency content** (< 0.0001 Nm above 10 Hz): the recorded data
   was heavily filtered by FloBaRoID's preprocessing (filterLowPass1: [8.0, 5] = 8 Hz 5th-order
   Butterworth). The simulated raw sensor noise (0.5% at 600 Hz) is realistic for the raw signal
   but gets removed by preprocessing — this is correct behavior.

### Implications for Simulator

- The simulator's noise model is appropriate for raw sensor data. The identification pipeline's
  own filtering will clean it up, matching what happens with real data.
- The URDF friction values (Fv, Fc) are reasonable starting points but will differ from the
  real robot (wear, individual unit variation). This is expected — identification is meant
  to find the actual values.
- The main unmodeled effects visible in the residuals are: friction model mismatch on distal
  joints, sensor bias/offset, and possibly cable routing forces.
