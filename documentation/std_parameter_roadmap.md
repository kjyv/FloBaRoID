# Toward better standard-parameter identification of floating-base robots

Status roadmap and design notes. The recurring question: simulated data from a "measured"
model *should* let us identify parameters close to the real ones starting from a CAD-only
model, yet the **standard** parameters often move *away* from the real values.

Legend: ✅ done · ⏸ deferred · ◻ open

## The core issue: base vs. standard parameters

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

## The structural constraint (decisive for what can help)

For a fixed kinematic structure and sensor set, **which** standard parameters are
identifiable is *structural* — it is **not** improvable by excitation. Excitation diversity
and multi-trajectory pooling make the identifiable (base) params more *accurate*; they do
**not** recover more standard parameters. Therefore standard-parameter quality has exactly
two levers:

  (a) **better priors** in the null space, or
  (b) **adding equations** to shrink the null space.

Everything below is grouped accordingly.

## (a) Better priors — null-space std accuracy

- ✅ **Observability-weighted CAD regularization** (`cadRegularizationMode: observability`):
  per-parameter pull toward CAD weighted by how poorly the data determines each parameter
  (ridge-regularized normal matrix of the reduced system). Well-determined params stay free,
  weakly-determined ones stay near CAD. Robust across independent CAD priors; reduces
  std-param drift and improves base-param distance without changing the torque fit or
  held-out validation. Default `uniform` preserved; enabled for walkman.
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

## (b) Add equations — recover *more* standard params (quantified: low ROI here)

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

## (c) Reduce unmodeled-effect bias (matters on real hardware; indirectly helps std)

On real data, unmodeled effects bias even the identifiable params, which propagates into the
std decomposition. Priority roughly by how pervasively each acts:
- ◻ **Cable forces, thermal/load-dependent drift, backlash** — act across the whole motion,
  so they bias identification pervasively. Model or filter them. Dominant real-robot error.
- ◻ **Stribeck/stiction friction** — lower priority. Real geared joints do have a
  stiction→kinetic drop near zero velocity that Coulomb+viscous cannot represent, so real
  data does *not* "handle it automatically". But its impact on inertial/std identification
  is small because good excitation keeps joints moving and the velocity dead zone already
  excludes the near-zero-velocity samples where Stribeck lives. Worth modeling only if an
  accurate low-speed friction model is needed for control, or if real-data residuals near
  zero crossings prove harmful. (The disabled implementation also needs a redesign: its
  velocity constant `vs` is a nonlinear parameter, not linearly identifiable.)
- For algorithm development, validate first with only friction enabled (no
  backlash/cable/thermal) to isolate identification performance from model mismatch.

## (d) Method / workflow

- ◻ **Staged identification** (gravity params from slow/static poses → then dynamics):
  partly supported already (`identifyGravityParamsOnly` + the `*_static` configs); the open
  part is chaining them as a standard workflow. Improves conditioning and separates
  gravity-dependent params.
- ◻ **Essential-parameter identification** (`useEssentialParams`, Pham/Gautier): reduces to a
  well-determined subset rather than recovering more; evaluate if a robust reduced set is the
  goal.
- ◻ **Nonlinear constrained path** (`constrainUsingNL`): exists, untested recently; may
  navigate the feasible set differently than the SDP.
- ◻ **Reproducibility**: a configurable Optuna seed (+ single-job mode) for reproducing a
  *specific* optimization run; keep the seed random for diverse multi-trajectory sets.

## Evaluation discipline (adopt as standard)

- Report **base-parameter distance** as the primary metric; standard-parameter distance only
  with the null-space caveat in mind.
- Report torque NRMS on a **held-out** trajectory (`--validation`), not only on the training
  trajectory — the training residual understates the true error.
- When a real model is available, report friction and parameter errors against it directly.

## Bottom line for standard-parameter fidelity

The identifiability analysis (section b) shows the data side is close to exhausted: ~207
std-param directions are unrecoverable from torque + base-wrench sensing, and practical
extra instrumentation reaches only ~10% of that. So **priors are essentially the only
practical lever** for getting standard parameters closer to real:
- ✅ observability-weighted CAD regularization (done) — keeps the decomposition sensible
  where the data is weak;
- ⏸ per-link soft-trust priors — the remaining lever, valuable only when CAD quality
  genuinely varies between links (real-robot knowledge).

Adding sensors/payloads/contacts is low ROI for std params on a high-DOF floating base, and
known payloads don't change the rank at all. The base parameters and torque model are
already strong and data-determined; chasing more std-param recovery via more equations is
not worthwhile here. If individual link parameters are genuinely needed, the realistic
options are trusting/pinning the links whose CAD is reliable, or instrumenting essentially
every joint — not a few extremity sensors.

## Pointers
Friction/excitation/collision design rationale (the *why* behind current settings) is in
`documentation/friction_velocity_filter_analysis.md`. This file is the higher-level roadmap
toward standard-parameter quality.
