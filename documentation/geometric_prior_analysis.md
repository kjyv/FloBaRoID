# Geometric (log-det divergence) CAD prior — findings

Analysis of the `cadRegularizationMode: geometric` prior (Lee et al. 2020) against the
existing `uniform` and `observability` Euclidean CAD pulls. Context and the design
rationale are in `std_parameter_roadmap.md` section (a). Numbers below are from specific
runs and are findings, not durable documentation.

## What the prior does

Instead of pulling standard parameters toward CAD with a Euclidean (weighted L2) residual,
it adds the log-det Bregman divergence between each link's 4×4 pseudo-inertia and its CAD
value to the SDP objective:

    D(P ‖ P0) = tr(P0^-1 P) − logdet(P) + logdet(P0) − 4

It is coordinate/frame invariant, convex, zero iff `P = P0`, and diverges as a link
approaches degeneracy — so it repels the zero-mass / flat-inertia solutions the Euclidean
pull leaves unpenalized.

## Numerical scaling was required to reach the floating-base robot

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

## Walkman (29 DOF, suspended), CAD = `walkman_apriori`, real = `walkman_measured`

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

## KUKA LWR4 (7 DOF, fixed base), real hardware data

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

## Takeaways

- The geometric prior is the better default *metric* for the null-space CAD pull: it
  improves base-parameter distance on the floating-base robot and does not hurt the fit on
  the well-conditioned arm.
- Whitening is mandatory for it to scale; the raw divergence form does not.
- Observability-weighting it adds nothing on uniformly-perturbed synthetic CAD; revisit it
  only with heterogeneous per-link CAD trust.
