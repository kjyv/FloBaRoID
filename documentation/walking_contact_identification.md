# Identifying from walking data via foot-contact wrenches — findings

Can recorded walking data with foot force/torque (F/T) sensors be used for floating-base
inertial identification? The contact pipeline supports it, but the data is poor for
identification. Numbers below are from specific runs (real WALK-MAN walking logs) and are
findings, not durable documentation.

## Setup

Real WALK-MAN walking logs in `data/WALKMAN/` (`measurements_mpc_01.npz` train,
`measurements_mpc_02.npz` validate): 200 Hz, ~69 s / 13770 samples, 29 DOF, both-feet F/T
(`r_leg_ft`, `l_leg_ft`), full base state. CAD start `walkman_apriori.urdf`. There is **no
ground-truth parameter set** (real hardware), so accuracy is judged by held-out fit and
physical plausibility, not by distance to any model.

The floating base is closed through the measured foot wrenches:
`floatingBaseAttachment: free`, `addContacts: 1`. Each contact frame's 6D wrench is
projected through its free-floating Jacobian into the base+joint equations
(`model.py`), moved to the known side of the identification system.

## What works

The pipeline runs end-to-end on the real walking data: the free-floating base is closed by
the two foot wrenches, and the (geometric-prior) SDP solves. So **using contact forces /
walking data is an existing capability**, not a missing one.

## Why the data is poor for identification

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

## Caveats / not yet ruled out

The foot-F/T **frame and sign conventions** were not independently audited. A systematic
contact-wrench sign/frame error would inflate the joint-torque residual, so the absolute
122 % may improve with a convention check. However, the conditioning (~4.3×10⁵) and the
excitation profile (a third of joints static) are independent of conventions and cap how
useful this data can be regardless.

## Recommendation

- Walking data is **not** a substitute for a dedicated excitation trajectory and should not
  be used standalone for inertial identification.
- Realistic uses: **validation in the operating regime**, or pooled as a supplementary file
  alongside an excitation run (multi-file inverse-noise weighting), where the excitation run
  carries the inertial information and the walking run anchors the contact/leg behavior.
- If walking identification is pursued, first audit the F/T frame/sign conventions, then
  restrict to the leg subchain and the parameters the gait actually excites.
