# Literature Review: Friction Identification in Robot Dynamic Parameter Identification

**Date:** 2026-03-31

---

## 1. Do Authors Identify Friction Alongside Inertial Parameters, or Separately?

### The dominant approach: simultaneous identification

The overwhelming majority of the robot identification literature identifies friction parameters
**simultaneously** with inertial parameters. The standard framework, established by Gautier & Khalil
(1990) and refined by Swevers et al. (2007), formulates the inverse dynamic identification model
(IDIM) as a system linear in all parameters:

    tau = Y(q, dq, ddq) * theta

where theta contains both inertial parameters (masses, first moments, inertia tensor components,
rotor inertias) **and** friction parameters (Fc, Fv, and optionally offset) for each joint. The
regressor matrix Y has columns for all parameters, and ordinary or weighted least squares (OLS/WLS)
is applied to solve for everything at once.

**Key references for simultaneous identification:**
- Gautier & Khalil (1990) - foundational IDIM framework
- Swevers, Verdonck & De Schutter (2007) - IEEE Control Systems tutorial, simultaneous with periodic excitation
- Khalil & Dombre (2002) - textbook treatment, simultaneous
- Janot, Vandanjon & Gautier (2014) - DIDIM/IV methods, simultaneous
- Traversaro, Brossette, Nori et al. (2014) - iCub, simultaneous (inertial + friction + motor via PLS)
- Sousa & Cortesao (2014) - SDP/LMI, simultaneous with physical consistency constraints
- Lee, Wensing & Park (2020) - geometric SDP, simultaneous

### Two-step / sequential approaches (the minority)

Several papers have explored identifying friction first, then inertials, or vice versa:

- **Grotjahn, Daemi & Heimann (2001)** - "Friction and rigid body identification of robot dynamics."
  Uses point-to-point motions to **separate friction and rigid-body dynamics** into different parameter
  groups. Friction is identified from low-speed data, rigid body from high-speed data. Avoids
  systematic errors from time-varying friction. The base parameters are split into three groups,
  identified sequentially.

- **Elhami & Brookfield (1996)** - "Sequential identification of Coulomb and viscous friction in
  robot drives." Identifies Coulomb first using sign information, then viscous. Uses nonlinear
  filtering (Detchmendy-Sridhar). Explicitly models **asymmetric** friction (different Fc/Fv for
  positive and negative directions).

- **Park, Lee et al. (2022)** - Two-step method for Indy7 collaborative robot. **Step 1:** identify
  comprehensive friction model (LuGre + temperature-dependent viscous) from constant-velocity and
  sinusoidal torque experiments. **Step 2:** subtract identified friction from measured torques, then
  identify inertial parameters from the residual using constrained WLS. Reports **up to 70% improvement**
  in tracking accuracy vs. single-step methods. This is the most convincing argument for two-step.

- **Ogawa, Venture & Ott (2014)** - Humanoid (TORO/HRP-2). Uses both joint-torque-based and
  base-link approaches. The base-link approach (Ayusawa method) inherently **avoids friction** since
  the unactuated base dynamics have no joint torques and therefore no friction.

### Humanoid-specific approaches

For humanoid robots, friction handling diverges:

- **Ayusawa, Venture & Nakamura (2014)** - The base-link dynamics method for floating-base robots
  **completely avoids friction**. Since the method uses only the unactuated 6-DOF base-link dynamics
  (which have no joint torques), friction and actuator dynamics are simply not in the model. This is
  a major advantage but requires external force measurements or free-flying conditions.

- **Traversaro, Brossette, Nori et al. (2014)** - iCub arm identification. Identifies inertial,
  friction, and motor parameters **simultaneously** using PWM signals and F/T sensor data with
  Partial Least Squares (PLS) regression. Uses Coulomb + viscous friction model.

- **Ogawa, Venture & Ott (2014)** - TORO/HRP-2. When using joint torque measurements, friction is
  included in the standard IDIM. When using the base-link method, friction is avoided entirely.

**No published work was found on WALKMAN-specific identification.**

---

## 2. What Friction Models Are Used in Identification?

### Standard model (most common): Coulomb + viscous

    tau_f_j = Fc_j * sign(dq_j) + Fv_j * dq_j

This is the model used by the vast majority of identification papers. It is **linear in parameters**,
which is its key advantage -- it fits directly into the IDIM linear regressor framework.

**Used by:** Gautier & Khalil (1990), Swevers et al. (2007), Sousa & Cortesao (2014), Lee et al.
(2020), and most industrial robot identification work.

### Coulomb + viscous + offset (three terms per joint)

    tau_f_j = Fc_j * sign(dq_j) + Fv_j * dq_j + tau_off_j

The offset term (sometimes called "beta" or "tau_off") absorbs:
- Amplifier bias / current sensor offset
- Asymmetric Coulomb friction (if not modeled explicitly)
- Any constant torque bias in the measurement chain

This is the model recommended by Gautier in later work and used in many practical implementations.
The offset column in the regressor is simply a column of ones, making it linear. Sousa's SymPyBotics
framework supports configuring friction as combinations of 'Coulomb', 'viscous', and 'offset'.

**Used by:** Gautier (various), Hamon, Gautier, Garrec & Janot (2010), system identification for
constrained robots (2024, which uses Fc, Fv, Ia, beta per joint).

### Asymmetric Coulomb + viscous

    tau_f_j = Fc_j+ * max(sign(dq_j), 0) + Fc_j- * min(sign(dq_j), 0)
            + Fv_j+ * max(dq_j, 0) + Fv_j- * min(dq_j, 0)

Different friction coefficients for positive and negative velocity directions.
Elhami & Brookfield (1996) explicitly showed asymmetry is "essential" for DC servo motors.
Still linear in parameters (4 friction parameters per joint instead of 2-3).

### Load-dependent friction

    tau_f = f(dq, tau_load)

Hamon, Gautier, Garrec & Janot (2010) proposed a model where dry friction depends linearly on the
joint load torque. This makes the model **nonlinear** in the dynamic parameters (since tau_load
depends on inertial parameters). Requires iterative identification (DIDIM-like approach).

### Stribeck friction

Adds a velocity-dependent decrease of friction at low speeds (the "Stribeck curve"):

    tau_f = (Fc + (Fs - Fc) * exp(-(dq/vs)^2)) * sign(dq) + Fv * dq

**Not commonly used in standard identification** because it is nonlinear in vs (Stribeck velocity).
Requires nonlinear optimization. Some papers identify the Stribeck curve separately from
constant-velocity experiments.

### LuGre model

A dynamic friction model with internal state (bristle deflection):

    dz/dt = dq - sigma_0 * |dq| / g(dq) * z
    tau_f = sigma_0 * z + sigma_1 * dz/dt + sigma_2 * dq

Has 6 parameters including 2 dynamic parameters that are notoriously difficult to identify.
**Not used in standard IDIM-based identification** because it is nonlinear and requires internal
state estimation. Used only in specialized friction identification setups.

Park et al. (2022) used LuGre in their two-step method but only for Step 1 (friction-specific
identification with dedicated experiments), not in the IDIM regressor.

### Temperature-dependent friction

Park et al. (2022) showed that friction parameters vary significantly with temperature (19-51 C
range). They model temperature dependence as linear: sigma_j(T) = sigma_j0 + sigma_jT * T.
This is relevant for long experiments but is not standard practice.

### Summary table

| Model | Params/joint | Linear? | Used in IDIM? | Common? |
|-------|-------------|---------|---------------|---------|
| Coulomb + viscous | 2 | Yes | Yes | Very common |
| Coulomb + viscous + offset | 3 | Yes | Yes | Common |
| Asymmetric Coulomb + viscous | 4 | Yes | Yes | Occasional |
| Load-dependent | 3+ | No | Iterative | Rare |
| Stribeck | 4+ | No | No | Rare in ID |
| LuGre | 6 | No | No | Rare in ID |

---

## 3. Two-Phase Approaches

### Phase 1: friction, Phase 2: inertials

This is the approach taken by:

- **Park, Lee et al. (2022)** - Most detailed. Friction identified from constant-velocity experiments
  across temperature range, then subtracted from torques before inertial identification.

- **Grotjahn et al. (2001)** - Separates parameters into three groups using PTP motions. Avoids
  systematic errors from time-varying friction characteristics.

### Phase 1: inertials (avoiding friction), Phase 2: friction

This is implicitly the Ayusawa approach for humanoids:

- **Step 1:** Use base-link dynamics (no friction) to identify inertial parameters.
- **Step 2:** Could then identify friction from joint-level data with known inertials. However,
  Ayusawa et al. do not explicitly describe this second step in their papers.

### Alternating / iterative

- **DIDIM (Gautier & Poignet, 2008)** and **PC-DIDIM (Janot et al., 2021)** iterate between direct
  and inverse dynamic models. The direct dynamic simulation produces noise-free acceleration
  estimates, which are then used in the IDIM. This implicitly handles friction better because the
  simulation uses the current friction estimate. Not a two-phase approach per se, but an iterative
  refinement.

- **Load-dependent friction (Hamon et al., 2010)** requires iteration because friction depends on
  joint loads which depend on inertial parameters.

### Practical recommendation from literature

The two-step approach (friction first, then inertials) is **not** mainstream but has shown significant
benefits when friction is complex (temperature-dependent, LuGre dynamics). For the standard
Coulomb+viscous model, simultaneous identification is simpler and works adequately.

---

## 4. How Do They Handle Fc/Fv Correlation?

### The problem

In periodic trajectories (the standard for robot identification since Swevers 1997), the velocity
signal dq(t) is smooth and periodic. The Coulomb friction column in the regressor is sign(dq(t)),
and the viscous column is dq(t). For smooth trajectories where velocity stays mostly in one direction
(or is large enough that sign(dq) tracks the shape of dq), these columns become nearly proportional:

    sign(dq) ~ dq / |dq|

When the trajectory has high velocities and few zero-crossings, sign(dq) and dq are highly
correlated, leading to:
- Ill-conditioned regressor matrix (high condition number)
- Fc and Fv estimates that are individually unreliable but whose combined effect is accurate
- Large variance on individual friction parameters

### How the literature addresses this

**1. Trajectory design (most common approach):**
Swevers et al. (2007) and others design trajectories to minimize condition number of the regressor.
This implicitly reduces Fc/Fv correlation by ensuring sufficient zero-crossings and velocity
variation. However, for periodic trajectories, correlation is inherent because the signal structure
is smooth.

**2. Velocity thresholding:**
Multiple papers (referenced in Swevers 2007, the academia.edu version) use a "boundary velocity"
concept: data is only used when |dq| exceeds a threshold, avoiding the nonlinear low-speed regime.
This actually **worsens** Fc/Fv correlation because at high speeds sign(dq) is even more correlated
with dq.

**3. Essential parameters (Pham & Gautier, 1991):**
When Fc and Fv are highly correlated, one of them may become a "non-essential" parameter, i.e., it
can be dropped from the model without significantly affecting prediction accuracy. The essential
parameter concept acknowledges that individually unreliable parameters can be eliminated if their
regressor columns are near-linearly-dependent. This is a principled way to handle the correlation:
**accept it and drop one**.

**4. Asymmetric friction modeling:**
Using separate Fc+/Fc- and Fv+/Fv- for positive/negative directions breaks the correlation pattern
somewhat, because the sign-based columns become more structurally different from the velocity
columns.

**5. No explicit solution in most papers:**
The majority of identification papers do **not** explicitly discuss Fc/Fv correlation. They rely on:
- Well-designed excitation trajectories (D-optimal or condition-number-optimal)
- The linear least squares framework, which gives minimum-variance estimates even with correlated
  regressors (the combined prediction tau_f = Fc*sign(dq) + Fv*dq is still accurate)
- Reporting only the torque prediction error, not individual parameter uncertainties

### Key insight

The Fc/Fv correlation is **not a practical problem for torque prediction** -- it is only a problem
for **individual parameter interpretation**. If you need physically meaningful individual Fc and Fv
values (e.g., for SDP constraints, or for comparing with prior values), the correlation is a real
issue. If you only need accurate torque prediction, the correlation does not matter.

---

## 5. Regularization Approaches

### Euclidean regularization (Tikhonov / ridge)

    min || Y*theta - tau ||^2 + lambda * || theta - theta_prior ||^2

Several papers add a regularization term that penalizes deviation from prior (e.g., CAD) values.
This is applied to all parameters including friction. The weight lambda controls the trade-off.

### Geometric / Riemannian regularization (Lee, Wensing & Park, 2018, 2020)

The set of physically feasible inertial parameters forms a Riemannian manifold (cone of positive
definite matrices). Lee et al. propose using the natural Riemannian metric instead of the Euclidean
norm for regularization:

    min || Y*theta - tau ||^2 + lambda * d_Riemannian(theta, theta_prior)^2

This is **coordinate-invariant** and physically meaningful for inertial parameters. For friction
parameters (which are scalar positives), the Riemannian metric reduces to a log-ratio distance.
The key advantage: regularization strength is consistent across different parameter scales.

### SDP with physical consistency constraints (Sousa & Cortesao, 2014)

Not regularization per se, but **constraints**:
- Inertia tensors must be positive definite (LMI)
- Fc >= 0, Fv >= 0 (simple linear inequality)
- Mass > 0

These constraints act as implicit regularization by restricting the feasible set. When the
unconstrained OLS solution has negative friction, the SDP will project to the nearest feasible point.

### PC-DIDIM (Janot et al., 2021)

Sequential semidefinite optimization that maintains both physical consistency (SDP constraints) and
statistical consistency (proper noise model). Uses iterative refinement where each step solves an SDP.
The physical consistency constraints on friction are Fc >= 0 and Fv >= 0.

### Constrained robot identification (2024)

Recent work explicitly includes Fc_j >= 0, Fv_j >= 0, Ia_j >= 0 as linear constraints alongside
the LMI constraints on inertia tensors. Friction parameters are identified simultaneously with
inertial parameters in a single SDP.

### Practical recommendation from literature

For SDP-based identification:
1. **Always** constrain Fc >= 0 and Fv >= 0 (simple linear inequalities)
2. **Consider** adding regularization toward prior values for friction, especially when Fc/Fv
   correlation is high
3. The geometric regularization of Lee et al. is the state of the art but complex to implement
4. Simple Tikhonov regularization toward CAD/prior values is effective and easy to implement

---

## 6. Full-Body / Humanoid Identification Specifically

### HRP-2 (Venture, Ayusawa, Nakamura)

- **Ayusawa et al. (2008, 2014):** Base-link dynamics method. Identifies **only inertial parameters**
  from the unactuated floating-base dynamics. **No friction model at all.** This is the defining
  feature of the approach: by using the 6-DOF base dynamics (which have no actuators), friction is
  completely bypassed. Demonstrated on HRP-2.

- **Ogawa, Venture & Ott (2014):** Compared joint-torque-based (with friction) and base-link-based
  (without friction) identification on TORO humanoid. When using joint torques, standard Coulomb +
  viscous friction is included. The base-link method gives competitive results without needing to
  model friction.

### iCub (Traversaro, Nori, IIT)

- **Traversaro et al. (2014):** Identified right arm of iCub. Simultaneous identification of
  inertial, **Coulomb + viscous** friction, and DC motor parameters (resistance, back-EMF constant).
  Used PWM measurements and F/T sensor data. Partial Least Squares (PLS) regression to handle
  multicollinearity. **Single-step simultaneous identification.**

- **IIT torque-control-params-estimation (GitHub):** Practical friction identification toolbox for
  iCub joints. Joint-by-joint friction identification from dedicated experiments.

### TORO / DLR (Ogawa, Ott)

- Standard IDIM with Coulomb + viscous friction when using joint torque sensors.
- High-ratio harmonic drives introduce significant friction, making friction modeling critical.

### WALKMAN (IIT)

- **No published identification study found.** The WALKMAN robot uses similar actuation (SEA-based)
  to iCub but at larger scale. Identification methods from iCub would likely transfer.

### Atlas (Boston Dynamics)

- **Koenemann et al. (2015):** Used model-predictive control on HRP-2, not Atlas identification.
- No published full identification of Atlas friction parameters found.

### General humanoid pattern

For humanoids, there are essentially **two schools**:

1. **Base-link/floating-base methods (Ayusawa school):** Avoid friction entirely by using only
   unactuated dynamics. Elegant but requires either free-flying conditions or accurate external
   force measurements.

2. **Joint-torque methods (Gautier/Traversaro school):** Include friction in the standard IDIM,
   typically Coulomb + viscous. Works with any actuated system but requires friction modeling.

---

## 7. Practical Recommendations for SDP-Based Identification with Friction

Based on the literature survey, the state of the art for handling friction in SDP-based
identification is:

### 7.1 Standard friction model

Use **Coulomb + viscous + offset** (3 parameters per joint):

    tau_f_j = Fc_j * sign(dq_j) + Fv_j * dq_j + tau_off_j

The offset term is important practically (absorbs amplifier bias, sensor calibration errors,
asymmetric Coulomb effects). It is used by Gautier in his later work and by many practical
implementations.

### 7.2 Physical consistency constraints

In the SDP, add linear inequality constraints:

    Fc_j >= 0,  Fv_j >= 0   for all j

The offset term tau_off_j is **unconstrained** (can be positive or negative).
These are simple box constraints that integrate trivially into the SDP.

### 7.3 Regularization

Add a penalty term to keep identified friction close to prior/initial values:

    min || Y*theta - tau ||^2_W + lambda_f * sum_j (Fc_j - Fc_j_prior)^2 + (Fv_j - Fv_j_prior)^2

This helps with:
- Fc/Fv correlation (stabilizes individual values)
- Preventing unphysical solutions (e.g., huge Fc compensated by negative Fv)
- Incorporating prior knowledge (from datasheets, previous identification)

The regularization weight lambda_f should be chosen carefully -- too large biases the result,
too small does not help.

### 7.4 Two-step approach (when needed)

Consider a two-step approach when:
- Friction is complex (temperature-dependent, load-dependent, Stribeck effects)
- The standard Coulomb+viscous model is clearly inadequate
- You have access to dedicated friction identification experiments (constant velocity, etc.)

The two-step approach:
1. **Step 1:** Identify friction per joint from dedicated low-speed experiments or constant-velocity
   motions. Can use nonlinear models (Stribeck, LuGre) since this is joint-by-joint.
2. **Step 2:** Subtract identified friction from measured torques, then identify inertial parameters
   from the residual using SDP with physical consistency constraints.

### 7.5 Alternative: base-link dynamics for humanoids

For floating-base robots (humanoids), consider using the Ayusawa base-link dynamics approach to
identify inertial parameters **without any friction model**. Then identify friction separately if
needed.

### 7.6 Handling Fc/Fv correlation specifically

Options, in order of practicality:
1. **Accept it:** If torque prediction is the goal, the combined effect is well-determined even when
   individual Fc/Fv are correlated. SDP constraints (Fc>=0, Fv>=0) prevent the worst pathologies.
2. **Regularize:** Add Tikhonov regularization toward prior friction values.
3. **Use essential parameters:** Drop the less essential of Fc/Fv if correlation is extreme (Pham &
   Gautier, 1991).
4. **Trajectory design:** Include more zero-crossings and velocity variation to reduce correlation.
   This may conflict with other trajectory requirements (excitation of inertial parameters).
5. **Two-step:** Identify friction separately with dedicated experiments.

---

## Key References

### Foundational / textbook
- Gautier, M. & Khalil, W. (1990). "Direct calculation of minimum set of inertial parameters of serial robots." IEEE T-RA.
- Khalil, W. & Dombre, E. (2002). "Modeling, Identification and Control of Robots." Butterworth-Heinemann.
- Swevers, J., Verdonck, W. & De Schutter, J. (2007). "Dynamic model identification for industrial robots." IEEE Control Systems, 27(5), 58-71.

### SDP / physical consistency
- Sousa, C.D. & Cortesao, R. (2014). "Physical feasibility of robot base inertial parameter identification: A linear matrix inequality approach." IJRR, 33(6), 931-944.
- Lee, T., Wensing, P.M. & Park, F.C. (2020). "Geometric robot dynamic identification: A convex programming approach." IEEE T-RO, 36(2), 348-365.
- Janot, A. et al. (2021). "Sequential semidefinite optimization for physically and statistically consistent robot identification." Control Engineering Practice, 101, 104496.

### Friction-specific
- Elhami, M.R. & Brookfield, D.J. (1996). "Sequential identification of Coulomb and viscous friction in robot drives." Automatica, 32(10), 1479-1482.
- Grotjahn, M., Daemi, M. & Heimann, B. (2001). "Friction and rigid body identification of robot dynamics." IJSS, 38, 1889-1902.
- Hamon, P., Gautier, M., Garrec, P. & Janot, A. (2010). "Dynamic identification of robot with a load-dependent joint friction model." IEEE RAM.

### Two-step approaches
- Park, D.I. et al. (2022). "A two-step method for dynamic parameter identification of Indy7 collaborative robot manipulator." Sensors, 22(24), 9708.

### Humanoid identification
- Ayusawa, K., Venture, G. & Nakamura, Y. (2014). "Identifiability and identification of inertial parameters using the underactuated base-link dynamics for legged multibody systems." IJRR, 33(3), 446-468.
- Traversaro, S. et al. (2014). "Inertial parameter identification including friction and motor dynamics." IEEE-RAS Humanoids.
- Ogawa, Y., Venture, G. & Ott, C. (2014). "Dynamic parameters identification of a humanoid robot using joint torque sensors and/or contact forces." IEEE-RAS Humanoids.

### Essential parameters
- Pham, C.M. & Gautier, M. (1991). "Essential parameters of robots." IEEE CDC.

### Modern review
- Lee, T., Kwon, J., Wensing, P.M. & Park, F.C. (2024). "Robot model identification and learning: A modern perspective." Annual Review of Control, Robotics, and Autonomous Systems, 7.
