# Tutorial

The goal of identification is usually to find dynamic model parameters from
measurements of motions. Ideally, some previous parameters are available e.g.
from a CAD model which also gives the kinematic model. In the following, an
example identification is carried out for the Kuka LWR4+ robot. The options for
all steps of this task are held within a configuration file in the config/ dir.

1. copy an existing .yaml configuration file and customize it for your setup with a text editor.
`cp configs/kuka_lwr4.yaml configs/example.yaml`

2. Use the trajectory.py script to generate an optimal exciting trajectory
(works for both fixed-base and floating-base robots). The corresponding options
in the configuration should be set (for the case of the LWR4+ that is done) and
optionally supply a world urdf file that includes the ground and objects that
the robot might collide with, e.g. a table. The optimization will simulate each
trajectory and check for all constraints to be met while maximizing the
D-optimality of the dynamics regressor. This might take a while depending on
the degrees of freedom. An output file containing the found parameters of the
trajectory will be saved.
`uv run trajectory.py --config configs/example.yaml --model model/example.urdf --world model/world.urdf`

3. Get joint torque measurements for the trajectory from your robotic system, if
suitable by using the excite.py script. It will load the previously created
trajectory file and move the robot through the specified module (in the config
file). If necessary, look at the existing modules and write a custom one for
your communication method. After retrieving the measurements, filtering as well
as deriving velocity and acceleration is done and is saved to a measurements
file. If you are using other means of motion control and data recording and
don't use the excite.py script, the data needs to be filtered and saved to a
numpy data file that has the expected data fields (see README.md in
excitation/). There is also the **csv2npz.py** script that loads raw data from
csv text files, preprocesses them with the same filtering and writes to the
container format (you'll need to customize it for the columns in your csv file
etc.).
`uv run excite.py --model model/example.urdf --config configs/example.yaml --plot`

Alternatively, use **simulator.py** to simulate realistic measurements from
the trajectory without a physical robot. This adds configurable effects
(friction, sensor noise, backlash, cable forces, thermal drift, etc.):
`uv run simulator.py --config configs/example.yaml --model model/example.urdf`

4. Finally, run identifier.py with the measurements file and again the kinematic
model in a .urdf file with the a priori parameters. These parameters don't
have to be physical consistent but it's recommended (they should be when they
come from a CAD system). Optionally you can supply an output .urdf file path
to which the input urdf with exchanged identified parameters is written.
Another measurements file can be supplied for validation. A separate
`--model_real` URDF can be provided as ground truth for comparison.
This can be the model that was used for simulation while the identification
starts at a slightly different model. 
`uv run identifier.py --config configs/example.yaml --model model/example.urdf \`
`--measurements model/example.urdf.measurements.npz --output model/example_identified.urdf`
For real measurement data, the real URDF is of course not known so the parameter has no use.

The output html file in output/ should look similar to the following:

```Linear (relative to Frame) Standard Parameters
|A priori     |Ident        |Change |%e     |Constr  |Description
|   1.60000000|   1.60000000| 0.0000|    0.0|nID     |#0: m_0 - mass of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#1: c_0x - first moment of mass (x) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#2: c_0y - first moment of mass (y) of link lwr_base_link
|   0.08800000|   0.08800000| 0.0000|    0.0|nID     |#3: c_0z - first moment of mass (z) of link lwr_base_link
|   0.00789333|   0.00789333| 0.0000|    0.0|nID     |#4: I_0xx - moment of inertia (xx) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#5: I_0xy - moment of inertia (xy) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#6: I_0xz - moment of inertia (xz) of link lwr_base_link
|   0.00772000|   0.00772000| 0.0000|    0.0|nID     |#7: I_0yy - moment of inertia (yy) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#8: I_0yz - moment of inertia (yz) of link lwr_base_link
|   0.00305333|   0.00305333| 0.0000|    0.0|nID     |#9: I_0zz - moment of inertia (zz) of link lwr_base_link
|   2.70000000|   2.70038805| 0.0004|    0.0|mA nID  |#10: m_1 - mass of link lwr_1_link
|   0.00000000|   0.00000000| 0.0000|      -|hull nID|#11: c_1x - first moment of mass (x) of link lwr_1_link
|  -0.16200000|  -0.14756097| 0.0144|   -8.9|hull nID|#12: c_1y - first moment of mass (y) of link lwr_1_link
|   0.35100000|   0.35109926| 0.0001|    0.0|hull nID|#13: c_1z - first moment of mass (z) of link lwr_1_link
|   0.07380000|   0.07380000| 0.0000|    0.0|nID     |#14: I_1xx - moment of inertia (xx) of link lwr_1_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#15: I_1xy - moment of inertia (xy) of link lwr_1_link
|   0.00000000|   0.00000000| 0.0000|      -|nID     |#16: I_1xz - moment of inertia (xz) of link lwr_1_link
|   0.04968000|   0.04968639| 0.0000|    0.0|nID     |#17: I_1yy - moment of inertia (yy) of link lwr_1_link
|   0.02106000|   0.01921333|-0.0018|   -8.8|nID     |#18: I_1yz - moment of inertia (yz) of link lwr_1_link
|   0.02574000|   0.00806457|-0.0177|  -68.7|        |#19: I_1zz - moment of inertia (zz) of link lwr_1_link
|   2.70000000|   2.70093962| 0.0009|    0.0|mA nID  |#20: m_2 - mass of link lwr_2_link
|   0.00000000|   0.07999894| 0.0800|      -|hull    |#21: c_2x - first moment of mass (x) of link lwr_2_link
|   0.16200000|   0.14356275|-0.0184|  -11.4|hull nID|#22: c_2y - first moment of mass (y) of link lwr_2_link
|   0.18900000|   0.15034978|-0.0387|  -20.4|hull    |#23: c_2z - first moment of mass (z) of link lwr_2_link
|   0.04140000|   0.03734072|-0.0041|   -9.8|        |#24: I_2xx - moment of inertia (xx) of link lwr_2_link
|   0.00000000|  -0.04357467|-0.0436|      -|        |#25: I_2xy - moment of inertia (xy) of link lwr_2_link
|   0.00000000|  -0.01901573|-0.0190|      -|        |#26: I_2xz - moment of inertia (xz) of link lwr_2_link
|   0.01728000|   0.08319940| 0.0659|  381.5|        |#27: I_2yy - moment of inertia (yy) of link lwr_2_link
|  -0.01134000|   0.01884285| 0.0302| -266.2|        |#28: I_2yz - moment of inertia (yz) of link lwr_2_link
|   0.02574000|   0.01993899|-0.0058|  -22.5|        |#29: I_2zz - moment of inertia (zz) of link lwr_2_link
|   2.70000000|   2.69594300|-0.0041|   -0.2|mA      |#30: m_3 - mass of link lwr_3_link
|   0.00000000|  -0.05705169|-0.0571|      -|hull    |#31: c_3x - first moment of mass (x) of link lwr_3_link
|   0.16200000|   0.07116409|-0.0908|  -56.1|hull    |#32: c_3y - first moment of mass (y) of link lwr_3_link
|   0.35100000|   0.30821349|-0.0428|  -12.2|hull    |#33: c_3z - first moment of mass (z) of link lwr_3_link
|   0.07380000|   0.16312944| 0.0893|  121.0|        |#34: I_3xx - moment of inertia (xx) of link lwr_3_link
|   0.00000000|   0.07386126| 0.0739|      -|        |#35: I_3xy - moment of inertia (xy) of link lwr_3_link
|   0.00000000|  -0.00914788|-0.0091|      -|        |#36: I_3xz - moment of inertia (xz) of link lwr_3_link
|   0.04968000|   0.07799029| 0.0283|   57.0|        |#37: I_3yy - moment of inertia (yy) of link lwr_3_link
|  -0.02106000|  -0.01713353| 0.0039|  -18.6|        |#38: I_3yz - moment of inertia (yz) of link lwr_3_link
|   0.02574000|   0.00503550|-0.0207|  -80.4|        |#39: I_3zz - moment of inertia (zz) of link lwr_3_link
|   2.70000000|   2.69846481|-0.0015|   -0.1|mA      |#40: m_4 - mass of link lwr_4_link
|   0.00000000|   0.02357962| 0.0236|      -|hull    |#41: c_4x - first moment of mass (x) of link lwr_4_link
|  -0.16200000|  -0.07291156| 0.0891|  -55.0|hull    |#42: c_4y - first moment of mass (y) of link lwr_4_link
|   0.18900000|   0.25459268| 0.0656|   34.7|hull    |#43: c_4z - first moment of mass (z) of link lwr_4_link
|   0.04140000|   0.02599117|-0.0154|  -37.2|        |#44: I_4xx - moment of inertia (xx) of link lwr_4_link
|   0.00000000|   0.00063577| 0.0006|      -|        |#45: I_4xy - moment of inertia (xy) of link lwr_4_link
|   0.00000000|  -0.00222416|-0.0022|      -|        |#46: I_4xz - moment of inertia (xz) of link lwr_4_link
|   0.01728000|   0.02600157| 0.0087|   50.5|        |#47: I_4yy - moment of inertia (yy) of link lwr_4_link
|   0.01134000|   0.00619794|-0.0051|  -45.3|        |#48: I_4yz - moment of inertia (yz) of link lwr_4_link
|   0.02574000|   0.00243850|-0.0233|  -90.5|        |#49: I_4zz - moment of inertia (zz) of link lwr_4_link
|   1.70000000|   1.70059056| 0.0006|    0.0|mA      |#50: m_5 - mass of link lwr_5_link
|   0.00000000|  -0.00000231|-0.0000|      -|hull    |#51: c_5x - first moment of mass (x) of link lwr_5_link
|   0.00000000|  -0.03348347|-0.0335|      -|hull    |#52: c_5y - first moment of mass (y) of link lwr_5_link
|   0.21080000|   0.02605965|-0.1847|  -87.6|hull    |#53: c_5z - first moment of mass (z) of link lwr_5_link
|   0.03689227|   0.00126564|-0.0356|  -96.6|        |#54: I_5xx - moment of inertia (xx) of link lwr_5_link
|   0.00000000|   0.00434789| 0.0043|      -|        |#55: I_5xy - moment of inertia (xy) of link lwr_5_link
|   0.00000000|  -0.00357500|-0.0036|      -|        |#56: I_5xz - moment of inertia (xz) of link lwr_5_link
|   0.02868920|   0.09215477| 0.0635|  221.2|        |#57: I_5yy - moment of inertia (yy) of link lwr_5_link
|   0.00000000|  -0.07493086|-0.0749|      -|        |#58: I_5yz - moment of inertia (yz) of link lwr_5_link
|   0.00922307|   0.06269313| 0.0535|  579.7|        |#59: I_5zz - moment of inertia (zz) of link lwr_5_link
|   1.60000000|   1.53803501|-0.0620|   -3.9|mA      |#60: m_6 - mass of link lwr_6_link
|   0.00000000|   0.05572949| 0.0557|      -|hull    |#61: c_6x - first moment of mass (x) of link lwr_6_link
|   0.00000000|   0.01178897| 0.0118|      -|hull    |#62: c_6y - first moment of mass (y) of link lwr_6_link
|   0.10000000|   0.02552880|-0.0745|  -74.5|hull    |#63: c_6z - first moment of mass (z) of link lwr_6_link
|   0.01041667|   0.00694724|-0.0035|  -33.3|        |#64: I_6xx - moment of inertia (xx) of link lwr_6_link
|   0.00000000|   0.02900805| 0.0290|      -|        |#65: I_6xy - moment of inertia (xy) of link lwr_6_link
|   0.00000000|  -0.00342234|-0.0034|      -|        |#66: I_6xz - moment of inertia (xz) of link lwr_6_link
|   0.01041667|   0.13714757| 0.1267| 1216.6|        |#67: I_6yy - moment of inertia (yy) of link lwr_6_link
|   0.00000000|  -0.01162409|-0.0116|      -|        |#68: I_6yz - moment of inertia (yz) of link lwr_6_link
|   0.00416667|   0.00308028|-0.0011|  -26.1|        |#69: I_6zz - moment of inertia (zz) of link lwr_6_link
|   0.30000000|   0.32613457| 0.0261|    8.7|mA      |#70: m_7 - mass of link lwr_7_link
|   0.00000000|  -0.01304538|-0.0130|      -|hull    |#71: c_7x - first moment of mass (x) of link lwr_7_link
|   0.00000000|  -0.01299741|-0.0130|      -|hull    |#72: c_7y - first moment of mass (y) of link lwr_7_link
|   0.00000000|  -0.01011016|-0.0101|      -|hull    |#73: c_7z - first moment of mass (z) of link lwr_7_link
|   0.05000000|   0.04107579|-0.0089|  -17.8|        |#74: I_7xx - moment of inertia (xx) of link lwr_7_link
|   0.00000000|   0.04393115| 0.0439|      -|        |#75: I_7xy - moment of inertia (xy) of link lwr_7_link
|   0.00000000|   0.01303511| 0.0130|      -|        |#76: I_7xz - moment of inertia (xz) of link lwr_7_link
|   0.05000000|   0.15415436| 0.1042|  208.3|        |#77: I_7yy - moment of inertia (yy) of link lwr_7_link
|   0.00000000|   0.05689212| 0.0569|      -|        |#78: I_7yz - moment of inertia (yz) of link lwr_7_link
|   0.05000000|   0.02472662|-0.0253|  -50.5|        |#79: I_7zz - moment of inertia (zz) of link lwr_7_link
|   0.50000000|   0.61068699| 0.1107|   22.1|        |#80: Fc_0 - Coulomb friction of joint lwr_0_joint
|   0.80000000|   0.36327134|-0.4367|  -54.6|        |#81: Fc_1 - Coulomb friction of joint lwr_1_joint
|   0.40000000|   0.30395749|-0.0960|  -24.0|        |#82: Fc_2 - Coulomb friction of joint lwr_2_joint
|   0.30000000|   0.47277531| 0.1728|   57.6|        |#83: Fc_3 - Coulomb friction of joint lwr_3_joint
|   0.20000000|   0.59497383| 0.3950|  197.5|        |#84: Fc_4 - Coulomb friction of joint lwr_4_joint
|   0.10000000|   0.15956173| 0.0596|   59.6|        |#85: Fc_5 - Coulomb friction of joint lwr_5_joint
|   0.05000000|   0.03785973|-0.0121|  -24.3|        |#86: Fc_6 - Coulomb friction of joint lwr_6_joint
|   1.00000000|   1.06901601| 0.0690|    6.9|>0      |#87: Fv_0 - Viscous friction of joint lwr_0_joint
|   1.20000000|   0.95833323|-0.2417|  -20.1|>0      |#88: Fv_1 - Viscous friction of joint lwr_1_joint
|   0.90000000|   0.22328950|-0.6767|  -75.2|>0      |#89: Fv_2 - Viscous friction of joint lwr_2_joint
|   0.80000000|   0.68860168|-0.1114|  -13.9|>0      |#90: Fv_3 - Viscous friction of joint lwr_3_joint
|   0.50000000|   0.06502273|-0.4350|  -87.0|>0      |#91: Fv_4 - Viscous friction of joint lwr_4_joint
|   0.30000000|   0.25192485|-0.0481|  -16.0|>0      |#92: Fv_5 - Viscous friction of joint lwr_5_joint
|   0.20000000|   0.07587319|-0.1241|  -62.1|>0      |#93: Fv_6 - Viscous friction of joint lwr_6_joint
|   0.00000000|   0.86747016| 0.8675|      -|        |#94: off_0 - Torque offset of joint lwr_0_joint
|   0.00000000|  -0.00623038|-0.0062|      -|        |#95: off_1 - Torque offset of joint lwr_1_joint
|   0.00000000|   0.00067483| 0.0007|      -|        |#96: off_2 - Torque offset of joint lwr_2_joint
|   0.00000000|   0.13734284| 0.1373|      -|        |#97: off_3 - Torque offset of joint lwr_3_joint
|   0.00000000|  -0.50697281|-0.5070|      -|        |#98: off_4 - Torque offset of joint lwr_4_joint
|   0.00000000|   0.17727199| 0.1773|      -|        |#99: off_5 - Torque offset of joint lwr_5_joint
|   0.00000000|  -0.11691153|-0.1169|      -|        |#100: off_6 - Torque offset of joint lwr_6_joint

Parameters
Estimated overall mass: 15.96 kg vs. a priori 16.0 kg
A priori parameters are physical consistent
Identified parameters are physical consistent
Squared distance of identifiable std parameter vectors to a priori: 2.37
Squared distance of base parameter vectors (identified vs. a priori): 1.63

Torque prediction errors
Relative mean residual error: 2.15% vs. A priori: 9.24%
Absolute mean residual error: 0.58 vs. A priori: 2.52
NRMS of residual error: 0.15% vs. A priori: 0.48%

Relative validation error: 2.81%
Absolute validation error: 0.72 Nm
NRMS validation error: 0.18%
```

The table columns show parameters for A priori (URDF), Identified and the absolute change between them. There also is a percentual difference value (%e) given in relation to the magnitude of the a priori value; it shows `-` when the a priori value is zero (a percentage relative to zero is not meaningful, so parameters such as the off-diagonal inertias and the torque offsets — whose CAD value is 0 — are read from the absolute Change column instead). The inertial parameters of each link are followed by the per-joint friction parameters (`Fc` Coulomb, `Fv` viscous, and a torque offset `off` that absorbs amplifier bias and sensor offsets). On this fixed-base robot they are first identified together with the inertials in the SDP and then refit per joint from the joint-torque residual (see below).

The different estimation error measures that are given are

Absolute mean error:
The mean over the error vector norms for each joint.

Relative mean error: 
The absolute mean error normalized with the norm of the measured data vectors.

Normalized root mean square (NRMS) error:
The square root of the mean over the joints of the squared error, normalized by the possible torque range of each joint (as given in the URDF).

An assessment of the quality of the result should be made through the combination of torque prediction accuracy, validation accuracy (ideally multiple different validation trajectories) and also the estimated torque curve shapes compared to the measured torques.

When a ground truth model is given (`--model_real`), the output also shows the
distance of base parameters to real. Base parameters (linear combinations of standard
parameters) are what the data actually determines — standard parameters are non-unique
(many sets produce the same base parameters and torques). A standard parameter distance
that increases while base parameters improve is expected and not a problem.

Some individual standard parameters can still differ a lot from the a priori value (large
`%e` or a large absolute Change), e.g. a weakly-excited inertia such as `I_2yy`. This is the
same null space: the data does not determine those parameters individually, so the
closest-to-CAD recovery has freedom there. With `cadRegularizationMode: 'uniform'` that freedom is allocated to
minimize the *total* distance, which can let one or two weakly-determined parameters absorb
the physical-consistency constraints and grow far from CAD. The `observability` mode
(used in this example) weights the pull toward CAD by how poorly each parameter is
determined, so those parameters stay near CAD while the well-determined ones remain free —
on this real KUKA data it keeps the per-link parameters plausible and lowers the held-out
validation error at the same training fit. The change is purely in the null space; it does
not (and cannot) make the data determine more.

On this fixed-base robot the friction is refit after the inertial identification
(`postIdentifyFriction`): with the inertials held fixed, `Fc`/`Fv`/offset are re-estimated
per joint from the joint-torque residual, using a velocity dead zone (dropping the
unreliable near-zero-velocity samples) and a prior that pulls `Fv` toward the URDF value
where a joint is weakly excited. The prior weight is set unit-free via
`frictionFvRegularizationRelative` (a fraction of the joint's own excitation energy), so it
transfers across robots without guessing an absolute number — raise it if a joint's `Fv`
collapses to zero. Because the refit trusts the data over the rough CAD friction where
joints move well, the friction parameters move noticeably from a priori, which enlarges the
*standard*-parameter distance (it includes friction) even though the base parameters and the
held-out validation improve. Reading the friction errors directly needs a `--model_real`,
which real hardware does not have, so judge friction by held-out validation instead.

Setting `frictionFvRegularizationRelative` on a real robot (no ground truth needed): the
weight becomes `alpha * median(per-joint velocity energy)`, where the energy is computed
from your own measurements, so `alpha` is a dimensionless trust knob — at `alpha = 1` a
joint excited at the median level gets a 50/50 data/prior blend, smaller values trust the
data more. A practical recipe:

1. Start around `alpha = 0.1–0.2`.
2. Look at the printed `Fv:` range and the per-joint values: if any joint's `Fv` is pinned
   at `0` (its `Fc`/`Fv` split has degenerated — typical for a joint that moved at a narrow
   speed band), raise `alpha` until no joint collapses.
3. Sweep `alpha` and keep the value that minimizes the held-out validation error (or sits at
   the flat part of the curve); this needs only a second measurement file, not real
   parameters.

The absolute `frictionFvRegularization` (raw energy units) is still available but harder to
transfer between robots/trajectories, so prefer the relative form. The same two friction
settings apply to the floating-base two-step path as well.

Note also that the identified torques (orange in the per-joint plots) track the measured
torques (green) far better than CAD (blue) but can still miss sharp features, most visibly
the spikes at velocity reversals. Those come from friction effects near zero velocity
(stiction/Stribeck) and backlash that a Coulomb+viscous model cannot represent; on real
hardware they are an expected residual, not an identification error. A poorly-excited joint
(little movement on this trajectory) will likewise have less certain friction and inertia
parameters.


## Important configuration settings

### Total robot mass

Set `limitMassVal` to the actual total robot mass (from weighing). This
constraint allows the identifier to distribute mass based only on torque
fitting, which can lead to the wrong total mass and a constant offset in base
force predictions. Set `limitMassRange` to a small tolerance (e.g. 0.5 kg):

```yaml
limitOverallMass: 1
limitMassVal: 16.0        # total mass in kg (from weighing the robot)
limitMassRange: 0.5       # allowed deviation (kg)
```

Per-link mass bounds (`limitMassToApriori`, `limitMassAprioriBoundary`) should allow
enough room for each link to adjust. A boundary of 10-20% is typical:

```yaml
limitMassToApriori: 1
limitMassAprioriBoundary: 0.15   # ±15% from a priori per link
```

### Pinning non-identifiable links

Links connected via fixed joints (sensor frames, end-effectors, protective covers)
cannot be independently identified from torque data. Pin them to their a priori values
using `dontChangeLinks` to prevent the solver from assigning them arbitrary values:

```yaml
dontChangeLinks: ['imu_link', 'camera_frame', 'end_effector', ...]
```

This also prevents infeasible SDP constraints from zero-mass virtual links.

### SDP solver

The default solver is CLARABEL (bundled with cvxpy). For large robots, a warm start
from a priori parameters helps the solver. If CLARABEL fails, try SCS (`sdpSolver: 'scs'`)
which is more robust but slower and less precise. MOSEK (`sdpSolver: 'mosek'`) is
also a good option but requires a [license](https://www.mosek.com/license/request/trial/).

```yaml
sdpSolver: 'clarabel'
sdpSafeMargin: 1.0e-6     # eigenvalue lower bound for physical consistency
```


## Floating-base robots

For floating-base robots (humanoids, mobile manipulators), additional configuration
is needed.

### Base attachment type

The `floatingBaseAttachment` option defines how the robot's base is connected to
the world during the identification experiment:

```yaml
floatingBase: 1
floatingBaseAttachment: 'suspended'     # 'fixed', 'suspended', or 'free'
floatingBaseAttachmentFrame: 'crane_ft' # URDF frame where the chain/crane attaches
suspendedDamping: 500.0                 # ball joint damping (Nm·s/rad)
```

- **`fixed`**: Base is rigidly mounted (e.g. prop/stand). The 6 base wrench equations
  are simulated from the a priori model and add no independent information. Only joint
  torque equations contribute to identification.

- **`suspended`**: Robot hangs from a ball joint at the attachment frame (e.g. crane).
  The base swings as joints move, producing real base dynamics. Both joint torques AND
  base wrench equations contribute to identification. The URDF must have a link at the
  attachment point (e.g. `crane_ft`) with an `<inertial>` element (even a tiny dummy mass).

- **`free`**: Truly free-floating (e.g. walking). Base motion and contact wrenches must
  come from real measurements (IMU + F/T sensors). Not yet supported for trajectory
  generation.

### Trajectory optimization for suspended robots

The trajectory optimizer accounts for suspended dynamics when
`floatingBaseAttachment: 'suspended'`. The optimizer evaluates each candidate
trajectory with the pendulum simulation, checking that joint torques stay within
limits even with the swinging base. This makes optimization slower but produces
trajectories that are safe for the suspended setup.

Collision checking uses the swung base pose for each sample, not the fixed mount
pose: as the joints move, the simulated base swings, and self- and world-collision
checks are evaluated at the resulting pose.
Provide the world geometry with `--world model/world_suspended.urdf` so the crane and
ground can be checked against it.

```yaml
worldCollisionMargin: 0.1        # required clearance (m) to world geometry;
                                 # box hulls under-approximate protruding parts
ignoreCollisionBetweenGroups: [] # [[groupA, groupB], ...] to skip cross-group pairs.
                                 # Do NOT group-ignore arm/hand vs leg pairs: under a
                                 # swinging base a hand can reach a lower leg.
```

Two robustness aids make global optimization practical when the feasible region is
tiny (common for high-DOF humanoids under collision + torque constraints):

```yaml
globalOptAmplitudeRepair: 1      # back off Fourier amplitudes of infeasible Optuna
                                 # candidates until constraints pass, so most trials
                                 # contribute a feasible solution instead of being lost
trajectorySeedSolutions: []      # .npz files of previously found trajectories to enqueue
                                 # as initial trials (joint/harmonic structure must match
                                 # trajectoryNf) — starts from a known-feasible point
```

### Exciting velocity for friction identification

D-optimality of the inertial regressor rewards acceleration and pose diversity but
does *not* by itself reward joint velocity, so a purely inertia-optimal trajectory can
leave some joints barely moving. Friction parameters — viscous `Fv` especially — are
then unidentifiable for those joints. Add a soft per-joint peak-velocity target so the
optimizer keeps every joint moving:

```yaml
trajectoryTargetVelocity: 0.5    # target peak velocity (rad/s) per joint, soft cost.
                                 # 0 = disabled.
trajectoryPulseMax: 0.3          # upper bound on Fourier pulsation; higher gives the
                                 # velocity target headroom (also lengthens the period)
```

This is a soft cost, so it trades against the D-optimality and torque/collision
constraints rather than overriding them. Slow joints (proximal/torso joints of a
suspended humanoid) benefit the most.

### Two-step friction identification

For floating-base robots, identify friction *after* the inertial parameters rather than
simultaneously. The inertial parameters are estimated friction-free from the base-wrench
equations (which contain no joint friction), then `Fc`/`Fv`/offset are fit per joint from
the joint-torque residual:

```yaml
useBaseWrenchForBaseParams: 1    # base params from the base wrench (friction-free)
identifyFrictionSimultaneously: 0
postIdentifyFriction: 1          # fit friction per joint from the residual afterwards
identifySymmetricVelFriction: 1  # one Fv per joint (required to write back to URDF)
```

The per-joint friction fit has a few settings worth knowing about. Their defaults are
tuned for the WALKMAN simulator and documented inline in `configs/walkman_full.yaml`;
the key ideas:

- `frictionVelocityCutoff` (Hz): the velocity used for the Coulomb sign (`tanh`) term is
  low-pass filtered at this cutoff so it does not chatter by ±`Fc` when the velocity hovers
  in its noise floor. The filter is zero-phase, so zero crossings are not shifted. Choose
  just above the trajectory's velocity bandwidth; set ≥ Nyquist to disable.
- `frictionVelocityDeadZone` (rad/s): samples with `|v|` below this are dropped from the
  fit — there the sign is unreliable and the `tanh` is in its linear region (collinear with
  the viscous column). Choose above both the velocity noise floor and a small multiple of
  `frictionSignThreshold`; too large loses slow joints' data.
- `frictionFvRegularization`: Tikhonov weight pulling `Fv` toward the a priori URDF value,
  expressed as equivalent excitation energy `sum(v^2)`. Joints with velocity energy well
  above it are barely affected; weakly-excited joints stay near a priori. A moderate prior
  helps even when the a priori `Fv` is off, because an unregularized `Fv` otherwise absorbs
  unmodeled effects (thermal drift, cable forces).

When a `--model_real` ground-truth URDF is given, the identifier also reports friction
parameter errors against it, which is the most direct way to tune these.

### Multiple trajectories

A single trajectory rarely excites every parameter direction well, and on a high-DOF
floating base the largest accuracy gains come from combining several. There are three
ways to use multiple trajectories, in increasing sophistication:

1. **Independent pooling.** Optimize several trajectories from separate optimizer runs
   (each uses its own random seed), execute/simulate each, then pass all measurement
   files to the identifier at once:

   ```bash
   uv run identifier.py --config configs/example.yaml --model model/example.urdf \
     --measurements traj1.npz --measurements traj2.npz --measurements traj3.npz
   ```

   Pooling diverse excitation makes the *identifiable* (base) parameters markedly more
   accurate. Different random seeds give genuinely different motions, which is what we
   want here.

2. **Per-trajectory inverse-noise weighting.** When pooling, weight each trajectory by
   its inverse noise so cleaner data counts more:

   ```yaml
   useTrajectoryWeighting: 1
   ```

   In noise-free simulation this is a no-op; it pays off on real data where trajectories
   differ in excitation quality.

3. **Sequential experiment design.** Optimize the *next* trajectory to add information in
   exactly the directions the previous ones left weak. Point the optimizer at the
   already-executed measurement files; their accumulated information matrix is added to the
   D-optimality objective:

   ```yaml
   trajectoryPriorMeasurements: ['traj1.npz', 'traj2.npz']
   ```

   Then identify on all files together as in option 1.

Note that pooling and weighting improve the *base* parameters and the torque model;
they do **not** recover more *standard* parameters — which standard parameters are
identifiable is structural (fixed by the kinematics and sensor set) and excitation cannot
change it.

### Standard-parameter recovery from a feasible base solution

The physically-consistent base solution leaves the standard parameters non-unique in the
data null space; FloBaRoID recovers a full standard-parameter set closest to the a priori
(CAD) values. By default every parameter is pulled toward CAD uniformly. The
`observability` mode instead weights each parameter's pull by how poorly the data
determines it: well-determined parameters stay free, weakly-determined ones stay near CAD.

```yaml
cadRegularizationMode: observability   # 'uniform' (default) or 'observability'
```

This keeps the decomposition sensible where the data is weak. It is opt-in because on
well-conditioned robots it barely changes the fit but on small or poorly-conditioned
systems it can trade a little of the torque fit for staying closer to CAD. Standard parameters
are prior-dominated regardless of mode — report the base-parameter distance (shown with
`--model_real`) as the primary metric.

### Sensor noise and simulation effects

When using the simulator for testing, be aware that unmodeled effects (backlash,
cable forces, thermal drift, Stribeck friction) can significantly degrade
identification quality — the identifier's linear friction model cannot capture them,
and they get absorbed into the inertial parameters. For benchmarking, consider
disabling effects that the identifier cannot model:

```yaml
simulateFriction: 0            # includes Stribeck stiction (not modeled by identifier)
simulateThermalDrift: 0
simulateCableForces: 0
simulateStructuralDeflection: 0
simulateBacklash: 0
```
