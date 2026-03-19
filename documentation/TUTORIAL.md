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
trajectory and check for all constraints to be met while minimizing the
condition number of the dynamics regressor. This might take a while depending on
the degrees of freedom. An output file containing the found parameters of the
trajectory will be saved.
`./trajectory.py --config configs/example.yaml --model model/example.urdf --world model/world.urdf`

3. Get joint torque measurements for the trajectory from your robotic system, if
suitable by using the excite.py script. It will load the previously created
trajectory file and move the robot through the specified module (in the config
file). Alternatively, simulation can be enabled to simulate the torques using
the supplied model parameters. If necessary, look at the existing modules and
write a custom one for your communication method. After retrieving the
measurements, filtering as well as deriving velocity and acceleration is done
and is saved to a measurements file. If you are using other means of motion
control and data recording and don't use the excite.py script, the data needs to
be filtered and saved to a numpy data file that has the expected data fields
(see README.md in excitation/). There is also the **csv2npz.py** script that
loads raw data from csv text files, preprocesses them with the same filtering
and writes to the container format (you'll need to customize it for the columns
in your csv file etc.).
In this example for the LWR4+, we simply simulate the trajectory file to receive a measurements file.
`./excite.py --model model/example.urdf --config configs/example.yaml --plot \`
`--trajectory model/example.urdf.trajectory.npz --filename measurements.npz`

4. Finally, run identifier.py with the measurements file and again the kinematic
   model in a .urdf file with the a priori parameters. These parameters don't
   have to be physical consistent but it's recommended (they should be when they
   come from a CAD system). The constrained optimization for identification
   Optionally you can supply an output .urdf file path to which the input urdf
   with exchanged identified parameters is written. Another measurements file
   can be supplied for validation.
`./identifier.py --config configs/example.yaml --model model/example.urdf --measurements \`
`measurements.npz --validation measurements_2.npz --output model/example_identified.urdf`

The output html file in output/ should look similar to the following:

```Linear (relative to Frame) Standard Parameters
|A priori     |Ident        |Change |%e     |Constr  |Description
|   1.60000000|   1.60000000| 0.0000|    0.0|nID     |#0: m_0 - mass of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#1: c_0x - first moment of mass (x) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#2: c_0y - first moment of mass (y) of link lwr_base_link
|   0.08800000|   0.08800000| 0.0000|    0.0|nID     |#3: c_0z - first moment of mass (z) of link lwr_base_link
|   0.00789333|   0.00789333| 0.0000|    0.0|nID     |#4: I_0xx - moment of inertia (xx) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#5: I_0xy - moment of inertia (xy) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#6: I_0xz - moment of inertia (xz) of link lwr_base_link
|   0.00772000|   0.00772000| 0.0000|    0.0|nID     |#7: I_0yy - moment of inertia (yy) of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#8: I_0yz - moment of inertia (yz) of link lwr_base_link
|   0.00305333|   0.00305333| 0.0000|    0.0|nID     |#9: I_0zz - moment of inertia (zz) of link lwr_base_link
|   2.70000000|   2.03324708|-0.6668|  -24.7|mA nID  |#10: m_1 - mass of link lwr_1_link
|   0.00000000|   0.00000000| 0.0000|    0.0|hull nID|#11: c_1x - first moment of mass (x) of link lwr_1_link
|  -0.16200000|  -0.05794951| 0.1041|  -64.2|hull nID|#12: c_1y - first moment of mass (y) of link lwr_1_link
|   0.35100000|   0.33404496|-0.0170|   -4.8|hull nID|#13: c_1z - first moment of mass (z) of link lwr_1_link
|   0.07380000|   0.09366320| 0.0199|   26.9|nID     |#14: I_1xx - moment of inertia (xx) of link lwr_1_link
|   0.00000000|  -0.00000000|-0.0000|   -0.0|nID     |#15: I_1xy - moment of inertia (xy) of link lwr_1_link
|   0.00000000|  -0.00000000|-0.0000|   -0.0|nID     |#16: I_1xz - moment of inertia (xz) of link lwr_1_link
|   0.04968000|   0.07864973| 0.0290|   58.3|nID     |#17: I_1yy - moment of inertia (yy) of link lwr_1_link
|   0.02106000|   0.00959092|-0.0115|  -54.5|nID     |#18: I_1yz - moment of inertia (yz) of link lwr_1_link
|   0.02574000|   0.00198631|-0.0238|  -92.3|        |#19: I_1zz - moment of inertia (zz) of link lwr_1_link
|   2.70000000|   3.56688319| 0.8669|   32.1|mA nID  |#20: m_2 - mass of link lwr_2_link
|   0.00000000|   0.04335586| 0.0434|  433.6|hull    |#21: c_2x - first moment of mass (x) of link lwr_2_link
|   0.16200000|   0.18364408| 0.0216|   13.4|hull nID|#22: c_2y - first moment of mass (y) of link lwr_2_link
|   0.18900000|   0.39412456| 0.2051|  108.5|hull    |#23: c_2z - first moment of mass (z) of link lwr_2_link
|   0.04140000|   0.05633210| 0.0149|   36.1|        |#24: I_2xx - moment of inertia (xx) of link lwr_2_link
|   0.00000000|  -0.00226915|-0.0023|  -22.7|        |#25: I_2xy - moment of inertia (xy) of link lwr_2_link
|   0.00000000|  -0.00622061|-0.0062|  -62.2|        |#26: I_2xz - moment of inertia (xz) of link lwr_2_link
|   0.01728000|   0.04408216| 0.0268|  155.1|        |#27: I_2yy - moment of inertia (yy) of link lwr_2_link
|  -0.01134000|  -0.02027960|-0.0089|   78.8|        |#28: I_2yz - moment of inertia (yz) of link lwr_2_link
|   0.02574000|   0.01364303|-0.0121|  -47.0|        |#29: I_2zz - moment of inertia (zz) of link lwr_2_link
|   2.70000000|   2.74891581| 0.0489|    1.8|mA      |#30: m_3 - mass of link lwr_3_link
|   0.00000000|  -0.03303629|-0.0330| -330.4|hull    |#31: c_3x - first moment of mass (x) of link lwr_3_link
|   0.16200000|   0.02369860|-0.1383|  -85.4|hull    |#32: c_3y - first moment of mass (y) of link lwr_3_link
|   0.35100000|   0.00580811|-0.3452|  -98.3|hull    |#33: c_3z - first moment of mass (z) of link lwr_3_link
|   0.07380000|   0.94524143| 0.8714| 1180.8|        |#34: I_3xx - moment of inertia (xx) of link lwr_3_link
|   0.00000000|   0.09173686| 0.0917|  917.4|        |#35: I_3xy - moment of inertia (xy) of link lwr_3_link
|   0.00000000|   0.00584161| 0.0058|   58.4|        |#36: I_3xz - moment of inertia (xz) of link lwr_3_link
|   0.04968000|   0.01051394|-0.0392|  -78.8|        |#37: I_3yy - moment of inertia (yy) of link lwr_3_link
|  -0.02106000|   0.00049973| 0.0216| -102.4|        |#38: I_3yz - moment of inertia (yz) of link lwr_3_link
|   0.02574000|   0.00066744|-0.0251|  -97.4|        |#39: I_3zz - moment of inertia (zz) of link lwr_3_link
|   2.70000000|   3.19045098| 0.4905|   18.2|mA      |#40: m_4 - mass of link lwr_4_link
|   0.00000000|   0.00962424| 0.0096|   96.2|hull    |#41: c_4x - first moment of mass (x) of link lwr_4_link
|  -0.16200000|   0.05386036| 0.2159| -133.2|hull    |#42: c_4y - first moment of mass (y) of link lwr_4_link
|   0.18900000|   0.46384000| 0.2748|  145.4|hull    |#43: c_4z - first moment of mass (z) of link lwr_4_link
|   0.04140000|   0.07176238| 0.0304|   73.3|        |#44: I_4xx - moment of inertia (xx) of link lwr_4_link
|   0.00000000|  -0.02078452|-0.0208| -207.8|        |#45: I_4xy - moment of inertia (xy) of link lwr_4_link
|   0.00000000|  -0.00083147|-0.0008|   -8.3|        |#46: I_4xz - moment of inertia (xz) of link lwr_4_link
|   0.01728000|   0.20657605| 0.1893| 1095.5|        |#47: I_4yy - moment of inertia (yy) of link lwr_4_link
|   0.01134000|  -0.01203690|-0.0234| -206.1|        |#48: I_4yz - moment of inertia (yz) of link lwr_4_link
|   0.02574000|   0.00110839|-0.0246|  -95.7|        |#49: I_4zz - moment of inertia (zz) of link lwr_4_link
|   1.70000000|   1.94081370| 0.2408|   14.2|mA      |#50: m_5 - mass of link lwr_5_link
|   0.00000000|  -0.00272555|-0.0027|  -27.3|hull    |#51: c_5x - first moment of mass (x) of link lwr_5_link
|   0.00000000|  -0.09923248|-0.0992| -992.3|hull    |#52: c_5y - first moment of mass (y) of link lwr_5_link
|   0.21080000|   0.00094548|-0.2099|  -99.6|hull    |#53: c_5z - first moment of mass (z) of link lwr_5_link
|   0.03689227|   0.07334665| 0.0365|   98.8|        |#54: I_5xx - moment of inertia (xx) of link lwr_5_link
|   0.00000000|  -0.00159872|-0.0016|  -16.0|        |#55: I_5xy - moment of inertia (xy) of link lwr_5_link
|   0.00000000|   0.00027793| 0.0003|    2.8|        |#56: I_5xz - moment of inertia (xz) of link lwr_5_link
|   0.02868920|   0.05112050| 0.0224|   78.2|        |#57: I_5yy - moment of inertia (yy) of link lwr_5_link
|   0.00000000|  -0.03269277|-0.0327| -326.9|        |#58: I_5yz - moment of inertia (yz) of link lwr_5_link
|   0.00922307|   0.02617024| 0.0169|  183.7|        |#59: I_5zz - moment of inertia (zz) of link lwr_5_link
|   1.60000000|   0.80534947|-0.7947|  -49.7|mA      |#60: m_6 - mass of link lwr_6_link
|   0.00000000|   0.02013234| 0.0201|  201.3|hull    |#61: c_6x - first moment of mass (x) of link lwr_6_link
|   0.00000000|   0.05522807| 0.0552|  552.3|hull    |#62: c_6y - first moment of mass (y) of link lwr_6_link
|   0.10000000|   0.01739372|-0.0826|  -82.6|hull    |#63: c_6z - first moment of mass (z) of link lwr_6_link
|   0.01041667|   0.00459147|-0.0058|  -55.9|        |#64: I_6xx - moment of inertia (xx) of link lwr_6_link
|   0.00000000|  -0.00054817|-0.0005|   -5.5|        |#65: I_6xy - moment of inertia (xy) of link lwr_6_link
|   0.00000000|  -0.00151613|-0.0015|  -15.2|        |#66: I_6xz - moment of inertia (xz) of link lwr_6_link
|   0.01041667|   0.00355936|-0.0069|  -65.8|        |#67: I_6yy - moment of inertia (yy) of link lwr_6_link
|   0.00000000|  -0.00449026|-0.0045|  -44.9|        |#68: I_6yz - moment of inertia (yz) of link lwr_6_link
|   0.00416667|   0.00875268| 0.0046|  110.1|        |#69: I_6zz - moment of inertia (zz) of link lwr_6_link
|   0.30000000|   0.41345440| 0.1135|   37.8|mA      |#70: m_7 - mass of link lwr_7_link
|   0.00000000|  -0.00215891|-0.0022|  -21.6|hull    |#71: c_7x - first moment of mass (x) of link lwr_7_link
|   0.00000000|  -0.01631895|-0.0163| -163.2|hull    |#72: c_7y - first moment of mass (y) of link lwr_7_link
|   0.00000000|  -0.00457308|-0.0046|  -45.7|hull    |#73: c_7z - first moment of mass (z) of link lwr_7_link
|   0.05000000|   0.01390119|-0.0361|  -72.2|        |#74: I_7xx - moment of inertia (xx) of link lwr_7_link
|   0.00000000|   0.01105989| 0.0111|  110.6|        |#75: I_7xy - moment of inertia (xy) of link lwr_7_link
|   0.00000000|   0.00705626| 0.0071|   70.6|        |#76: I_7xz - moment of inertia (xz) of link lwr_7_link
|   0.05000000|   0.01849256|-0.0315|  -63.0|        |#77: I_7yy - moment of inertia (yy) of link lwr_7_link
|   0.00000000|   0.01073201| 0.0107|  107.3|        |#78: I_7yz - moment of inertia (yz) of link lwr_7_link
|   0.05000000|   0.00718351|-0.0428|  -85.6|        |#79: I_7zz - moment of inertia (zz) of link lwr_7_link
|   0.00000000|   0.87560076| 0.8756| 8756.0|        |#80: Fc_0 - Constant friction / offset of joint lwr_0_joint
|   0.00000000|   0.31504493| 0.3150| 3150.4|        |#81: Fc_1 - Constant friction / offset of joint lwr_1_joint
|   0.00000000|  -0.04823953|-0.0482| -482.4|        |#82: Fc_2 - Constant friction / offset of joint lwr_2_joint
|   0.00000000|  -0.13318455|-0.1332|-1331.8|        |#83: Fc_3 - Constant friction / offset of joint lwr_3_joint
|   0.00000000|  -0.62334133|-0.6233|-6233.4|        |#84: Fc_4 - Constant friction / offset of joint lwr_4_joint
|   0.00000000|   0.43222213| 0.4322| 4322.2|        |#85: Fc_5 - Constant friction / offset of joint lwr_5_joint
|   0.00000000|  -0.08610412|-0.0861| -861.0|        |#86: Fc_6 - Constant friction / offset of joint lwr_6_joint
|   1.00000000|   1.33796412| 0.3380|   33.8|>0      |#87: Fv_0 - Velocity dep. friction joint lwr_0_joint
|   1.00000000|   0.11698830|-0.8830|  -88.3|>0      |#88: Fv_1 - Velocity dep. friction joint lwr_1_joint
|   1.00000000|   0.61170200|-0.3883|  -38.8|>0      |#89: Fv_2 - Velocity dep. friction joint lwr_2_joint
|   1.00000000|   1.51162391| 0.5116|   51.2|>0      |#90: Fv_3 - Velocity dep. friction joint lwr_3_joint
|   1.00000000|   0.38340285|-0.6166|  -61.7|>0      |#91: Fv_4 - Velocity dep. friction joint lwr_4_joint
|   1.00000000|   0.26718543|-0.7328|  -73.3|>0      |#92: Fv_5 - Velocity dep. friction joint lwr_5_joint
|   1.00000000|   0.04957313|-0.9504|  -95.0|>0      |#93: Fv_6 - Velocity dep. friction joint lwr_6_joint


Parameters
Estimated overall mass: 16.30 kg vs. a priori 16.0 kg
A priori parameters are physical consistent
Identified parameters are physical consistent
Squared distance of identifiable std parameter vectors to a priori: 6.73

Torque prediction errors
Relative mean residual error: 3.34% vs. A priori: 10.53%
Absolute mean residual error: 0.79 vs. A priori: 2.56
NRMS of residual error: 0.19% vs. A priori: 0.61%

Relative validation error: 4.13%
Absolute validation error: 1.04 Nm
NRMS validation error: 0.23%
```

The table columns show parameters for A priori (URDF), Identified and the absolute change between them. There also is a percentual difference value (%e) that is given in relation to the magnitude of the a priori value.

The different estimation error measures that are given are

Absolute mean error:
The mean over the error vector norms for each joint.

Relative mean error: 
The absolute mean error normalized with the norm of the measured data vectors.

Normalized root mean square (NRMS) error:
The square root of the mean over the joints of the squared error, normalized by the possible torque range of each joint (as given in the URDF).

An assessment of the quality of the result should be made through the combination of torque prediction accuracy, validation accuracy (ideally multiple different validation trajcetories) and also the estimated torque curve shapes compared to the measured torques.
