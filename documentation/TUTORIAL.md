# Tutorial

The goal of identification is usually to find dynamic model parameters from measurements of motions. Ideally, some previous parameters are available e.g. from a CAD model which also gives the kinematic model. In the following, an example identification is carried out for the Kuka LWR4+ robot. The options for all steps of this task are held within a configuration file in the config/ dir.

1. copy an existing .yaml configuration file and customize it for your setup with a text editor.
`cp config/kuka_lwr.yaml config/example.yaml`

2. Use the trajectory.py script to generate an optimal exciting trajectory (only fixed base at the moment). The corresponding options in the configuration should be set (for the case of the LWR4+ that is done) and optionally supply a world urdf file that includes the ground and objects that the robot might collide with, e.g. a table. The optimization will simulate each trajectory and check for all constraints to be met while minimizing the condition number of the dynamics regressor. This might take a while depending on the degrees of freedom. You can prefix the call with `mpirun -n <n>` to parallelize this. An output file containing the found parameters of the trajectory will be saved.
`./trajectory.py --config configs/example.yaml --model model/example.urdf --world model/world.urdf`

3. Get joint torque measurements for the trajectory from your robotic system, if suitable by using the excite.py script. It will load the previously created trajectory file and move the robot through the specified module (in the config file). Alternatively, simulation can be enabled to simulate the torques using the supplied model parameters. If necessary, look at the existing modules and write a custom one for your communication method. After retrieving the measurements, filtering as well as deriving velocity and acceleration is done and is saved to a measurements file. If you are using other means of motion control and data recording and don't use the excite.py script, the data needs to be filtered and saved to a numpy data file that has the expected data fields (see README.md in excitation/). There is also the **csv2npz.py** script that loads raw data from csv text files, preprocesses them with the same filtering and writes to the container format (you'll need to customize it for the columns in your csv file etc.).
In this example for the LWR4+, we simply simulate the trajectory file to receive a measurements file.
`./excite.py --model model/example.urdf --config configs/example.yaml --plot \`
`--trajectory model/example.urdf.trajectory.npz --filename measurements.npz`

4. Finally, run identify.py with the measurements file and again the
  kinematic model in a .urdf file with the a priori parameters. These parameters don't have to be physical consistent but it's recommended (they should be when they come from a CAD system). The constrained optimization for identification Optionally you can supply an output .urdf file path to which the input urdf with exchanged
identified parameters is written. Another measurements file can be supplied for validation.
`./identify.py --config configs/example.yaml  --model model/example.urdf --measurements \`
`measurements.npz --verify measurements_2.npz --output model/example_identified.urdf`

The output html file in output/ should look similar to the following:

```Linear (relative to Frame) Standard Parameters
|A priori     |Ident        |Change |%e     |Constr  |Description
|   1.20000000|   1.20000000| 0.0000|    0.0|nID     |#0: m_0 - mass of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#1: c_0x - x component of first moment of mass of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#2: c_0y - y component of first moment of mass of link lwr_base_link
|   0.06600000|   0.06600000| 0.0000|    0.0|nID     |#3: c_0z - z component of first moment of mass of link lwr_base_link
|   0.00592000|   0.00592000| 0.0000|    0.0|nID     |#4: I_0xx - xx component of inertia matrix of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#5: I_0xy - xy component of inertia matrix of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#6: I_0xz - xz component of inertia matrix of link lwr_base_link
|   0.00579000|   0.00579000| 0.0000|    0.0|nID     |#7: I_0yy - yy component of inertia matrix of link lwr_base_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#8: I_0yz - yz component of inertia matrix of link lwr_base_link
|   0.00229000|   0.00229000| 0.0000|    0.0|nID     |#9: I_0zz - zz component of inertia matrix of link lwr_base_link
|   2.00000000|   2.06844561| 0.0684|    3.4|mA nID  |#10: m_1 - mass of link lwr_1_link
|   0.00000000|  -0.00000000|-0.0000|   -0.0|hull nID|#11: c_1x - x component of first moment of mass of link lwr_1_link
|   0.00000000|  -0.00000013|-0.0000|   -0.0|hull nID|#12: c_1y - y component of first moment of mass of link lwr_1_link
|   0.26000000|   0.26000222| 0.0000|    0.0|hull nID|#13: c_1z - z component of first moment of mass of link lwr_1_link
|   0.04746670|   0.04746256|-0.0000|   -0.0|nID     |#14: I_1xx - xx component of inertia matrix of link lwr_1_link
|   0.00000000|   0.00000000| 0.0000|    0.0|nID     |#15: I_1xy - xy component of inertia matrix of link lwr_1_link
|   0.00000000|  -0.00000000|-0.0000|   -0.0|nID     |#16: I_1xz - xz component of inertia matrix of link lwr_1_link
|   0.04566670|   0.04566244|-0.0000|   -0.0|nID     |#17: I_1yy - yy component of inertia matrix of link lwr_1_link
|   0.00000000|   0.00000002| 0.0000|    0.0|nID     |#18: I_1yz - yz component of inertia matrix of link lwr_1_link
|   0.00300000|   0.00000099|-0.0030| -100.0|        |#19: I_1zz - zz component of inertia matrix of link lwr_1_link
|   2.00000000|   2.07648644| 0.0765|    3.8|mA nID  |#20: m_2 - mass of link lwr_2_link
|   0.00000000|   0.07379650| 0.0738|  738.0|hull    |#21: c_2x - x component of first moment of mass of link lwr_2_link
|  -0.12000000|  -0.10987083| 0.0101|   -8.4|hull nID|#22: c_2y - y component of first moment of mass of link lwr_2_link
|   0.14000000|  -0.12458821|-0.2646| -189.0|hull    |#23: c_2z - z component of first moment of mass of link lwr_2_link
|   0.03066670|   0.03548297| 0.0048|   15.7|        |#24: I_2xx - xx component of inertia matrix of link lwr_2_link
|   0.00000000|  -0.02325258|-0.0233| -232.5|        |#25: I_2xy - xy component of inertia matrix of link lwr_2_link
|   0.00000000|  -0.03237258|-0.0324| -323.7|        |#26: I_2xz - xz component of inertia matrix of link lwr_2_link
|   0.02166670|   0.04333062| 0.0217|  100.0|        |#27: I_2yy - yy component of inertia matrix of link lwr_2_link
|   0.00840000|   0.03843948| 0.0300|  357.6|        |#28: I_2yz - yz component of inertia matrix of link lwr_2_link
|   0.01020000|   0.06945869| 0.0593|  581.0|        |#29: I_2zz - zz component of inertia matrix of link lwr_2_link
|   2.00000000|   2.99999901| 1.0000|   50.0|mA      |#30: m_3 - mass of link lwr_3_link
|   0.00000000|   0.01939185| 0.0194|  193.9|hull    |#31: c_3x - x component of first moment of mass of link lwr_3_link
|  -0.12000000|  -0.02599430| 0.0940|  -78.3|hull    |#32: c_3y - y component of first moment of mass of link lwr_3_link
|   0.26000000|   0.00000091|-0.2600| -100.0|hull    |#33: c_3z - z component of first moment of mass of link lwr_3_link
|   0.05466670|   0.00022623|-0.0544|  -99.6|        |#34: I_3xx - xx component of inertia matrix of link lwr_3_link
|   0.00000000|   0.00016803| 0.0002|    1.7|        |#35: I_3xy - xy component of inertia matrix of link lwr_3_link
|   0.00000000|   0.00000000| 0.0000|    0.0|        |#36: I_3xz - xz component of inertia matrix of link lwr_3_link
|   0.04566670|   0.00012636|-0.0455|  -99.7|        |#37: I_3yy - yy component of inertia matrix of link lwr_3_link
|   0.01560000|   0.00000003|-0.0156| -100.0|        |#38: I_3yz - yz component of inertia matrix of link lwr_3_link
|   0.01020000|   0.00035159|-0.0098|  -96.6|        |#39: I_3zz - zz component of inertia matrix of link lwr_3_link
|   2.00000000|   2.99999899| 1.0000|   50.0|mA      |#40: m_4 - mass of link lwr_4_link
|   0.00000000|  -0.00872751|-0.0087|  -87.3|hull    |#41: c_4x - x component of first moment of mass of link lwr_4_link
|   0.12000000|  -0.03327066|-0.1533| -127.7|hull    |#42: c_4y - y component of first moment of mass of link lwr_4_link
|   0.14000000|   0.16651652| 0.0265|   18.9|hull    |#43: c_4z - z component of first moment of mass of link lwr_4_link
|   0.03066670|   0.04825581| 0.0176|   57.4|        |#44: I_4xx - xx component of inertia matrix of link lwr_4_link
|   0.00000000|   0.02186680| 0.0219|  218.7|        |#45: I_4xy - xy component of inertia matrix of link lwr_4_link
|   0.00000000|  -0.01160328|-0.0116| -116.0|        |#46: I_4xz - xz component of inertia matrix of link lwr_4_link
|   0.02166670|   0.02175239| 0.0001|    0.4|        |#47: I_4yy - yy component of inertia matrix of link lwr_4_link
|  -0.00840000|  -0.00502356| 0.0034|  -40.2|        |#48: I_4yz - yz component of inertia matrix of link lwr_4_link
|   0.01020000|   0.00417643|-0.0060|  -59.1|        |#49: I_4zz - zz component of inertia matrix of link lwr_4_link
|   2.00000000|   2.99999899| 1.0000|   50.0|mA      |#50: m_5 - mass of link lwr_5_link
|   0.00000000|   0.01600572| 0.0160|  160.1|hull    |#51: c_5x - x component of first moment of mass of link lwr_5_link
|   0.00000000|   0.04725954| 0.0473|  472.6|hull    |#52: c_5y - y component of first moment of mass of link lwr_5_link
|   0.24800000|   0.00000103|-0.2480| -100.0|hull    |#53: c_5z - z component of first moment of mass of link lwr_5_link
|   0.04340270|   0.02762440|-0.0158|  -36.4|        |#54: I_5xx - xx component of inertia matrix of link lwr_5_link
|   0.00000000|   0.02232973| 0.0223|  223.3|        |#55: I_5xy - xy component of inertia matrix of link lwr_5_link
|   0.00000000|   0.00341760| 0.0034|   34.2|        |#56: I_5xz - xz component of inertia matrix of link lwr_5_link
|   0.04160270|   0.01905817|-0.0225|  -54.2|        |#57: I_5yy - yy component of inertia matrix of link lwr_5_link
|   0.00000000|   0.00287123| 0.0029|   28.7|        |#58: I_5yz - yz component of inertia matrix of link lwr_5_link
|   0.00300000|   0.00126542|-0.0017|  -57.8|        |#59: I_5zz - zz component of inertia matrix of link lwr_5_link
|   1.00000000|   1.30507296| 0.3051|   30.5|mA      |#60: m_6 - mass of link lwr_6_link
|   0.00000000|  -0.01258784|-0.0126| -125.9|hull    |#61: c_6x - x component of first moment of mass of link lwr_6_link
|   0.00000000|  -0.00356337|-0.0036|  -35.6|hull    |#62: c_6y - y component of first moment of mass of link lwr_6_link
|   0.00000000|   0.00434072| 0.0043|   43.4|hull    |#63: c_6z - z component of first moment of mass of link lwr_6_link
|   0.00260417|   0.02225876| 0.0197|  754.7|        |#64: I_6xx - xx component of inertia matrix of link lwr_6_link
|   0.00000000|  -0.00568387|-0.0057|  -56.8|        |#65: I_6xy - xy component of inertia matrix of link lwr_6_link
|   0.00000000|   0.00853369| 0.0085|   85.3|        |#66: I_6xz - xz component of inertia matrix of link lwr_6_link
|   0.00260417|   0.00157237|-0.0010|  -39.6|        |#67: I_6yy - yy component of inertia matrix of link lwr_6_link
|   0.00000000|  -0.00214590|-0.0021|  -21.5|        |#68: I_6yz - yz component of inertia matrix of link lwr_6_link
|   0.00260417|   0.00337547| 0.0008|   29.6|        |#69: I_6zz - zz component of inertia matrix of link lwr_6_link
|   0.20000000|   0.29999899| 0.1000|   50.0|mA      |#70: m_7 - mass of link lwr_7_link
|   0.00000000|  -0.00183000|-0.0018|  -18.3|hull    |#71: c_7x - x component of first moment of mass of link lwr_7_link
|   0.00000000|   0.01195798| 0.0120|  119.6|hull    |#72: c_7y - y component of first moment of mass of link lwr_7_link
|   0.00000000|  -0.00929895|-0.0093|  -93.0|hull    |#73: c_7z - z component of first moment of mass of link lwr_7_link
|   0.00006667|   0.00311685| 0.0031| 4575.3|        |#74: I_7xx - xx component of inertia matrix of link lwr_7_link
|   0.00000000|   0.00860004| 0.0086|   86.0|        |#75: I_7xy - xy component of inertia matrix of link lwr_7_link
|   0.00000000|   0.00251793| 0.0025|   25.2|        |#76: I_7xz - xz component of inertia matrix of link lwr_7_link
|   0.00006667|   0.03122864| 0.0312|46742.9|        |#77: I_7yy - yy component of inertia matrix of link lwr_7_link
|   0.00000000|   0.00970905| 0.0097|   97.1|        |#78: I_7yz - yz component of inertia matrix of link lwr_7_link
|   0.00012000|   0.00330841| 0.0032| 2657.0|        |#79: I_7zz - zz component of inertia matrix of link lwr_7_link
|   0.00000000|   0.94874066| 0.9487| 9487.4|        |#80: Fc_0 - Constant friction / offset of joint lwr_0_joint
|   0.00000000|   0.20681690| 0.2068| 2068.2|        |#81: Fc_1 - Constant friction / offset of joint lwr_1_joint
|   0.00000000|   0.05280955| 0.0528|  528.1|        |#82: Fc_2 - Constant friction / offset of joint lwr_2_joint
|   0.00000000|  -0.37840295|-0.3784|-3784.0|        |#83: Fc_3 - Constant friction / offset of joint lwr_3_joint
|   0.00000000|  -0.47408009|-0.4741|-4740.8|        |#84: Fc_4 - Constant friction / offset of joint lwr_4_joint
|   0.00000000|  -1.19962543|-1.1996|-11996.3|        |#85: Fc_5 - Constant friction / offset of joint lwr_5_joint
|   0.00000000|  -0.27127420|-0.2713|-2712.7|        |#86: Fc_6 - Constant friction / offset of joint lwr_6_joint
|   5.00000000|   0.67127157|-4.3287|  -86.6|>0      |#87: Fv_0 - Velocity dep. friction joint lwr_0_joint
|   5.00000000|   0.69485924|-4.3051|  -86.1|>0      |#88: Fv_1 - Velocity dep. friction joint lwr_1_joint
|   3.00000000|   0.31338770|-2.6866|  -89.6|>0      |#89: Fv_2 - Velocity dep. friction joint lwr_2_joint
|   3.00000000|   0.85107198|-2.1489|  -71.6|>0      |#90: Fv_3 - Velocity dep. friction joint lwr_3_joint
|   1.00000000|   0.81920099|-0.1808|  -18.1|>0      |#91: Fv_4 - Velocity dep. friction joint lwr_4_joint
|   1.00000000|   0.22270897|-0.7773|  -77.7|>0      |#92: Fv_5 - Velocity dep. friction joint lwr_5_joint
|   1.00000000|   0.18990165|-0.8101|  -81.0|>0      |#93: Fv_6 - Velocity dep. friction joint lwr_6_joint


Parameters
Estimated overall mass: 15.950000992 kg vs. a priori 12.4 kg
A priori parameters are physical consistent
Identified parameters are physical consistent
Squared distance of identifiable std parameter vectors to a priori: 56.5871450393

Torque prediction errors
Relative mean residual error: 4.56636279822% vs. A priori: 27.3742947535%
Absolute mean residual error: 1.13770446029 vs. A priori: 6.74491747194
NRMS of residual error: 0.238283958734% vs. A priori: 1.1176836913%

Relative validation error: 10.0111137844%
Absolute validation error: 2.19425241416 Nm
NRMS validation error: 0.563549164927%
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


