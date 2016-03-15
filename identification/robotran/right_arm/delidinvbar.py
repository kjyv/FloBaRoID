#	ROBOTRAN - Version 6.3B (build : 6 mars 2008]
#	==> Model: Identification matrix (inverse model)
#	==> Formalism: Symbolic Derivation of Recursive Barycentric Parameters (from Standard)
#	==> Nbody: 9

import numpy as np
from math import sin, cos

def delidinvbar(delta_out, m, l, In, d):
    # delta_out is a vector of 48 values which the basis inertial parameters will be written to.
    # l contains the positions (3x1) of the center of mass of each body relative to its (parent) joint. (()()
    # LShp ...) so overall (4x10) with matlab indexing
    # m are the masses for each of the links (child after the joint)
    # In contains the body inertia matrix represented as a 9x1 vector (although only 6 elements are
    # used due to the symmetry) for each body. It is represented relative to the center of mass.
    # d contains the position of the joints relative to their parent joints (kinematic parameters) (3 x N_DOF)

    # Note, the "0" elements of vectors (vec[0]) are always set to zero. We use a matlab-like convention
    # with array elements starting at index 1 (vec[1]).
    # (not the outputs though)

    #relative joint positions, parent to child (as in urdf)
    #() () () LShSag LShLat LShYaw LElbj LForearmPlate LWrj1 LWrj2
    """
    d = np.array(
        [
            [0.,  0.,  0.,    0.,        0.,        0.,        0.,        0.,        0.,        0.   ],
            [0.,  0.,  0.,    0.045646,  0.,        0.,        0.036,    -0.075,     0.,        0.   ],
            [0.,  0.,  0.,    0.219164,  0.116568,  0.,        0.,        0.,        0.,        0.   ],
            [0.,  0.,  0.,    0.001531,  0.03364,  -0.222,    -0.15,     -0.1955,    0.,       -0.092 ],
        ]
    )
    """

    # Barycentric Masses

    mb1 = m[3]+m[4]+m[5]+m[6]+m[7]+m[8]+m[9]
    mb4 = m[4]+m[5]+m[6]+m[7]+m[8]+m[9]
    mb5 = m[5]+m[6]+m[7]+m[8]+m[9]
    mb6 = m[6]+m[7]+m[8]+m[9]
    mb7 = m[7]+m[8]+m[9]
    mb8 = m[8]+m[9]
    mb9 = m[9]

    # Barycentric Vectors

    b31 = m[3]*l[1,3]
    b32 = m[3]*l[2,3]+mb4*d[2,4]
    b33 = m[3]*l[3,3]+mb4*d[3,4]
    b41 = m[4]*l[1,4]
    b42 = m[4]*l[2,4]
    b43 = m[4]*l[3,4]+mb5*d[3,5]
    b51 = m[5]*l[1,5]+mb6*d[1,6]
    b52 = m[5]*l[2,5]
    b53 = m[5]*l[3,5]+mb6*d[3,6]
    b61 = m[6]*l[1,6]+mb7*d[1,7]
    b62 = m[6]*l[2,6]
    b63 = m[6]*l[3,6]+mb7*d[3,7]
    b71 = m[7]*l[1,7]
    b72 = m[7]*l[2,7]
    b73 = m[7]*l[3,7]
    b81 = m[8]*l[1,8]
    b82 = m[8]*l[2,8]
    b83 = m[8]*l[3,8]+mb9*d[3,9]
    b91 = m[9]*l[1,9]
    b92 = m[9]*l[2,9]
    b93 = m[9]*l[3,9]

    # Barycentric Tensors

    K311 = In[1,3]+m[3]*l[2,3]*l[2,3]+m[3]*l[3,3]*l[3,3]+mb4*d[2,4]*d[2,4]+mb4*d[3,4]*d[3,4]
    K312 = In[2,3]-m[3]*l[1,3]*l[2,3]
    K313 = In[3,3]-m[3]*l[1,3]*l[3,3]
    K322 = In[5,3]+m[3]*l[1,3]*l[1,3]+m[3]*l[3,3]*l[3,3]+mb4*d[3,4]*d[3,4]
    K323 = In[6,3]-m[3]*l[2,3]*l[3,3]-mb4*d[2,4]*d[3,4]
    K333 = In[9,3]+m[3]*l[1,3]*l[1,3]+m[3]*l[2,3]*l[2,3]+mb4*d[2,4]*d[2,4]
    K411 = In[1,4]+m[4]*l[2,4]*l[2,4]+m[4]*l[3,4]*l[3,4]+mb5*d[3,5]*d[3,5]
    K412 = In[2,4]-m[4]*l[1,4]*l[2,4]
    K413 = In[3,4]-m[4]*l[1,4]*l[3,4]
    K422 = In[5,4]+m[4]*l[1,4]*l[1,4]+m[4]*l[3,4]*l[3,4]+mb5*d[3,5]*d[3,5]
    K423 = In[6,4]-m[4]*l[2,4]*l[3,4]
    K433 = In[9,4]+m[4]*l[1,4]*l[1,4]+m[4]*l[2,4]*l[2,4]
    K511 = In[1,5]+m[5]*l[2,5]*l[2,5]+m[5]*l[3,5]*l[3,5]+mb6*d[3,6]*d[3,6]
    K512 = In[2,5]-m[5]*l[1,5]*l[2,5]
    K513 = In[3,5]-m[5]*l[1,5]*l[3,5]-mb6*d[1,6]*d[3,6]
    K522 = In[5,5]+m[5]*l[1,5]*l[1,5]+m[5]*l[3,5]*l[3,5]+mb6*d[1,6]*d[1,6]+mb6*d[3,6]*d[3,6]
    K523 = In[6,5]-m[5]*l[2,5]*l[3,5]
    K533 = In[9,5]+m[5]*l[1,5]*l[1,5]+m[5]*l[2,5]*l[2,5]+mb6*d[1,6]*d[1,6]
    K611 = In[1,6]+m[6]*l[2,6]*l[2,6]+m[6]*l[3,6]*l[3,6]+mb7*d[3,7]*d[3,7]
    K612 = In[2,6]-m[6]*l[1,6]*l[2,6]
    K613 = In[3,6]-m[6]*l[1,6]*l[3,6]-mb7*d[1,7]*d[3,7]
    K622 = In[5,6]+m[6]*l[1,6]*l[1,6]+m[6]*l[3,6]*l[3,6]+mb7*d[1,7]*d[1,7]+mb7*d[3,7]*d[3,7]
    K623 = In[6,6]-m[6]*l[2,6]*l[3,6]
    K633 = In[9,6]+m[6]*l[1,6]*l[1,6]+m[6]*l[2,6]*l[2,6]+mb7*d[1,7]*d[1,7]
    K711 = In[1,7]+m[7]*l[2,7]*l[2,7]+m[7]*l[3,7]*l[3,7]
    K712 = In[2,7]-m[7]*l[1,7]*l[2,7]
    K713 = In[3,7]-m[7]*l[1,7]*l[3,7]
    K722 = In[5,7]+m[7]*l[1,7]*l[1,7]+m[7]*l[3,7]*l[3,7]
    K723 = In[6,7]-m[7]*l[2,7]*l[3,7]
    K733 = In[9,7]+m[7]*l[1,7]*l[1,7]+m[7]*l[2,7]*l[2,7]
    K811 = In[1,8]+m[8]*l[2,8]*l[2,8]+m[8]*l[3,8]*l[3,8]+mb9*d[3,9]*d[3,9]
    K812 = In[2,8]-m[8]*l[1,8]*l[2,8]
    K813 = In[3,8]-m[8]*l[1,8]*l[3,8]
    K822 = In[5,8]+m[8]*l[1,8]*l[1,8]+m[8]*l[3,8]*l[3,8]+mb9*d[3,9]*d[3,9]
    K823 = In[6,8]-m[8]*l[2,8]*l[3,8]
    K833 = In[9,8]+m[8]*l[1,8]*l[1,8]+m[8]*l[2,8]*l[2,8]
    K911 = In[1,9]+m[9]*l[2,9]*l[2,9]+m[9]*l[3,9]*l[3,9]
    K912 = In[2,9]-m[9]*l[1,9]*l[2,9]
    K913 = In[3,9]-m[9]*l[1,9]*l[3,9]
    K922 = In[5,9]+m[9]*l[1,9]*l[1,9]+m[9]*l[3,9]*l[3,9]
    K923 = In[6,9]-m[9]*l[2,9]*l[3,9]
    K933 = In[9,9]+m[9]*l[1,9]*l[1,9]+m[9]*l[2,9]*l[2,9]

    # Barycentric Tensors: elements Ks, Kd

    Ks9 = 0.500*(K922+K933)
    Kd9 = 0.500*(K922-K933)
    Ks8 = 0.500*(K811+K833+Ks9)
    Kd8 = 0.500*(-K811+K833+Ks9)
    Ks7 = 0.500*(K711+K722+Ks8)
    Kd7 = 0.500*(K711-K722+Ks8)
    Ks6 = 0.500*(K611+K633+Ks7+2.000*b73*d[3,7])
    Kd6 = 0.500*(-K611+K633-Ks7-2.000*b73*d[3,7])
    Ks5 = 0.500*(K511+K522+Ks6)
    Kd5 = 0.500*(K511-K522+Ks6)
    Ks4 = 0.500*(K422+K433+Ks5+2.000*b53*d[3,5])
    Kd4 = 0.500*(K422-K433+Ks5+2.000*b53*d[3,5])
    Ks3 = 0.500*(K311+K333+Ks4)
    Kd3 = 0.500*(-K311+K333+Ks4)

    # Reduced Barycentric Vectors

    br31 = b31+b41
    br43 = b43+b53
    br52 = b52+b62
    br63 = b63+b73
    br72 = b72+b82
    br81 = b81+b91

    # Reduced Barycentric Tensors

    Kr312 = K312-b41*d[2,4]
    Kr313 = K313-b41*d[3,4]
    Kr322 = K322+Ks4
    Kr411 = K411+Ks5+2.000*b53*d[3,5]
    Kr512 = K512-b62*d[1,6]
    Kr523 = K523-b62*d[3,6]
    Kr533 = K533+Ks6
    Kr613 = K613-b73*d[1,7]
    Kr622 = K622+Ks7+2.000*b73*d[3,7]
    Kr733 = K733+Ks8
    Kr813 = K813-b91*d[3,9]
    Kr822 = K822+Ks9

    # Definition of Minimal Set of Dynamical Paramaters

    delta_out[0] = b32
    delta_out[1] = br31
    delta_out[2] = b33
    delta_out[3] = Kr312
    delta_out[4] = Kr322
    delta_out[5] = K323
    delta_out[6] = b42
    delta_out[7] = br43
    delta_out[8] = Kr411
    delta_out[9] = K412
    delta_out[10] = K413
    delta_out[11] = Kd4
    delta_out[12] = K423
    delta_out[13] = b51
    delta_out[14] = br52
    delta_out[15] = Kd5
    delta_out[16] = Kr512
    delta_out[17] = K513
    delta_out[18] = Kr523
    delta_out[19] = Kr533
    delta_out[20] = b61
    delta_out[21] = br63
    delta_out[22] = K612
    delta_out[23] = Kr613
    delta_out[24] = Kr622
    delta_out[25] = K623
    delta_out[26] = Kd6
    delta_out[27] = b71
    delta_out[28] = br72
    delta_out[29] = Kd7
    delta_out[30] = K712
    delta_out[31] = K713
    delta_out[32] = K723
    delta_out[33] = Kr733
    delta_out[34] = br81
    delta_out[35] = b83
    delta_out[36] = K812
    delta_out[37] = Kr813
    delta_out[38] = Kr822
    delta_out[39] = K823
    delta_out[40] = Kd8
    delta_out[41] = b92
    delta_out[42] = b93
    delta_out[43] = K911
    delta_out[44] = K912
    delta_out[45] = K913
    delta_out[46] = Kd9
    delta_out[47] = K923

