#	ROBOTRAN - Version 6.3B (build : 6 mars 2008]
#	==> Model: Identification matrix (inverse model)
#	==> Formalism: Symbolic Derivation of Recursive Barycentric Parameters (from Standard)
#	==> Nbody: 9

import numpy as np
from math import sin, cos
from get_bar import get_bar

N_ACTUATED_JONTS = 7
N_BLOCKED_JONTS = 2
N_JOINTS = (N_ACTUATED_JONTS + N_BLOCKED_JONTS)
N_DELTA = 48

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

    if d is None:
        d = np.array(
            [
                [  0.,	 0.,	        0., 	0.,    	0.,	0.,     0.,	0.,		0.,	0.],
                [  0.,	 0.045646,      0., 	0.,     0.,     0.,  0.036,	-0.075,		0.,	0.],
                [  0.,	 0.219164,	      0., 	0., 0.1091367,	0., 	0.,	0.,		0.,	0.],
                [  0.,	 0.001531,      0., 	0., -0.053,	-0.222,  -0.15,	   -0.1955,    0.,	-0.092]
            ]
        )

    # bodies CoM
    if l is None:
        l = np.array(
            [
                [0.,0.,0.,	0.,		0.,		0.,		0.,		0.,		0.,		0.],
                [0.,0.,0.,	-0.00543349,    0.00729315,	0.02765373,    	-0.055754915,	0.0000126,	0.009819377,	0.00185034968],
                [0.,0.,0.,	0.0438724, 	-0.000027,	0.03258647,   	0.007582604,	0.0042975381,	0.0013797184,	-0.00000284232419],
                [0.,0.,0.,	-0.0182276,	-0.07529696,	-0.12698781,   	-0.043820396,	0.044717986,	-0.080380065,	-0.135735294]
            ]
        )

    # body masses
    if m is None:
        m = np.array(
            [0.,	0.,	0.,	1.81745527,	5.0048646,	3.20289133,	0.86765385,	2.2127686,	1.2838072,	1.62801447]
        )

    if In is None:
        In = np.array(
            [
                [0.,0.,	0.,	0.,	0.,	0.,0.,	0.,	0.,	0.],
                [0.,0.,	0.,	0.00571053,	0.01644345,	0.01129046,0.0040395238,0.0065428368,	0.001994124,	7.96509435e-03],
                [0.,0.,	0.,	-3.39632588745E-4,1.187E-5,	-1.3518E-4,-3.4973136E-4,-4.1196967E-5,	2.569706E-5,	7.65758309e-08],
                [0.,0.,	0.,	2.36775597481E-4,-1.0546E-4,	-0.0013476,-6.8400465E-4,3.272727E-6, 	1.7443538E-4,	-3.28563114e-04],
                [0.,0.,	0.,	0.,	0.,	0.,0.,	0.,	0.,	0.],
                [0.,0.,	0.,	0.012397269634111,0.01804804,	0.00984951,0.0022460561,0.005587829,	0.0021755329,	9.04713878e-03],
                [0.,0.,	0.,	0.001218371526612,5.65E-6,	2.6836E-4,-2.8844444E-4,4.2088668E-4,	-1.2100987E-4,	3.49510343e-07],
                [0.,0.,	0.,	0.,		0.,		0.,	0.,		0.,		0.,		0.],
                [0.,0.,	0.,	0.,		0.,		0.,	0.,		0.,		0.,		0.],
                [0.,0.,	0.,	0.012583780365889,	0.0080242,	0.00728549,	0.0044895027,	0.0024783023,	0.001830376,	1.99007009e-03]
            ]
        )

    # Barycentric Masses (matrix)

    mb1 = np.zeros((70))
    mb4 = np.zeros((70))
    mb5 = np.zeros((70))
    mb6 = np.zeros((70))
    mb7 = np.zeros((70))
    mb8 = np.zeros((70))
    mb9 = np.zeros((70))
    
    mb1[0] = 1
    mb4[1] = 1
    mb5[2] = 1
    mb6[3] = 1
    mb7[4] = 1
    mb8[5] = 1
    mb9[6] = 1

    # Barycentric Vectors (matrix)

    b31 = np.zeros((70))
    b32 = np.zeros((70))
    b33 = np.zeros((70))
    b41 = np.zeros((70))
    b42 = np.zeros((70))
    b43 = np.zeros((70))
    b51 = np.zeros((70))
    b52 = np.zeros((70))
    b53 = np.zeros((70))
    b61 = np.zeros((70))
    b62 = np.zeros((70))
    b63 = np.zeros((70))
    b71 = np.zeros((70))
    b72 = np.zeros((70))
    b73 = np.zeros((70))
    b81 = np.zeros((70))
    b82 = np.zeros((70))
    b83 = np.zeros((70))
    b91 = np.zeros((70))
    b92 = np.zeros((70))
    b93 = np.zeros((70))
    
    b31[7] = 1
    b32[8] = 1
    b33[9] = 1
    b41[10] = 1
    b42[11] = 1
    b43[12] = 1
    b51[13] = 1
    b52[14] = 1
    b53[15] = 1
    b61[16] = 1
    b62[17] = 1
    b63[18] = 1
    b71[19] = 1
    b72[20] = 1
    b73[21] = 1
    b81[22] = 1
    b82[23] = 1
    b83[24] = 1
    b91[25] = 1
    b92[26] = 1
    b93[27] = 1
    

    # Barycentric Tensors (matrix)
    
    K311 = np.zeros((70))
    K312 = np.zeros((70))
    K313 = np.zeros((70))
    K322 = np.zeros((70))
    K323 = np.zeros((70))
    K333 = np.zeros((70))
    K411 = np.zeros((70))
    K412 = np.zeros((70))
    K413 = np.zeros((70))
    K422 = np.zeros((70))
    K423 = np.zeros((70))
    K433 = np.zeros((70))
    K511 = np.zeros((70))
    K512 = np.zeros((70))
    K513 = np.zeros((70))
    K522 = np.zeros((70))
    K523 = np.zeros((70))
    K533 = np.zeros((70))
    K611 = np.zeros((70))
    K612 = np.zeros((70))
    K613 = np.zeros((70))
    K622 = np.zeros((70))
    K623 = np.zeros((70))
    K633 = np.zeros((70))
    K711 = np.zeros((70))
    K712 = np.zeros((70))
    K713 = np.zeros((70))
    K722 = np.zeros((70))
    K723 = np.zeros((70))
    K733 = np.zeros((70))
    K811 = np.zeros((70))
    K812 = np.zeros((70))
    K813 = np.zeros((70))
    K822 = np.zeros((70))
    K823 = np.zeros((70))
    K833 = np.zeros((70))
    K911 = np.zeros((70))
    K912 = np.zeros((70))
    K913 = np.zeros((70))
    K922 = np.zeros((70))
    K923 = np.zeros((70))
    K933 = np.zeros((70))

    K311[28] = 1
    K312[29] = 1
    K313[30] = 1
    K322[31] = 1
    K323[32] = 1
    K333[33] = 1
    K411[34] = 1
    K412[35] = 1
    K413[36] = 1
    K422[37] = 1
    K423[38] = 1
    K433[39] = 1
    K511[40] = 1
    K512[41] = 1
    K513[42] = 1
    K522[43] = 1
    K523[44] = 1
    K533[45] = 1
    K611[46] = 1
    K612[47] = 1
    K613[48] = 1
    K622[49] = 1
    K623[50] = 1
    K633[51] = 1
    K711[52] = 1
    K712[53] = 1
    K713[54] = 1
    K722[55] = 1
    K723[56] = 1
    K733[57] = 1
    K811[58] = 1
    K812[59] = 1
    K813[60] = 1
    K822[61] = 1
    K823[62] = 1
    K833[63] = 1
    K911[64] = 1
    K912[65] = 1
    K913[66] = 1
    K922[67] = 1
    K923[68] = 1
    K933[69] = 1



    # Barycentric Tensors: elements Ks, Kd
    Ks9 = 0.500*(K922+K933)
    Kd9 = 0.500*(K922-K933)
    Ks8 = 0.500*(K811+K833+Ks9)
    Kd8 = 0.500*(-K811+K833+Ks9)
    Ks7 = 0.500*(K711+K722+Ks8)
    Kd7 = 0.500*(K711-K722+Ks8)
    Ks6 = 0.500*(K611+K633+Ks7+2.000*b73*d[3][7])
    Kd6 = 0.500*(-K611+K633-Ks7-2.000*b73*d[3][7])
    Ks5 = 0.500*(K511+K522+Ks6)
    Kd5 = 0.500*(K511-K522+Ks6)
    Ks4 = 0.500*(K422+K433+Ks5+2.000*b53*d[3][5])
    Kd4 = 0.500*(K422-K433+Ks5+2.000*b53*d[3][5])
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

    A = b32
    A = np.vstack((A,br31))
    A = np.vstack((A,b33))
    A = np.vstack((A,Kr312))
    A = np.vstack((A,Kr322))
    A = np.vstack((A,K323))
    A = np.vstack((A,b42))
    A = np.vstack((A,br43))
    A = np.vstack((A,Kr411))
    A = np.vstack((A,K412))
    A = np.vstack((A,K413))
    A = np.vstack((A,Kd4))
    A = np.vstack((A,K423))
    A = np.vstack((A,b51))
    A = np.vstack((A,br52))
    A = np.vstack((A,Kd5))
    A = np.vstack((A,Kr512))
    A = np.vstack((A,K513))
    A = np.vstack((A,Kr523))
    A = np.vstack((A,Kr533))
    A = np.vstack((A,b61))
    A = np.vstack((A,br63))
    A = np.vstack((A,K612))
    A = np.vstack((A,Kr613))
    A = np.vstack((A,Kr622))
    A = np.vstack((A,K623))
    A = np.vstack((A,Kd6))
    A = np.vstack((A,b71))
    A = np.vstack((A,br72))
    A = np.vstack((A,Kd7))
    A = np.vstack((A,K712))
    A = np.vstack((A,K713))
    A = np.vstack((A,K723))
    A = np.vstack((A,Kr733))
    A = np.vstack((A,br81))
    A = np.vstack((A,b83))
    A = np.vstack((A,K812))
    A = np.vstack((A,Kr813))
    A = np.vstack((A,Kr822))
    A = np.vstack((A,K823))
    A = np.vstack((A,Kd8))
    A = np.vstack((A,b92))
    A = np.vstack((A,b93))
    A = np.vstack((A,K911))
    A = np.vstack((A,K912))
    A = np.vstack((A,K913))
    A = np.vstack((A,Kd9))
    A = np.vstack((A,K923))

    #get barycentric parameters
    bar = get_bar(m,l,d,In)

    # get base parameters from barycentric ones (using linear relation, A matrix) 
    delta_out *= 0 # ensure delta is iput as 0
    delta_out += np.dot(A,bar)
    return A
