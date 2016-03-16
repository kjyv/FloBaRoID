#	ROBOTRAN - Version 6.3B (build : 6 mars 2008)
#	==> Model: Open-Loop Inverse Dynamics for one arm
#	==> Formalism: Recursive Barycentric Newton/Euler
#	==> Nbody: 9

import numpy as np
from math import sin, cos

def invdynabar(Qq_out, q, qd, qdd, delta, d):
    #Qq_out is a vector of 9 values which the calculated torques will be written to.
    #q,qd,qdd are the joint positions and their derivates
    #d contains the position of the joints relative to their parent joints (kinematic parameters) (3 x N_DOF)
    #delta is the base inertia parameters vector of 48 values

    #Note, the "0" elements of vectors (vec[0]) are always set to zero. We use a matlab-like convention
    #with array elements starting at index 1 (vec[1]).

    # points of the kinematic chain (relative distances for one joint from the previous)
    if d is None:
        d = np.array(
            [
                [0., 0.,	        0., 	0.,    	0.,	0.,     0.,	0.,	0.,	0.],
                [0., 0.045646,      0., 	0.,     0.,     0.,  0.036, -0.075,	0.,	0.],
                [0., 0.219164,	0., 	0., 0.1091,	0., 	0.,	0.,		0.,	0.],
                [0., 0.001531,      0., 	0., -0.053, -0.222,  -0.15,-0.1955,     0., -0.092]
            ]
        )

    # gravity
    g = [0.0, 0.0, 0.0, -9.81]

    # fixed joints
    q[1] = -0.3491;
    q[2]  = 0.7330382858903738;

    # Trigonometric Variables

    S1 = sin(q[1])
    C1 = cos(q[1])
    S2 = sin(q[2])
    C2 = cos(q[2])
    S3 = sin(q[3])
    C3 = cos(q[3])
    S4 = sin(q[4])
    C4 = cos(q[4])
    S5 = sin(q[5])
    C5 = cos(q[5])
    S6 = sin(q[6])
    C6 = cos(q[6])
    S7 = sin(q[7])
    C7 = cos(q[7])
    S8 = sin(q[8])
    C8 = cos(q[8])
    S9 = sin(q[9])
    C9 = cos(q[9])

    # Minimal Set of Dynamical Paramaters

    b32 = delta[1]
    br31 = delta[2]
    b33 = delta[3]
    Kr312 = delta[4]
    Kr322 = delta[5]
    K323 = delta[6]
    b42 = delta[7]
    br43 = delta[8]
    Kr411 = delta[9]
    K412 = delta[10]
    K413 = delta[11]
    Kd4 = delta[12]
    K423 = delta[13]
    b51 = delta[14]
    br52 = delta[15]
    Kd5 = delta[16]
    Kr512 = delta[17]
    K513 = delta[18]
    Kr523 = delta[19]
    Kr533 = delta[20]
    b61 = delta[21]
    br63 = delta[22]
    K612 = delta[23]
    Kr613 = delta[24]
    Kr622 = delta[25]
    K623 = delta[26]
    Kd6 = delta[27]
    b71 = delta[28]
    br72 = delta[29]
    Kd7 = delta[30]
    K712 = delta[31]
    K713 = delta[32]
    K723 = delta[33]
    Kr733 = delta[34]
    br81 = delta[35]
    b83 = delta[36]
    K812 = delta[37]
    Kr813 = delta[38]
    Kr822 = delta[39]
    K823 = delta[40]
    Kd8 = delta[41]
    b92 = delta[42]
    b93 = delta[43]
    K911 = delta[44]
    K912 = delta[45]
    K913 = delta[46]
    Kd9 = delta[47]
    K923 = delta[48]

    # Forward Kinematics
    ALS22 = -g[3]*S2
    ALS32 = -g[3]*C2
    BS93 = -qd[3]*qd[3]
    ALS13 = -ALS32*S3
    ALS33 = ALS32*C3
    OM24 = qd[3]*C4
    OM34 = -qd[3]*S4
    OMP24 = -qd[3]*qd[4]*S4+qdd[3]*C4
    OMP34 = -qd[3]*qd[4]*C4-qdd[3]*S4
    BS24 = qd[4]*OM24
    BS34 = qd[4]*OM34
    BS54 = -qd[4]*qd[4]-OM34*OM34
    BS64 = OM24*OM34
    BS94 = -qd[4]*qd[4]-OM24*OM24
    BETA24 = BS24-OMP34
    BETA34 = BS34+OMP24
    BETA64 = -qdd[4]+BS64
    BETA84 = qdd[4]+BS64
    ALS14 = ALS13+qdd[3]*d[3][4]
    ALS24 = ALS22*C4+S4*(ALS33+BS93*d[3][4])
    ALS34 = -ALS22*S4+C4*(ALS33+BS93*d[3][4])
    OM15 = qd[4]*C5+OM24*S5
    OM25 = -qd[4]*S5+OM24*C5
    OM35 = qd[5]+OM34
    OMP15 = C5*(qdd[4]+qd[5]*OM24)+S5*(OMP24-qd[4]*qd[5])
    OMP25 = C5*(OMP24-qd[4]*qd[5])-S5*(qdd[4]+qd[5]*OM24)
    OMP35 = qdd[5]+OMP34
    BS15 = -OM25*OM25-OM35*OM35
    BS25 = OM15*OM25
    BS35 = OM15*OM35
    BS55 = -OM15*OM15-OM35*OM35
    BS65 = OM25*OM35
    BS95 = -OM15*OM15-OM25*OM25
    BETA25 = BS25-OMP35
    BETA35 = BS35+OMP25
    BETA45 = BS25+OMP35
    BETA65 = BS65-OMP15
    BETA75 = BS35-OMP25
    BETA85 = BS65+OMP15
    ALS15 = C5*(ALS14+BETA34*d[3][5])+S5*(ALS24+BETA64*d[3][5])
    ALS25 = C5*(ALS24+BETA64*d[3][5])-S5*(ALS14+BETA34*d[3][5])
    ALS35 = ALS34+BS94*d[3][5]
    OM16 = OM15*C6-OM35*S6
    OM26 = qd[6]+OM25
    OM36 = OM15*S6+OM35*C6
    OMP16 = C6*(OMP15-qd[6]*OM35)-S6*(OMP35+qd[6]*OM15)
    OMP26 = qdd[6]+OMP25
    OMP36 = C6*(OMP35+qd[6]*OM15)+S6*(OMP15-qd[6]*OM35)
    BS16 = -OM26*OM26-OM36*OM36
    BS26 = OM16*OM26
    BS36 = OM16*OM36
    BS66 = OM26*OM36
    BS96 = -OM16*OM16-OM26*OM26
    BETA36 = BS36+OMP26
    BETA46 = BS26+OMP36
    BETA66 = BS66-OMP16
    BETA76 = BS36-OMP26
    ALS16 = C6*(ALS15+BETA35*d[3][6]+BS15*d[1][6])-S6*(ALS35+BETA75*d[1][6]+BS95*d[3][6])
    ALS26 = ALS25+BETA45*d[1][6]+BETA65*d[3][6]
    ALS36 = C6*(ALS35+BETA75*d[1][6]+BS95*d[3][6])+S6*(ALS15+BETA35*d[3][6]+BS15*d[1][6])
    OM17 = OM16*C7+OM26*S7
    OM27 = -OM16*S7+OM26*C7
    OM37 = qd[7]+OM36
    OMP17 = C7*(OMP16+qd[7]*OM26)+S7*(OMP26-qd[7]*OM16)
    OMP27 = C7*(OMP26-qd[7]*OM16)-S7*(OMP16+qd[7]*OM26)
    OMP37 = qdd[7]+OMP36
    BS17 = -OM27*OM27-OM37*OM37
    BS27 = OM17*OM27
    BS37 = OM17*OM37
    BS57 = -OM17*OM17-OM37*OM37
    BS67 = OM27*OM37
    BETA27 = BS27-OMP37
    BETA47 = BS27+OMP37
    BETA77 = BS37-OMP27
    BETA87 = BS67+OMP17
    ALS17 = C7*(ALS16+BETA36*d[3][7]+BS16*d[1][7])+S7*(ALS26+BETA46*d[1][7]+BETA66*d[3][7])
    ALS27 = C7*(ALS26+BETA46*d[1][7]+BETA66*d[3][7])-S7*(ALS16+BETA36*d[3][7]+BS16*d[1][7])
    ALS37 = ALS36+BETA76*d[1][7]+BS96*d[3][7]
    OM18 = OM17*C8-OM37*S8
    OM28 = qd[8]+OM27
    OM38 = OM17*S8+OM37*C8
    OMP18 = C8*(OMP17-qd[8]*OM37)-S8*(OMP37+qd[8]*OM17)
    OMP28 = qdd[8]+OMP27
    OMP38 = C8*(OMP37+qd[8]*OM17)+S8*(OMP17-qd[8]*OM37)
    BS18 = -OM28*OM28-OM38*OM38
    BS28 = OM18*OM28
    BS38 = OM18*OM38
    BS68 = OM28*OM38
    BS98 = -OM18*OM18-OM28*OM28
    BETA38 = BS38+OMP28
    BETA48 = BS28+OMP38
    BETA68 = BS68-OMP18
    BETA78 = BS38-OMP28
    ALS18 = ALS17*C8-ALS37*S8
    ALS38 = ALS17*S8+ALS37*C8
    OM19 = qd[9]+OM18
    OM29 = OM28*C9+OM38*S9
    OM39 = -OM28*S9+OM38*C9
    OMP19 = qdd[9]+OMP18
    OMP29 = C9*(OMP28+qd[9]*OM38)+S9*(OMP38-qd[9]*OM28)
    OMP39 = C9*(OMP38-qd[9]*OM28)-S9*(OMP28+qd[9]*OM38)
    BS29 = OM19*OM29
    BS39 = OM19*OM39
    BS59 = -OM19*OM19-OM39*OM39
    BS69 = OM29*OM39
    BS99 = -OM19*OM19-OM29*OM29
    BETA29 = BS29-OMP39
    BETA39 = BS39+OMP29
    BETA69 = BS69-OMP19
    BETA89 = BS69+OMP19
    ALS19 = ALS18+BETA38*d[3][9]
    ALS29 = C9*(ALS27+BETA68*d[3][9])+S9*(ALS38+BS98*d[3][9])
    ALS39 = C9*(ALS38+BS98*d[3][9])-S9*(ALS27+BETA68*d[3][9])

    # Backward Dynamics

    GS19 = BETA29*b92+BETA39*b93
    GS29 = BETA69*b93+BS59*b92
    GS39 = BETA89*b92+BS99*b93
    CF19 = -ALS29*b93+ALS39*b92+K911*OMP19+K912*OMP29+K913*OMP39+OM29*(K913*OM19+K923*OM29-Kd9*OM39)-OM39*\
           (K912*OM19+K923*OM39+Kd9*OM29)
    CF29 = ALS19*b93+K912*OMP19+K923*OMP39+Kd9*OMP29-OM19*(K913*OM19+K923*OM29-Kd9*OM39)+OM39*\
            (K911*OM19+K912*OM29+K913*OM39)
    CF39 = -ALS19*b92+K913*OMP19+K923*OMP29-Kd9*OMP39+OM19*(K912*OM19+K923*OM39+Kd9*OM29)-OM29*(K911*OM19+K912*OM29+K913*
     OM39)
    GS18 = GS19+BETA38*b83+BS18*br81
    GS28 = BETA48*br81+BETA68*b83+GS29*C9-GS39*S9
    GS38 = BETA78*br81+BS98*b83+GS29*S9+GS39*C9
    CF18 = CF19-ALS27*b83+K812*OMP28-Kd8*OMP18+Kr813*OMP38+OM28*(K823*OM28+Kd8*OM38+Kr813*OM18)-OM38*(K812*OM18+K823*OM38+
     Kr822*OM28)-d[3][9]*(GS29*C9-GS39*S9)
    CF28 = ALS18*b83-ALS38*br81+CF29*C9-CF39*S9+GS19*d[3][9]+K812*OMP18+K823*OMP38+Kr822*OMP28-OM18*(K823*OM28+Kd8*OM38+
     Kr813*OM18)+OM38*(K812*OM28-Kd8*OM18+Kr813*OM38)
    CF38 = ALS27*br81+CF29*S9+CF39*C9+K823*OMP28+Kd8*OMP38+Kr813*OMP18+OM18*(K812*OM18+K823*OM38+Kr822*OM28)-OM28*(K812*
     OM28-Kd8*OM18+Kr813*OM38)
    GS17 = BETA27*br72+BS17*b71+GS18*C8+GS38*S8
    GS27 = GS28+BETA47*b71+BS57*br72
    GS37 = BETA77*b71+BETA87*br72-GS18*S8+GS38*C8
    CF17 = ALS37*br72+CF18*C8+CF38*S8+K712*OMP27+K713*OMP37+Kd7*OMP17+OM27*(K713*OM17+K723*OM27+Kr733*OM37)-OM37*(K712*
     OM17+K723*OM37-Kd7*OM27)
    CF27 = CF28-ALS37*b71+K712*OMP17+K723*OMP37-Kd7*OMP27-OM17*(K713*OM17+K723*OM27+Kr733*OM37)+OM37*(K712*OM27+K713*OM37+
     Kd7*OM17)
    CF37 = -ALS17*br72+ALS27*b71-CF18*S8+CF38*C8+K713*OMP17+K723*OMP27+Kr733*OMP37+OM17*(K712*OM17+K723*OM37-Kd7*OM27)- \
     OM27*(K712*OM27+K713*OM37+Kd7*OM17)
    GS16 = BETA36*br63+BS16*b61+GS17*C7-GS27*S7
    GS26 = BETA46*b61+BETA66*br63+GS17*S7+GS27*C7
    GS36 = GS37+BETA76*b61+BS96*br63
    CF16 = -ALS26*br63+CF17*C7-CF27*S7+K612*OMP26-Kd6*OMP16+Kr613*OMP36+OM26*(K623*OM26+Kd6*OM36+Kr613*OM16)-OM36*(K612*
     OM16+K623*OM36+Kr622*OM26)-d[3][7]*(GS17*S7+GS27*C7)
    CF26 = ALS16*br63-ALS36*b61+CF17*S7+CF27*C7-GS37*d[1][7]+K612*OMP16+K623*OMP36+Kr622*OMP26-OM16*(K623*OM26+Kd6*OM36+
     Kr613*OM16)+OM36*(K612*OM26-Kd6*OM16+Kr613*OM36)+d[3][7]*(GS17*C7-GS27*S7)
    CF36 = CF37+ALS26*b61+K623*OMP26+Kd6*OMP36+Kr613*OMP16+OM16*(K612*OM16+K623*OM36+Kr622*OM26)-OM26*(K612*OM26-Kd6*OM16+
     Kr613*OM36)+d[1][7]*(GS17*S7+GS27*C7)
    GS15 = BETA25*br52+BS15*b51+GS16*C6+GS36*S6
    GS25 = GS26+BETA45*b51+BS55*br52
    GS35 = BETA75*b51+BETA85*br52-GS16*S6+GS36*C6
    CF15 = ALS35*br52+CF16*C6+CF36*S6-GS26*d[3][6]+K513*OMP35+Kd5*OMP15+Kr512*OMP25+OM25*(K513*OM15+Kr523*OM25+Kr533*OM35)\
     -OM35*(-Kd5*OM25+Kr512*OM15+Kr523*OM35)
    CF25 = CF26-ALS35*b51-Kd5*OMP25+Kr512*OMP15+Kr523*OMP35-OM15*(K513*OM15+Kr523*OM25+Kr533*OM35)+OM35*(K513*OM35+Kd5*
     OM15+Kr512*OM25)-d[1][6]*(-GS16*S6+GS36*C6)+d[3][6]*(GS16*C6+GS36*S6)
    CF35 = -ALS15*br52+ALS25*b51-CF16*S6+CF36*C6+GS26*d[1][6]+K513*OMP15+Kr523*OMP25+Kr533*OMP35+OM15*(-Kd5*OM25+Kr512*
     OM15+Kr523*OM35)-OM25*(K513*OM35+Kd5*OM15+Kr512*OM25)
    GS14 = BETA24*b42+BETA34*br43+GS15*C5-GS25*S5
    GS24 = BETA64*br43+BS54*b42+GS15*S5+GS25*C5
    GS34 = GS35+BETA84*b42+BS94*br43
    CF14 = qdd[4]*Kr411-ALS24*br43+ALS34*b42+CF15*C5-CF25*S5+K412*OMP24+K413*OMP34+OM24*(qd[4]*K413+K423*OM24-Kd4*OM34)-\
     OM34*(qd[4]*K412+K423*OM34+Kd4*OM24)-d[3][5]*(GS15*S5+GS25*C5)
    CF24 = -qd[4]*(qd[4]*K413+K423*OM24-Kd4*OM34)+qdd[4]*K412+ALS14*br43+CF15*S5+CF25*C5+K423*OMP34+Kd4*OMP24+OM34*(qd[4]*
     Kr411+K412*OM24+K413*OM34)+d[3][5]*(GS15*C5-GS25*S5)
    CF34 = CF35+qd[4]*(qd[4]*K412+K423*OM34+Kd4*OM24)+qdd[4]*K413-ALS14*b42+K423*OMP24-Kd4*OMP34-OM24*(qd[4]*Kr411+K412*
     OM24+K413*OM34)
    CF13 = CF14+qd[3]*qd[3]*K323+qdd[3]*Kr312-ALS22*b33+d[2][4]*(GS24*S4+GS34*C4)-d[3][4]*(GS24*C4-GS34*S4)
    CF23 = qdd[3]*Kr322+ALS13*b33-ALS33*br31+CF24*C4-CF34*S4+GS14*d[3][4]
    CF33 = -qd[3]*qd[3]*Kr312+qdd[3]*K323+ALS22*br31+CF24*S4+CF34*C4-GS14*d[2][4]
    CF12 = ALS32*b32+CF13*C3+CF33*S3
    CF32 = -CF13*S3+CF33*C3
    CF31 = CF23*S2+CF32*C2

    # Symbolic Outputs

    #Qq[0] = CF11
    #Qq[1] = CF32
    Qq_out[0] = CF23
    Qq_out[1] = CF14
    Qq_out[2] = CF35
    Qq_out[3] = CF26
    Qq_out[4] = CF37
    Qq_out[5] = CF28
    Qq_out[6] = CF19

