#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

def merge_masses(m1,m2):
	return m1+m2


def merge_CoM(CoM1,CoM2,m1,m2):

	return (CoM1*m1 + CoM2*m2)/(m1+m2)

# shift inertia using the parallel axis theorem
# d = vector from the body CoM to the new point
def shift_inertia(In,m,d):
	In_shift = In + m*(np.dot(d,d)*np.identity(3)-np.outer(d,d))
	return In_shift


def merge_inertia(In1,In2,CoM1,CoM2,m1,m2):
	CoM_merge = merge_CoM(CoM1,CoM2,m1,m2)
	
	#shift1
	shift1 = CoM1 - CoM_merge
	In1_shift = shift_inertia(In1,m1,shift1)

	#shift2
	shift2 =  CoM2 - CoM_merge
	In2_shift = shift_inertia(In2,m2,shift2)

	return In1_shift + In2_shift



# example of merging LWrMot3 (body1) and LSoftHandLink (body2)

m1 = 0.42801447
m2 = 1.2

CoM1 = np.array([0.0070380706, -0.000010811188,	-0.039669274])
CoM2 = np.array([0.0, 0.0,	-0.17])

In1 = np.array(
            [
                [2.8620156E-4,	5.2570457E-8,	-3.9174213E-5],
                [5.2570457E-8, 	7.1261855E-4, 	-9.5020266E-8],
                [-3.9174213E-5, -9.5020266E-8, 	6.1444258E-4]
            ]
      )

In2 = np.array(
            [
                [0.00232,	0.,			0.],
                [0., 		0.00296, 	0.],
                [0., 		0., 		0.00136]
            ]
      )

# compute merge mass
m3 = merge_masses(m1,m2)
print(m3)

# compute merge CoM
CoM3 = merge_CoM(CoM1,CoM2,m1,m2)
print(CoM3)

#compute merge inertia
In3 = merge_inertia(In1,In2,CoM1,CoM2,m1,m2)
print(In3)