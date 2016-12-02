#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import numpy as np
import numpy.linalg as la
from scipy import signal
import scipy.linalg as sla
import sympy
from sympy import symbols
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from IPython import embed

import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from identification import helpers

if __name__ == '__main__':
    #urdf_file = '../model/centauro.urdf'
    urdf_file = '../model/walkman.urdf'
    dynamics = iDynTree.DynamicsComputations()
    dynamics.loadRobotModelFromFile(urdf_file)
    dynamics.setFloatingBase('LFoot')
    n_dofs = dynamics.getNrOfDegreesOfFreedom()
    jointNames = []
    for i in range(n_dofs):
        jointNames.append(dynamics.getJointName(i))
    limits = helpers.URDFHelpers.getJointLimits(urdf_file, use_deg=False)

    #for each joint, sweep through all possible joint angles and get mass matrix
    q = iDynTree.VectorDynSize(n_dofs)
    q.zero()
    dq = iDynTree.VectorDynSize(n_dofs)
    #dq.fromList([1.0]*n_dofs)
    dq.zero()
    ddq = iDynTree.VectorDynSize(n_dofs)
    ddq.zero()
    world_gravity = iDynTree.SpatialAcc.fromList([0, 0, -9.81, 0, 0, 0])
    base_velocity = iDynTree.Twist()
    base_velocity.zero()
    base_acceleration = iDynTree.ClassicalAcc()
    base_acceleration.zero()
    rot = iDynTree.Rotation.RPY(0, 0, 0)
    pos = iDynTree.Position.Zero()
    world_T_base = iDynTree.Transform(rot, pos)

    m = iDynTree.MatrixDynSize(n_dofs, n_dofs)
    maxima = [0]*n_dofs
    for i in range(n_dofs):
        for pos in np.arange(limits[jointNames[i]]['lower'], limits[jointNames[i]]['upper'], 0.05):
            q.zero()
            q[i] = pos
            dynamics.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration, world_gravity)
            dynamics.getFreeFloatingMassMatrix(m)
            i_j = np.diag(m.toNumPy())
            maxima[i] = np.max((i_j[i], maxima[i]))

    for l in map(lambda j: "{}: {}".format(jointNames[j], maxima[j]), range(len(maxima))):
        print(l)
