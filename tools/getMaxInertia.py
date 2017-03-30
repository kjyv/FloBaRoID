#!/usr/bin/env python
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
    urdf_file = '../model/walkman_orig.urdf'
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
    torques = iDynTree.VectorDynSize(n_dofs)
    baseReactionForce = iDynTree.Wrench()

    m = iDynTree.MatrixDynSize(n_dofs, n_dofs)
    maxima = [0]*n_dofs
    minima = [99999]*n_dofs
    getMaxima = 1

    for i in range(n_dofs):
        for pos in np.arange(limits[jointNames[i]]['lower'], limits[jointNames[i]]['upper'], 0.01):
            q.zero()

            q[i] = pos
            if getMaxima:
                # saggital = pitch, transversal = yaw, lateral = roll
                if i == jointNames.index('LHipYaw'):
                    # lift right leg up 90 deg
                    q[jointNames.index('RHipYaw')] = np.deg2rad(-90)
                    q[jointNames.index('RHipSag')] = np.deg2rad(-90)

                elif i == jointNames.index('RHipYaw'):
                    # lift left leg up 90 deg, turn outside
                    q[jointNames.index('LHipYaw')] = np.deg2rad(90)
                    q[jointNames.index('LHipSag')] = np.deg2rad(-90)

                elif i == jointNames.index('WaistSag'):
                    # lift both arms up
                    q[jointNames.index('LShSag')] = np.deg2rad(-162)
                    q[jointNames.index('LShLat')] = np.deg2rad(85)

                    q[jointNames.index('RShSag')] = np.deg2rad(-162)
                    q[jointNames.index('RShLat')] = np.deg2rad(-85)

                elif i == jointNames.index('WaistYaw'):
                    # lift arms up to 90 deg
                    q[jointNames.index('LShSag')] = np.deg2rad(25)
                    q[jointNames.index('LShLat')] = np.deg2rad(85)

                    q[jointNames.index('RShSag')] = np.deg2rad(25)
                    q[jointNames.index('RShLat')] = np.deg2rad(-85)

                #    print ("{} {}".format(i, jointNames[i]))

            dynamics.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration, world_gravity)
            dynamics.getFreeFloatingMassMatrix(m)
            I = m.toNumPy()

            # subtract inertia that results from floating base link having zero acceleration (it
            # should actually react to acceleration of current joint)
            I_qq = I[6:, 6:]
            I_xx = I[:6, :6]
            I_qx = I[6:, :6]
            I_xq = I[:6, 6:]
            I_eff = I_qq - I_qx.dot(la.inv(I_xx).dot(I_xq))
            i_j = np.diag(I_eff)

            maxima[i] = np.max((i_j[i], maxima[i]))
            minima[i] = np.min((i_j[i], minima[i]))

            '''
            # get only gravity vector for q (qdot = qddot = 0)
            dynamics.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration, world_gravity)
            dynamics.inverseDynamics(torques, baseReactionForce)
            gravity = torques.toNumPy()

            # get mass matrix for q, qddot
            ddq[i] = 1
            dynamics.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration, world_gravity)
            dynamics.inverseDynamics(torques, baseReactionForce)
            massVector = torques.toNumPy() - gravity
            ddq.zero()

            maxima[i] = np.max((massVector[i], maxima[i]))
            '''

    if getMaxima:
        print("maxima {}".format(dynamics.getFloatingBase()))
        for l in map(lambda j: "{}: {}".format(jointNames[j], maxima[j]), range(len(maxima))):
            print(l)
    else:
        print("minima {}".format(dynamics.getFloatingBase()))
        for l in map(lambda j: "{}: {}".format(jointNames[j], minima[j]), range(len(minima))):
            print(l)
