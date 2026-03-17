#!/usr/bin/env python

"""more or less specific to walk-man: get maximum inertia matrix values over the
whole motion range"""

import os
import sys

import numpy as np
import numpy.linalg as la
from idyntree import bindings as iDynTree

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.join(_script_dir, "..")
sys.path.insert(1, _project_dir)
from identification import helpers

if __name__ == "__main__":
    # urdf_file = os.path.join(_project_dir, 'model/centauro.urdf')
    urdf_file = os.path.join(_project_dir, "model/walkman.urdf")
    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(urdf_file)
    model = loader.model()

    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(loader.model())
    kinDyn.setFloatingBase("LFoot")
    n_dofs = kinDyn.getNrOfDegreesOfFreedom()
    jointNames = []
    for i in range(n_dofs):
        jointNames.append(model.getJointName(i))
    limits = helpers.URDFHelpers.getJointLimits(urdf_file, use_deg=False)

    # for each joint, sweep through all possible joint angles and get mass matrix
    s = iDynTree.JointPosDoubleArray(n_dofs)
    ds = iDynTree.JointDOFsDoubleArray(n_dofs)
    for j in range(n_dofs):
        s.setVal(j, 0.0)
        ds.setVal(j, 0.0)

    gravity_vec = iDynTree.Vector3()
    gravity_vec.setVal(0, 0.0)
    gravity_vec.setVal(1, 0.0)
    gravity_vec.setVal(2, -9.81)

    base_velocity = iDynTree.Twist()
    base_velocity.zero()
    rot = iDynTree.Rotation.RPY(0, 0, 0)
    pos = iDynTree.Position.Zero()
    world_T_base = iDynTree.Transform(rot, pos)

    m = iDynTree.MatrixDynSize(n_dofs + 6, n_dofs + 6)
    maxima = [0] * n_dofs
    minima = [99999] * n_dofs
    getMaxima = 1

    for i in range(n_dofs):
        for pos_val in np.arange(limits[jointNames[i]]["lower"], limits[jointNames[i]]["upper"], 0.01):
            for j in range(n_dofs):
                s.setVal(j, 0.0)

            s.setVal(i, pos_val)
            if getMaxima:
                # saggital = pitch, transversal = yaw, lateral = roll
                if i == jointNames.index("LHipYaw"):
                    # lift right leg up 90 deg
                    s.setVal(jointNames.index("RHipYaw"), np.deg2rad(-90))
                    s.setVal(jointNames.index("RHipSag"), np.deg2rad(-90))

                elif i == jointNames.index("RHipYaw"):
                    # lift left leg up 90 deg, turn outside
                    s.setVal(jointNames.index("LHipYaw"), np.deg2rad(90))
                    s.setVal(jointNames.index("LHipSag"), np.deg2rad(-90))

                elif i == jointNames.index("WaistSag"):
                    # lift both arms up
                    s.setVal(jointNames.index("LShSag"), np.deg2rad(-162))
                    s.setVal(jointNames.index("LShLat"), np.deg2rad(85))

                    s.setVal(jointNames.index("RShSag"), np.deg2rad(-162))
                    s.setVal(jointNames.index("RShLat"), np.deg2rad(-85))

                elif i == jointNames.index("WaistYaw"):
                    # lift arms up to 90 deg
                    s.setVal(jointNames.index("LShSag"), np.deg2rad(25))
                    s.setVal(jointNames.index("LShLat"), np.deg2rad(85))

                    s.setVal(jointNames.index("RShSag"), np.deg2rad(25))
                    s.setVal(jointNames.index("RShLat"), np.deg2rad(-85))

                #    print ("{} {}".format(i, jointNames[i]))

            kinDyn.setRobotState(world_T_base, s, base_velocity, ds, gravity_vec)
            kinDyn.getFreeFloatingMassMatrix(m)
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

            """
            # get only gravity vector for q (qdot = qddot = 0)
            kinDyn.setRobotState(world_T_base, s, base_velocity, ds, gravity_vec)
            ext_wrenches = iDynTree.LinkWrenches(model)
            ext_wrenches.zero()
            gen_torques = iDynTree.FreeFloatingGeneralizedTorques(model)
            base_acc_zero = iDynTree.Vector6()
            for j in range(6):
                base_acc_zero.setVal(j, 0.0)
            ddq_zero = iDynTree.JointDOFsDoubleArray(n_dofs)
            for j in range(n_dofs):
                ddq_zero.setVal(j, 0.0)
            kinDyn.inverseDynamics(base_acc_zero, ddq_zero, ext_wrenches, gen_torques)
            gravity = gen_torques.jointTorques().toNumPy()

            # get mass matrix for q, qddot
            ddq_unit = iDynTree.JointDOFsDoubleArray(n_dofs)
            for j in range(n_dofs):
                ddq_unit.setVal(j, 0.0)
            ddq_unit.setVal(i, 1.0)
            kinDyn.inverseDynamics(base_acc_zero, ddq_unit, ext_wrenches, gen_torques)
            massVector = gen_torques.jointTorques().toNumPy() - gravity

            maxima[i] = np.max((massVector[i], maxima[i]))
            """

    if getMaxima:
        print(f"maxima {kinDyn.getFloatingBase()}")
        for l in map(lambda j: f"{jointNames[j]}: {maxima[j]}", range(len(maxima))):
            print(l)
    else:
        print(f"minima {kinDyn.getFloatingBase()}")
        for l in map(lambda j: f"{jointNames[j]}: {minima[j]}", range(len(minima))):
            print(l)
