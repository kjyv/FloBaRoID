#!/usr/bin/env python3

import os

import numpy as np
import numpy.linalg as la

# kinematics, dynamics and URDF reading
from idyntree import bindings as iDynTree

urdf_file = os.path.join(os.path.dirname(__file__), "../model/threeLinks.urdf")
contactFrame = "contact_ft"


def test_regressors():
    # get some random state values and compare inverse dynamics torques with torques
    # obtained with regressor and parameter vector

    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(urdf_file)
    idyn_model = loader.model()

    # create KinDynComputations for regressor
    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(loader.model())
    n_dofs = kinDyn.getNrOfDegreesOfFreedom()
    num_links = idyn_model.getNrOfLinks()
    num_model_params = num_links * 10
    n_dofs + 6  # floating base
    num_samples = 100

    xStdModel = iDynTree.VectorDynSize(num_model_params)
    idyn_model.getInertialParameters(xStdModel)
    xStdModel = xStdModel.toNumPy()

    gravity_vec = iDynTree.Vector3()
    gravity_vec.setVal(0, 0.0)
    gravity_vec.setVal(1, 0.0)
    gravity_vec.setVal(2, -9.81)

    # create a second KinDynComputations for inverse dynamics
    kinDyn2 = iDynTree.KinDynComputations()
    kinDyn2.loadRobotModel(loader.model())

    regressor_stack = np.zeros(shape=((n_dofs + 6) * num_samples, num_model_params))
    idyn_torques = np.zeros(shape=((n_dofs + 6) * num_samples))
    contactForceSum = np.zeros(shape=((n_dofs + 6) * num_samples))

    for sample_index in range(0, num_samples):
        q_np = (np.random.ranf(n_dofs) * 2 - 1) * np.pi
        dq_np = (np.random.ranf(n_dofs) * 2 - 1) * np.pi
        ddq_np = (np.random.ranf(n_dofs) * 2 - 1) * np.pi
        base_vel_np = np.pi * np.random.rand(6)
        base_acc_np = np.pi * np.random.rand(6)

        # build JointPosDoubleArray / JointDOFsDoubleArray
        s = iDynTree.JointPosDoubleArray(n_dofs)
        ds = iDynTree.JointDOFsDoubleArray(n_dofs)
        ddq = iDynTree.JointDOFsDoubleArray(n_dofs)
        for j in range(n_dofs):
            s.setVal(j, q_np[j])
            ds.setVal(j, dq_np[j])
            ddq.setVal(j, ddq_np[j])

        # rpy = [0,0,0]
        rpy = np.random.ranf(3) * 0.1
        rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        pos = iDynTree.Position.Zero()
        world_T_base = iDynTree.Transform(rot, pos).inverse()

        base_velocity = iDynTree.Twist.FromPython(base_vel_np.tolist())

        base_acc_vec6 = iDynTree.Vector6()
        for j in range(6):
            base_acc_vec6.setVal(j, base_acc_np[j])

        # regressor
        kinDyn.setRobotState(world_T_base, s, base_velocity, ds, gravity_vec)

        regressor = iDynTree.MatrixDynSize()
        if not kinDyn.inverseDynamicsInertialParametersRegressor(base_acc_vec6, ddq, regressor):
            print("Error during numeric computation of regressor")

        regressor = regressor.toNumPy()

        row_index = (n_dofs + 6) * sample_index  # index for current row in stacked regressor matrix
        np.copyto(regressor_stack[row_index : row_index + n_dofs + 6], regressor)

        # inverse dynamics
        kinDyn2.setRobotState(world_T_base, s, base_velocity, ds, gravity_vec)

        ext_wrenches = iDynTree.LinkWrenches(idyn_model)
        ext_wrenches.zero()
        gen_torques = iDynTree.FreeFloatingGeneralizedTorques(idyn_model)

        base_acceleration_acc = iDynTree.Vector6()
        for j in range(6):
            base_acceleration_acc.setVal(j, base_acc_np[j])

        kinDyn2.inverseDynamics(base_acceleration_acc, ddq, ext_wrenches, gen_torques)

        baseWrench = gen_torques.baseWrench()
        jointTorques = gen_torques.jointTorques()
        torques = np.concatenate((baseWrench.toNumPy(), jointTorques.toNumPy()))
        np.copyto(idyn_torques[row_index : row_index + n_dofs + 6], torques)

        # contacts
        dim = n_dofs + 6
        contact = np.array([0, 0, 10, 0, 0, 0])
        jacobian = iDynTree.MatrixDynSize(6, dim)
        kinDyn2.getFrameFreeFloatingJacobian(contactFrame, jacobian)
        jacobian = jacobian.toNumPy()
        contactForceSum[sample_index * dim : (sample_index + 1) * dim] = jacobian.T.dot(contact)

    regressor_torques = np.dot(regressor_stack, xStdModel) + contactForceSum
    idyn_torques += contactForceSum

    error = np.reshape(regressor_torques - idyn_torques, (num_samples, dim))

    # plots = plt.plot(range(0, num_samples), error)
    # plt.legend(plots, ['f_x', 'f_y', 'f_z', 'm_x', 'm_y', 'm_z', 'j_0'])
    # plt.show()

    error_norm = la.norm(error)
    print(error_norm)
    assert error_norm <= 0.01


if __name__ == "__main__":
    test_regressors()
