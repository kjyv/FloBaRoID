#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

# kinematics, dynamics and URDF reading
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from IPython import embed
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
urdf_file = os.path.join(os.path.dirname(__file__), "../model/threeLinks.urdf")
contactFrame = 'contact_ft'

def test_regressors():
    #get some random state values and compare inverse dynamics torques with torques
    #obtained with regressor and parameter vector

    idyn_model = iDynTree.Model()
    iDynTree.modelFromURDF(urdf_file, idyn_model)

    # create generator instance and load model
    generator = iDynTree.DynamicsRegressorGenerator()
    generator.loadRobotAndSensorsModelFromFile(urdf_file)
    regrXml = '''
    <regressor>
      <baseLinkDynamics/>
      <jointTorqueDynamics>
        <allJoints/>
      </jointTorqueDynamics>
    </regressor>'''
    generator.loadRegressorStructureFromString(regrXml)
    num_model_params = generator.getNrOfParameters()
    num_out = generator.getNrOfOutputs()
    n_dofs = generator.getNrOfDegreesOfFreedom()
    num_samples = 100

    xStdModel = iDynTree.VectorDynSize(num_model_params)
    generator.getModelParameters(xStdModel)
    xStdModel = xStdModel.toNumPy()

    gravity_twist = iDynTree.Twist.fromList([0,0,-9.81,0,0,0])
    gravity_acc = iDynTree.SpatialAcc.fromList([0, 0, -9.81, 0, 0, 0])

    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(urdf_file)

    regressor_stack = np.zeros(shape=((n_dofs+6)*num_samples, num_model_params))
    idyn_torques = np.zeros(shape=((n_dofs+6)*num_samples))
    contactForceSum = np.zeros(shape=((n_dofs+6)*num_samples))

    for sample_index in range(0, num_samples):
        q = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())
        dq = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())
        ddq = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())
        base_vel = np.pi*np.random.rand(6)
        base_acc = np.pi*np.random.rand(6)

        #rpy = [0,0,0]
        rpy = np.random.ranf(3)*0.1
        rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        pos = iDynTree.Position.Zero()
        world_T_base = iDynTree.Transform(rot, pos).inverse()

        # rotate base vel and acc to world frame
        to_world = world_T_base.getRotation().toNumPy()
        base_vel[0:3] = to_world.dot(base_vel[0:3])
        base_vel[3:] = to_world.dot(base_vel[3:])
        base_acc[0:3] = to_world.dot(base_acc[0:3])
        base_acc[3:] = to_world.dot(base_acc[3:])

        base_velocity = iDynTree.Twist.fromList(base_vel)
        base_acceleration = iDynTree.Twist.fromList(base_acc)
        base_acceleration_acc = iDynTree.ClassicalAcc.fromList(base_acc)

        # regressor
        generator.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration, gravity_twist)

        regressor = iDynTree.MatrixDynSize(num_out, num_model_params)
        knownTerms = iDynTree.VectorDynSize(num_out)
        if not generator.computeRegressor(regressor, knownTerms):
            print("Error during numeric computation of regressor")

        #the base forces are expressed in the base frame for the regressor, so transform them
        to_world = np.fromstring(world_T_base.getRotation().toString(), sep=' ').reshape((3,3))
        regressor = regressor.toNumPy()
        regressor[0:3, :] = to_world.dot(regressor[0:3, :])
        regressor[3:6, :] = to_world.dot(regressor[3:6, :])

        row_index = (n_dofs+6)*sample_index   # index for current row in stacked regressor matrix
        np.copyto(regressor_stack[row_index:row_index+n_dofs+6], regressor)

        # inverse dynamics
        #dynComp.setFloatingBase('base_link')
        dynComp.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration_acc, gravity_acc)
        torques = iDynTree.VectorDynSize(n_dofs)
        baseReactionForce = iDynTree.Wrench()
        dynComp.inverseDynamics(torques, baseReactionForce)

        torques = np.concatenate((baseReactionForce.toNumPy(), torques.toNumPy()))
        np.copyto(idyn_torques[row_index:row_index+n_dofs+6], torques)

        # contacts
        dim = n_dofs + 6
        contact = np.array([0, 0, 10, 0, 0, 0])
        jacobian = iDynTree.MatrixDynSize(6, dim)
        dynComp.getFrameJacobian(contactFrame, jacobian)
        jacobian = jacobian.toNumPy()
        contactForceSum[sample_index*dim:(sample_index+1)*dim] = jacobian.T.dot(contact)

    regressor_torques = np.dot(regressor_stack, xStdModel) + contactForceSum
    idyn_torques += contactForceSum

    error = np.reshape(regressor_torques - idyn_torques, (num_samples, dim))

    #plots = plt.plot(range(0, num_samples), error)
    #plt.legend(plots, ['f_x', 'f_y', 'f_z', 'm_x', 'm_y', 'm_z', 'j_0'])
    #plt.show()

    error_norm = la.norm(error)
    print(error_norm)
    assert error_norm <= 0.01

if __name__ == '__main__':
    test_regressors()
