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
urdf_file = os.path.join(os.path.dirname(__file__), "../model/walkman_right_leg.urdf")

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
    num_params = generator.getNrOfParameters()
    num_out = generator.getNrOfOutputs()
    n_dofs = generator.getNrOfDegreesOfFreedom()
    num_samples = 100

    xStdModel = iDynTree.VectorDynSize(num_params)
    generator.getModelParameters(xStdModel)
    xStdModel = xStdModel.toNumPy()

    gravity_twist = iDynTree.Twist.fromList([0,0,-9.81,0,0,0])
    gravity_acc = iDynTree.SpatialAcc.fromList([0, 0, -9.81, 0, 0, 0])

    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(urdf_file)

    regressor_stack = np.zeros(shape=((n_dofs+6)*num_samples, num_params))
    idyn_torques = np.zeros(shape=((n_dofs+6)*num_samples))

    for sample_index in range(0,num_samples):
        q = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())
        dq = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())
        ddq = iDynTree.VectorDynSize.fromList(((np.random.ranf(n_dofs)*2-1)*np.pi).tolist())

        base_velocity = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
        base_acceleration = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
        base_acceleration_acc = iDynTree.ClassicalAcc.fromList(base_acceleration.toNumPy())
        rpy = np.random.ranf(3)*0.05
        #rpy = [0,0,0]
        rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        pos = iDynTree.Position.Zero()
        world_T_base = iDynTree.Transform(rot, pos)

        #regressor
        generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, gravity_twist)

        regressor = iDynTree.MatrixDynSize(num_out, num_params)
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

        #inverse dynamics
        dynComp.setFloatingBase('Waist')
        dynComp.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration_acc, gravity_acc)
        torques = iDynTree.VectorDynSize(n_dofs)
        baseReactionForce = iDynTree.Wrench()
        dynComp.inverseDynamics(torques, baseReactionForce)

        torques = np.concatenate((baseReactionForce.toNumPy(), torques.toNumPy()))
        np.copyto(idyn_torques[row_index:row_index+n_dofs+6], torques)

    regressor_torques = np.dot(regressor_stack, xStdModel)

    error = np.reshape(regressor_torques-idyn_torques, (num_samples, n_dofs+6))

    #plt.plot(range(0,num_samples), error)
    #plt.show()
    assert la.norm(error) <= 0.01

if __name__ == '__main__':
    test_regressors()
