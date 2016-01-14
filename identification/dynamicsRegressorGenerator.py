# -*- coding: utf-8 -*-

import iDynTree
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
#np.set_printoptions(suppress=True)

def vecToNumPy(self):
    return np.fromstring(self.toString(), sep=' ')
iDynTree.VectorDynSize.toNumPy = vecToNumPy
iDynTree.Wrench.toNumPy = vecToNumPy
iDynTree.Twist.toNumPy = vecToNumPy

#TODO: not suited for matrices of more than 2 dimensions...
def matToNumPy(self):
    return np.fromstring(self.toString(), sep=' ').reshape(self.rows(), self.cols())
iDynTree.MatrixDynSize.toNumPy = matToNumPy

URDF_FILE = '../urdf/robot2.urdf'

generator = iDynTree.DynamicsRegressorGenerator()
generator.loadRobotAndSensorsModelFromFile(URDF_FILE)

regrXml = '''
          <regressor>
             <subtreeBaseDynamics>'
               <FTSensorLink>arm</FTSensorLink>'
             </subtreeBaseDynamics>'
           </regressor>'''
generator.loadRegressorStructureFromString(regrXml)

N_DOFS = generator.getNrOfDegreesOfFreedom()

# Get the number of outputs of the regressor
# Given that we are considering only the base dynamics
# of a subtree, we will have just 6 outputs (3 force, 3 torques)
N_OUT = generator.getNrOfOutputs()

# get initial inertia params (probably from cad)
N_PARAMS = generator.getNrOfParameters()
cadParams = iDynTree.VectorDynSize(N_PARAMS)
generator.getModelParameters(cadParams)

# set system state
q = iDynTree.VectorDynSize(N_DOFS)
dq = iDynTree.VectorDynSize(N_DOFS)
ddq = iDynTree.VectorDynSize(N_DOFS)

for dof in range(N_DOFS):
    q.setVal(dof, 1.0)
    dq.setVal(dof, 0.4)
    ddq.setVal(dof, 0.3)

gravity_twist = iDynTree.Twist()
gravity_twist.zero()
gravity_twist.setVal(2, -9.81)
generator.setRobotState(q,dq,ddq, gravity_twist)  # fixed base, base acceleration etc. =0

# set torque measurements from experiment data
# TODO: how to do this for many experiment datasets (don't want to loop over them really)
# (functions with one big matrix possible?)

#set some fake data
sensorMeasure = iDynTree.Wrench()
sensorMeasure.setVal(0, 0.0)
sensorMeasure.setVal(0, 0.0)
sensorMeasure.setVal(0, 2.0)
sensorMeasure.setVal(0, 0.3)
sensorMeasure.setVal(0, 0.2)
sensorMeasure.setVal(0, 0.5)

sensorIndex = generator.getSensorsModel().getSensorIndex(iDynTree.SIX_AXIS_FORCE_TORQUE, 'base_to_arm')
generator.getSensorsMeasurements().setMeasurement(iDynTree.SIX_AXIS_FORCE_TORQUE, sensorIndex, sensorMeasure)

# get (standard) regressor
regressor = iDynTree.MatrixDynSize(N_OUT, N_PARAMS)
knownTerms = iDynTree.VectorDynSize(N_OUT) # TODO: what parameters are these?
if not generator.computeRegressor(regressor, knownTerms):
    print "Error while computing regressor"

# get subspace basis (for projection to base regressor/parameters)
subspaceBasis = iDynTree.MatrixDynSize()
if not generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
# if not generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
    print "Error while computing basis matrix"

# convert to numpy
YStd = regressor.toNumPy()
B = subspaceBasis.toNumPy()
tau = sensorMeasure.toNumPy()

print "YStd: {}".format(YStd.shape)

# project regressor to base regressor, Y_base = Y_std B
YBase = np.dot(YStd, B)
print "YBase: {}".format(YBase.shape)

# invert equation to get parameter vector from measurements and model + system state values
YBaseInv = np.linalg.pinv(YBase)

# TODO: get jacobian and contact force for each contact point
# assuming zero external forces for fixed base on trunk
#jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
#generator.getFrameJacobian('arm', jacobian)

xBase = YBaseInv*tau #- YBaseInv*jacobian*contactForces
print "The base parameter vector is \n{}".format(xBase)

# project back to standard parameters
xStd = np.dot(B, xBase)
print "The standard parameter vector is \n{}".format(xStd)
