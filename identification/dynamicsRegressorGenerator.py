#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import iDynTree
import numpy as np; np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import matplotlib.pyplot as plt

URDF_FILE = '../urdf/simple_robot.urdf'
#URDF_FILE = '../urdf/bigman.urdf'
#TODO: load full model and programmatically cut off chain from certain joints/links, get back urdf?
#URDF_FILE = '../urdf/bigman_left_arm.urdf'

measurements = np.load("../data/LARM/SIM/measurements.npz")
num_samples = measurements['positions'].shape[0]
print 'loaded {} samples'.format(num_samples)

#some iDynTree helpers
def vecToNumPy(self):
    return np.fromstring(self.toString(), sep=' ')
def pyToVec(self, vec):
    out = iDynTree.VectorDynSize(len(vec))
    for v in range(0,len(vec)):
        out.setVal(v, vec[v])
    return out
iDynTree.VectorDynSize.toNumPy = vecToNumPy
iDynTree.VectorDynSize.fromPy = pyToVec
iDynTree.Wrench.toNumPy = vecToNumPy
iDynTree.Twist.toNumPy = vecToNumPy

def matToNumPy(self):
    return np.fromstring(self.toString(), sep=' ').reshape(self.rows(), self.cols())
iDynTree.MatrixDynSize.toNumPy = matToNumPy

#create generator instance and load model
generator = iDynTree.DynamicsRegressorGenerator()
generator.loadRobotAndSensorsModelFromFile(URDF_FILE)
print 'loaded model {}'.format(URDF_FILE)

# define what regressor type to use
#(TODO: in order to skip links (that are just a fixed part of the chain, no joints)
#add something like <ignoredLink>r_wrist_1</ignoredLink>' for all the fixed joints)
#ideally try automatically

#regrXml = '''
#<regressor>
# <subtreeBaseDynamics>'
#   <FTSensorLink>arm</FTSensorLink>'
# </subtreeBaseDynamics>'
#</regressor>'''
regrXml = '''
<regressor>
  <jointTorqueDynamics>
    <joints>
        <joint>base_to_arm</joint>
    </joints>
  </jointTorqueDynamics>
</regressor>'''
#or use <allJoints/>
generator.loadRegressorStructureFromString(regrXml)

N_DOFS = generator.getNrOfDegreesOfFreedom()
print '# DOFs: {}'.format(N_DOFS)

# Get the number of outputs of the regressor
N_OUT = generator.getNrOfOutputs()
print '# outputs: {}'.format(N_OUT)

# get initial inertia params (from urdf)
N_PARAMS = generator.getNrOfParameters()
cadParams = iDynTree.VectorDynSize(N_PARAMS)
generator.getModelParameters(cadParams)
print '# params: {}'.format(N_PARAMS)

gravity_twist = iDynTree.Twist()
gravity_twist.zero()
gravity_twist.setVal(2, -9.81)

jointNames = [generator.getDescriptionOfDegreeOfFreedom(dof) for dof in range(0, N_DOFS)]

M = measurements['torques']
'''
plt.plot(M[:, 0], label="tq_LShSag")
plt.plot(M[:, 1], label="tq_LShLat")
plt.plot(M[:, 2], label="tq_LShYaw")
plt.plot(M[:, 3], label="tq_LElbj")
plt.plot(M[:, 4], label="tq_LForearmPlate")
plt.plot(M[:, 5], label="tq_LWrj1")
plt.plot(M[:, 6], label="tq_LWrj2")
plt.legend(loc='lower right')
plt.show()
'''

regressor_stack = np.empty(shape=(N_DOFS*num_samples, N_PARAMS))
torques_stack = np.empty(shape=(num_samples, N_DOFS))

#loop over measurements records
if(False):
    for row in range(0, num_samples):
        #q_LShSag, q_LShLat, q_LShYaw, q_LElbj, q_LForearmPlate, q_LWrj1, q_LWrj2 \
        #qdot_LShSag, qdot_LShLat, qdot_LShYaw, qdot_LElbj, qdot_LForearmPlate, qdot_LWrj1, qdot_LWrj2 \
        #tq_LShSag, tq_LShLat, tq_LShYaw, tq_LElbj, tq_LForearmPlate, tq_LWrj1, tq_LWrj2 \
        #time     = record
        pos = measurements['positions'][row]
        vel = measurements['velocities'][row]
        torq = measurements['torques'][row]

        # set system state
        q = iDynTree.VectorDynSize(N_DOFS)
        dq = iDynTree.VectorDynSize(N_DOFS)
        ddq = iDynTree.VectorDynSize(N_DOFS)

        for dof in range(N_DOFS):
            q.setVal(dof, pos[dof])
            dq.setVal(dof, vel[dof])
            ddq.setVal(dof, 0.001) #TODO: get acc[dof] from simulator/differentiate
            #generator.setTorqueSensorMeasurement(dof, torq[dof])

        generator.setTorqueSensorMeasurement(iDynTree.fromPy(torq))

        generator.setRobotState(q,dq,ddq, gravity_twist)  # fixed base, base acceleration etc. =0

        # get (standard) regressor
        regressor = iDynTree.MatrixDynSize(N_OUT, N_PARAMS)
        knownTerms = iDynTree.VectorDynSize(N_OUT)
        if not generator.computeRegressor(regressor, knownTerms):
            print "Error while computing regressor"

        YStd = regressor.toNumPy()

        #stack on previous regressors
        start = N_DOFS*row
        np.copyto(regressor_stack[start:start+6], YStd)
        np.copyto(torques_stack[row], torq)

## end measurements loop
measurements.close()

#single sample stuff, remove soon

# set system state
q = iDynTree.VectorDynSize(N_DOFS)
dq = iDynTree.VectorDynSize(N_DOFS)
ddq = iDynTree.VectorDynSize(N_DOFS)

#dummy data
#for dof in range(N_DOFS):
#    q.setVal(dof, 1.0)
#    dq.setVal(dof, 0.01)
#    ddq.setVal(dof, 0.001)

generator.setRobotState(q,dq,ddq, gravity_twist)  # fixed base, base acceleration etc. =0

# set torque measurements from experiment data
sensorMeasure = iDynTree.VectorDynSize(N_DOFS)
sensorMeasure.setVal(0, 0.0)
#sensorMeasure = iDynTree.Wrench()
#sensorMeasure.setVal(0, 0.0)
#sensorMeasure.setVal(1, 2.0)
#sensorMeasure.setVal(2, 5.0)
#sensorMeasure.setVal(3, 0.3)
#sensorMeasure.setVal(4, 10.2)
#sensorMeasure.setVal(5, 1.5)

#sensorIndex = generator.getSensorsModel().getSensorIndex(iDynTree.SIX_AXIS_FORCE_TORQUE, 'base_to_arm')
#generator.getSensorsMeasurements().setMeasurement(iDynTree.SIX_AXIS_FORCE_TORQUE, sensorIndex, sensorMeasure)
generator.setTorqueSensorMeasurement(sensorMeasure)

# get (standard) regressor
regressor = iDynTree.MatrixDynSize(N_OUT, N_PARAMS) #Y
knownTerms = iDynTree.VectorDynSize(N_OUT) #corresponds to measured torques in this case
if not generator.computeRegressor(regressor, knownTerms):
    print "Error while computing regressor"

print 'regressor: {}'.format(regressor.toString())

#TODO: from here, use stacked regressors when regressors are ready

# get subspace basis (for projection to base regressor/parameters)
subspaceBasis = iDynTree.MatrixDynSize()
if not generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
# if not generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
    print "Error while computing basis matrix"

# convert to numpy arrays
YStd = regressor.toNumPy()
B = subspaceBasis.toNumPy()
tau = sensorMeasure.toNumPy()

print "YStd: {}".format(YStd.shape)

# project regressor to base regressor, Y_base = Y_std B
YBase = np.dot(YStd, B)
print "YBase: {}".format(YBase.shape)

# invert equation to get parameter vector from measurements and model + system state values
YBaseInv = np.linalg.pinv(YBase)
print "YBaseInv: {}".format(YBaseInv.shape)

# TODO: get jacobian and contact force for each contact frame (when iDynTree allows it)
# assuming zero external forces for fixed base on trunk
#jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
#generator.getFrameJacobian('arm', jacobian)

xBase = YBaseInv*tau #- np.sum( YBaseInv*jacobian*contactForces )
print "The base parameter vector is \n{}".format(xBase)

# project back to standard parameters
# TODO: why are these matrices instead of vectors? need to stack columns?
xStd = np.dot(B, xBase)
print "The standard parameter vector is \n{}".format(xStd)
