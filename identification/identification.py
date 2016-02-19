#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from identificationHelpers import IdentificationHelpers
import numpy as np; #np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import matplotlib.pyplot as plt

#TODO: load full model and programmatically cut off chain from certain joints/links, get back urdf?

import argparse
parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--measurements', required=True, type=str, help='the file to load the measurements from')
parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
parser.add_argument('--explain', help='whether to explain parameters', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

URDF_FILE = args.model

measurements = np.load(args.measurements)
num_samples = measurements['positions'].shape[0]
print 'loaded {} measurement samples'.format(num_samples)

#create generator instance and load model
generator = iDynTree.DynamicsRegressorGenerator()
generator.loadRobotAndSensorsModelFromFile(URDF_FILE)
print 'loaded model {}'.format(URDF_FILE)

# define what regressor type to use

regrXml = '''
<regressor>
  <jointTorqueDynamics>
    <joints>
        <joint>LShSag</joint>
        <joint>LShLat</joint>
        <joint>LShYaw</joint>
        <joint>LElbj</joint>
        <joint>LForearmPlate</joint>
        <joint>LWrj1</joint>
        <joint>LWrj2</joint>
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
print '# params: {}'.format(N_PARAMS)

N_LINKS = generator.getNrOfLinks()
print '# links: {} ({} fake)'.format(N_LINKS, generator.getNrOfFakeLinks())

gravity_twist = iDynTree.Twist()
gravity_twist.zero()
gravity_twist.setVal(2, -9.81)

jointNames = [generator.getDescriptionOfDegreeOfFreedom(dof) for dof in range(0, N_DOFS)]

regressor_stack = np.empty(shape=(N_DOFS*num_samples, N_PARAMS))
torques_stack = np.empty(shape=(N_DOFS*num_samples))

#loop over measurements records
if(True):
    for row in range(0, num_samples):
        pos = measurements['positions'][row]
        vel = measurements['velocities'][row]
        acc = measurements['accelerations'][row]
        torq = measurements['torques'][row]

        # set system state
        q = iDynTree.VectorDynSize(N_DOFS)
        dq = iDynTree.VectorDynSize(N_DOFS)
        ddq = iDynTree.VectorDynSize(N_DOFS)

        for dof in range(N_DOFS):
            q.setVal(dof, pos[dof])
            dq.setVal(dof, vel[dof])
            ddq.setVal(dof, acc[dof])

        generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromPyList(torq))
        generator.setRobotState(q,dq,ddq, gravity_twist)  # fixed base, base acceleration etc. =0

        # get (standard) regressor
        regressor = iDynTree.MatrixDynSize(N_OUT, N_PARAMS)
        knownTerms = iDynTree.VectorDynSize(N_OUT)
        if not generator.computeRegressor(regressor, knownTerms):
            print "Error while computing regressor"

        YStd = regressor.toNumPy()

        #stack on previous regressors
        start = N_DOFS*row
        np.copyto(regressor_stack[start:start+N_DOFS], YStd)
        np.copyto(torques_stack[start:start+N_DOFS], torq)

## end measurements loop
measurements.close()

## inverse stacked regressors and identify parameter vector

# get subspace basis (for projection to base regressor/parameters)
subspaceBasis = iDynTree.MatrixDynSize()
if not generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
# if not generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
    print "Error while computing basis matrix"

# convert to numpy arrays
YStd = regressor_stack
tau = torques_stack
B = subspaceBasis.toNumPy()

print "YStd: {}".format(YStd.shape)
print "tau: {}".format(tau.shape)

# project regressor to base regressor, Y_base = Y_std B
YBase = np.dot(YStd, B)
print "YBase: {}".format(YBase.shape)

# invert equation to get parameter vector from measurements and model + system state values
YBaseInv = np.linalg.pinv(YBase)
print "YBaseInv: {}".format(YBaseInv.shape)

# TODO: get jacobian and contact force for each contact frame (when iDynTree allows it)
# in order to also use FT sensors in hands and feet
# assuming zero external forces for fixed base on trunk
#jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
#generator.getFrameJacobian('arm', jacobian)

xBase = np.dot(YBaseInv, tau.T) #- np.sum( YBaseInv*jacobian*contactForces )
print "The base parameter vector {} is \n{}".format(xBase.shape, xBase)

# project back to standard parameters
xStd = np.dot(B, xBase)
print "The standard parameter vector {} is \n{}".format(xStd.shape, xStd)

# thresholding
#zero_threshold = 0.0001
#low_values_indices = np.absolute(xStd) < zero_threshold
#xStd[low_values_indices] = 0  # set all low values to 0
#TODO: replace zeros with cad values

#get model parameters
xStdModel = iDynTree.VectorDynSize(N_PARAMS)
generator.getModelParameters(xStdModel)

## generate output

#show COM-relative instead of frame origin-relative (linearized parameters)
#helpers = IdentificationHelpers(N_PARAMS)
#helpers.paramsFromiDyn2URDF(xStdModel.toNumPy())
#helpers.paramsFromiDyn2URDF(xStd)

# TODO: save to urdf with new parameters

# some pretty printing of parameters
if(args.explain):
    #collect values for parameters
    description = generator.getDescriptionOfParameters()
    idx_p = 0
    lines = list()
    for l in description.replace(r'Parameter ', '#').replace(r'first moment', 'center').split('\n'):
        new = xStd[idx_p]
        old = xStdModel.getVal(idx_p)
        diff = old - new
        lines.append((old, new, diff, l))
        idx_p+=1
        if idx_p == len(xStd):
            break

    column_widths = [15, 15, 7, 45]   #widths of the columns
    precisions = [8, 8, 3, 0]         #numerical precision

    #print column header
    template = ''
    for w in range(0, len(column_widths)):
        template += '|{{{}:{}}}'.format(w, column_widths[w])
    print template.format("Model", "Approx", "Error", "Description")

    #print values/description
    template = ''
    for w in range(0, len(column_widths)):
        if(type(lines[0][w]) == str):
            template += '|{{{}:{}}}'.format(w, column_widths[w])
        else:
            template += '|{{{}:{}.{}f}}'.format(w, column_widths[w], precisions[w])
    for l in lines:
        print template.format(*l)
