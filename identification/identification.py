#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np; #np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import matplotlib.pyplot as plt

#numeric regression
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
import identificationHelpers

#symbolic regression
from robotran import idinvbar

#TODO: load full model and programmatically cut off chain from certain joints/links, change values
#in urdf?

import argparse
parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--measurements', required=True, type=str, help='the file to load the measurements from')
parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
parser.add_argument('--explain', help='whether to explain parameters', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

class Identification(object):
    def __init__(self):
        self.URDF_FILE = args.model
        self.measurements = np.load(args.measurements)
        self.num_samples = self.measurements['positions'].shape[0]
        self.start_offset = 200
        print 'loaded {} measurement samples (using {})'.format(
                self.num_samples, self.num_samples-self.start_offset)
        self.num_samples-=self.start_offset

        #TODO: get sample frequency from file and use to determine start_offset

        #create generator instance and load model
        self.generator = iDynTree.DynamicsRegressorGenerator()
        self.generator.loadRobotAndSensorsModelFromFile(self.URDF_FILE)
        print 'loaded model {}'.format(self.URDF_FILE)

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
        self.generator.loadRegressorStructureFromString(regrXml)

        self.N_DOFS = self.generator.getNrOfDegreesOfFreedom()
        print '# DOFs: {}'.format(self.N_DOFS)

        # Get the number of outputs of the regressor
        self.N_OUT = self.generator.getNrOfOutputs()
        print '# outputs: {}'.format(self.N_OUT)

        # get initial inertia params (from urdf)
        self.N_PARAMS = self.generator.getNrOfParameters()
        print '# params: {}'.format(self.N_PARAMS)

        self.N_LINKS = self.generator.getNrOfLinks()
        print '# links: {} ({} fake)'.format(self.N_LINKS, self.generator.getNrOfFakeLinks())

        self.gravity_twist = iDynTree.Twist()
        self.gravity_twist.zero()
        self.gravity_twist.setVal(2, -9.81)

        self.jointNames = [self.generator.getDescriptionOfDegreeOfFreedom(dof) for dof in range(0, self.N_DOFS)]

        self.regressor_stack = np.empty(shape=(self.N_DOFS*self.num_samples, self.N_PARAMS))
        self.regressor_stack_sym = np.empty(shape=(self.N_DOFS*self.num_samples, 45))
        self.torques_stack = np.empty(shape=(self.N_DOFS*self.num_samples))

    def computeRegressors(self):
        #test standard model dynamics with DynamicsComputations class
        self.dynTest = False

        #test robotran symbolic regressor code
        self.symbolic_test = False
        sym_time = 0
        num_time = 0

        if self.dynTest:
            dynComp = iDynTree.DynamicsComputations();
            dynComp.loadRobotModelFromFile(self.URDF_FILE);
            gravity = iDynTree.SpatialAcc();
            gravity.setVal(2, -9.81);

        self.torquesEst = list()

        #loop over measurements records (skip some values from the start)
        #and get regressors for each system state
        for row in range(0+self.start_offset, self.num_samples+self.start_offset):
            pos = self.measurements['positions'][row]
            vel = self.measurements['velocities'][row]
            acc = self.measurements['accelerations'][row]
            torq = self.measurements['torques'][row]

            #use zero based again for matrices etc.
            row-=self.start_offset

            # set system state
            q = iDynTree.VectorDynSize.fromPyList(pos)
            dq = iDynTree.VectorDynSize.fromPyList(vel)
            ddq = iDynTree.VectorDynSize.fromPyList(acc)

            if self.dynTest:
                dynComp.setRobotState(q, dq, ddq, gravity)
                """
                regressor = iDynTree.MatrixDynSize(N_DOFS, N_PARAMS)
                ok = dynComp.getDynamicsRegressor(regressor)
                if( not ok ):
                    print "Error in computing the dynamics regressor"
                """

                torques = iDynTree.VectorDynSize(self.N_DOFS)
                baseReactionForce = iDynTree.Wrench()

                # compute id with inverse dynamics
                dynComp.inverseDynamics(torques, baseReactionForce)
                self.torquesEst.append(torques.toNumPy())

            with identificationHelpers.Timer() as t:
                self.generator.setRobotState(q,dq,ddq, self.gravity_twist)  # fixed base
                self.generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromPyList(torq))

                # get (standard) regressor
                regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
                knownTerms = iDynTree.VectorDynSize(self.N_OUT)    #what are known terms useable for?
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print "Error while computing regressor"

                start = self.N_DOFS*row
                YStd = regressor.toNumPy()
                #stack on previous regressors
                np.copyto(self.regressor_stack[start:start+self.N_DOFS], YStd)
                np.copyto(self.torques_stack[start:start+self.N_DOFS], torq)
            num_time += t.interval

            if self.symbolic_test:
                with identificationHelpers.Timer() as t:
                    YSym = np.empty((7,48))
                    pad = [0,0]  #symbolic code expects values for two more (static joints)
                    idinvbar.idinvbar(YSym, np.concatenate([[0], pos, pad]),
                        np.concatenate([[0], vel, pad]), np.concatenate([[0], acc, pad]))
                    tmp = np.delete(YSym, (6,4,1), 1)
                    np.copyto(self.regressor_stack_sym[start:start+self.N_DOFS], tmp)
                sym_time += t.interval

        print('Numeric regressors took %.03f sec.' % num_time)
        if self.symbolic_test:
            print('Symbolic regressors took %.03f sec.' % sym_time)

    def invert(self):
        ## inverse stacked regressors and identify parameter vector

        # get subspace basis (for projection to base regressor/parameters)
        subspaceBasis = iDynTree.MatrixDynSize()
        if not self.generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
        # if not generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
            print "Error while computing basis matrix"

        # convert to numpy arrays
        YStd = self.regressor_stack
        YBaseSym = self.regressor_stack_sym
        tau = self.torques_stack
        B = subspaceBasis.toNumPy()

        print "YStd: {}".format(YStd.shape)
        print "YBaseSym: {}".format(YBaseSym.shape)
        print "tau: {}".format(tau.shape)

        # project regressor to base regressor, Y_base = Y_std*B
        YBase = np.dot(YStd, B)
        print "YBase: {}".format(YBase.shape)

        # invert equation to get parameter vector from measurements and model + system state values
        if self.symbolic_test:
            YBaseSymInv = np.linalg.pinv(YBaseSym)

        YBaseInv = np.linalg.pinv(YBase)
        print "YBaseInv: {}".format(YBaseInv.shape)

        # TODO: get jacobian and contact force for each contact frame (when iDynTree allows it)
        # in order to also use FT sensors in hands and feet
        # assuming zero external forces for fixed base on trunk
        #jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
        #generator.getFrameJacobian('arm', jacobian)

        xBase = np.dot(YBaseInv, tau.T) #- np.sum( YBaseInv*jacobian*contactForces )
        #print "The base parameter vector {} is \n{}".format(xBase.shape, xBase)
        if self.symbolic_test:
            self.xBaseSym = np.dot(YBaseSymInv, tau.T)

        # project back to standard parameters
        self.xStd = np.dot(B, xBase)
        #print "The standard parameter vector {} is \n{}".format(xStd.shape, xStd)

        # thresholding
        #zero_threshold = 0.0001
        #low_values_indices = np.absolute(xStd) < zero_threshold
        #xStd[low_values_indices] = 0  # set all low values to 0
        #TODO: replace zeros with cad values

        #get model parameters
        xStdModel = iDynTree.VectorDynSize(self.N_PARAMS)
        self.generator.getModelParameters(xStdModel)
        self.xStdModel = xStdModel.toNumPy()

        ## generate output

        # estimate torques again with regressor and parameters
        print "xStd: {}".format(self.xStd.shape)
        print "xStdModel: {}".format(self.xStdModel.shape)
    #    tauEst = np.dot(YStd, xStdModel) #idyntree standard regressor and parameters from URDF model
    #    tauEst = np.dot(YStd, xStd)    #idyntree standard regressor and estimated standard parameters
        tauEst = np.dot(YBase, xBase)   #idyntree base regressor and identified base parameters

        if self.symbolic_test:
            tauEst = np.dot(YBaseSym, self.xBaseSym)
            self.xStdSym = np.dot(B, self.xBaseSym)

        #put in list of np vectors for plotting
        if not self.dynTest:
            for i in range(0, tauEst.shape[0]):
                if i % self.N_DOFS == 0:
                    tmp = np.zeros(self.N_DOFS)
                    for j in range(0, self.N_DOFS):
                        tmp[j] = tauEst[i+j]
                    self.torquesEst.append(tmp)

    def explain(self):
        # some pretty printing of parameters
        if(args.explain):
            #optional: convert to COM-relative instead of frame origin-relative (linearized parameters)
            helpers = identificationHelpers.IdentificationHelpers(self.N_PARAMS)
            helpers.paramsFromiDyn2URDF(self.xStd)
            helpers.paramsFromiDyn2URDF(self.xStdModel)

            #collect values for parameters
            description = self.generator.getDescriptionOfParameters()
            idx_p = 0
            lines = list()
            for l in description.replace(r'Parameter ', '#').replace(r'first moment', 'center').split('\n'):
                new = self.xStd[idx_p]
                old = self.xStdModel[idx_p]
                diff = old - new
                lines.append((old, new, diff, l))
                idx_p+=1
                if idx_p == len(self.xStd):
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

    def plot(self):
        colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.94117647,  0.03137255,  0.59607843],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]

        datasets = [
                    ([self.measurements['torques'][self.start_offset:, :]], 'Measured Torques'),
                    ([np.array(self.torquesEst)], 'Estimated Torques'),
                   ]
        T = self.measurements['times'][self.start_offset:]
        for (data, title) in datasets:
            plt.figure()
            plt.title(title)
            for i in range(0, self.N_DOFS):
                for d_i in range(0, len(data)):
                    l = self.jointNames[i] if d_i == 0 else ''  #only put joint names in the legend once
                    plt.plot(T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
            plt.legend(loc='upper left')
        plt.show()
        self.measurements.close()

if __name__ == '__main__':
    #from IPython import embed; embed()

    try:
        identification = Identification()
        identification.computeRegressors()
        identification.invert()
        if(args.explain):
            identification.explain()
        if(args.plot):
            identification.plot()

    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            #open ipdb when an exception happens
            import sys, ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
