#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np; # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import matplotlib.pyplot as plt
from IPython import embed

# numeric regression
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
import identificationHelpers

# symbolic regression
from robotran import idinvbar, invdynabar, delidinvbar

# TODO: load full model and programmatically cut off chain from certain joints/links to allow
# subtree identification

class Identification(object):
    def __init__(self, urdf_file, measurements_file):
        ## options
        self.robotranRegressor = False # use robotran symbolic regressor to estimate torques (else iDyntreee)
        # simulate torques from target values, don't use both
        self.iDynSimulate = False # simulate torque using idyntree (instead of reading measurements)
        self.robotranSimulate = False # simulate torque using robotran (instead of reading measurements)
        # using which parameters to estimate torques for validation. Set to one of
        # ['base', 'std', 'model']
        self.estimateWith = 'base'
        self.useAPriori = False  #use known CAD parameters as a priori knowledge, generates (more) consistent std parameters
        self.start_offset = 200  #how many samples from the begginning of the measurements are skipped
        ## end options

        self.URDF_FILE = urdf_file
        self.measurements = np.load(measurements_file)
        self.num_samples = self.measurements['positions'].shape[0]
        print 'loaded {} measurement samples (using {})'.format(
                self.num_samples, self.num_samples-self.start_offset)
        self.num_samples-=self.start_offset

        # create generator instance and load model
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
        # or use <allJoints/>
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
        self.regressor_stack_sym = np.empty(shape=(self.N_DOFS*self.num_samples, 48))
        self.torques_stack = np.empty(shape=(self.N_DOFS*self.num_samples))
        self.torquesAP_stack = np.empty(shape=(self.N_DOFS*self.num_samples))

        self.helpers = identificationHelpers.IdentificationHelpers(self.N_PARAMS)

    def computeRegressors(self):
        """compute regressors for each time step of the measurement data, stack them"""

        if self.useAPriori:
            print("using a priori parameter data")
        if self.robotranRegressor:
            print("using robotran regressor")
        if self.iDynSimulate:
            print("using iDynTree to simulate robot dynamics")
        if self.robotranSimulate:
            print("using robotran to simulate robot dynamics")
        self.simulate = self.iDynSimulate or self.robotranSimulate
        if not self.simulate:
            print("using torque measurement data")
        print("estimating torques using {} parameters".format(self.estimateWith))

        sym_time = 0
        num_time = 0

        if self.simulate or self.useAPriori:
            dynComp = iDynTree.DynamicsComputations();
            dynComp.loadRobotModelFromFile(self.URDF_FILE);
            gravity = iDynTree.SpatialAcc();
            gravity.zero()
            gravity.setVal(2, -9.81);

        # get model parameters
        xStdModel = iDynTree.VectorDynSize(self.N_PARAMS)
        self.generator.getModelParameters(xStdModel)
        self.xStdModel = xStdModel.toNumPy()

        if self.robotranSimulate or self.robotranRegressor:
            # get urdf model parameters as base parameters (for robotran inverse kinematics)
            xStdModelBary = self.xStdModel.copy()
            self.helpers.paramsLink2Bary(xStdModelBary)
            m = np.zeros(self.N_DOFS+3)   #masses
            l = np.zeros((4, self.N_DOFS+3))  #com positions
            inert = np.zeros((10, 10))   #inertias
            for i in range(0, self.N_DOFS+2):
                m[i+1] = xStdModelBary[i*10]
                l[1, i+1] = xStdModelBary[i*10+1]
                l[2, i+1] = xStdModelBary[i*10+2]
                l[3, i+1] = xStdModelBary[i*10+3]
                inert[1, i+1] = xStdModelBary[i*10+4]     #xx w.r.t. com
                inert[2, i+1] = xStdModelBary[i*10+5]     #xy w.r.t. com
                inert[3, i+1] = xStdModelBary[i*10+6]     #xz w.r.t. com
                inert[4, i+1] = xStdModelBary[i*10+5]     #yx
                inert[5, i+1] = xStdModelBary[i*10+7]     #yy w.r.t. com
                inert[6, i+1] = xStdModelBary[i*10+8]     #yz w.r.t. com
                inert[7, i+1] = xStdModelBary[i*10+6]     #zx
                inert[8, i+1] = xStdModelBary[i*10+8]     #zy
                inert[9, i+1] = xStdModelBary[i*10+9]     #zz w.r.t. com

            # load into new model class for some functions
            model = iDynTree.Model()
            iDynTree.modelFromURDF(self.URDF_FILE, model)

            # get relative link positions from params
            # # (could also just get them from xStdModelBary...)
            d = np.zeros((4,10))  # should be 3 x 7, but invdynabar is funny and uses matlab indexing
            for i in range(1, self.N_DOFS+1):
                j = model.getJoint(i-1)
                # get position relative to parent joint
                l1 = j.getFirstAttachedLink()
                l2 = j.getSecondAttachedLink()
                t = j.getRestTransform(l1, l2)
                p = t.getPosition().toNumPy()
                d[1:4, i+2] = p

            # convert to base parameters with robotran equations
            self.xStdModelAsBase = np.zeros(48)
            delidinvbar.delidinvbar(self.xStdModelAsBase, m, l, inert, d)
            # self.xStdModelAsBase = np.delete(self.xStdModelAsBase, (5,3,0), 0)

        self.tauEstimated = list()
        self.tauMeasured = list()

        """loop over measurements records (skip some values from the start)
           and get regressors for each system state"""
        for row in range(0+self.start_offset, self.num_samples+self.start_offset):
            if self.simulate:
                pos = self.measurements['target_positions'][row]
                vel = self.measurements['target_velocities'][row]
                acc = self.measurements['target_accelerations'][row]
            else:
                # read measurements
                pos = self.measurements['positions'][row]
                vel = self.measurements['velocities'][row]
                acc = self.measurements['accelerations'][row]
                torq = self.measurements['torques'][row]

            # use zero based again for matrices etc.
            row-=self.start_offset

            #test
            """
            for i in range(0,6+1):
                pos[i] = 0
                vel[i] = 0
                acc[i] = 0
            """

            # system state for iDynTree
            q = iDynTree.VectorDynSize.fromPyList(pos)
            dq = iDynTree.VectorDynSize.fromPyList(vel)
            ddq = iDynTree.VectorDynSize.fromPyList(acc)

            if self.robotranRegressor or self.robotranSimulate:
                # system state for robotran
                # convert positions from urdf/idyntree convention to robotran conventions and
                # joint     |  zero at   | direction
                # 0 LShSag  |  20deg     |  1
                # 1 LShLat  |  -41deg    |  1
                # rest      |  0deg      |  1
                #
                # also from right to left arm (model is for right arm)
                # joint         | direction
                # 1 LShLat        | -1
                # 2 LShYaw        | -1
                # 4 LForearmPlate | -1
                # 6 LWrj2         | -1
                pos[0]+=np.deg2rad(20)
                pos[1]-=np.deg2rad(41)
                pos[1]*=-1; vel[1]*=-1; acc[1]*=-1
                pos[2]*=-1; vel[2]*=-1; acc[2]*=-1
                pos[4]*=-1; vel[4]*=-1; acc[4]*=-1
                pos[6]*=-1; vel[6]*=-1; acc[6]*=-1

            if self.iDynSimulate or self.useAPriori:
                # calc torques with iDynTree dynamicsComputation class
                dynComp.setRobotState(q, dq, ddq, gravity)

                torques = iDynTree.VectorDynSize(self.N_DOFS)
                baseReactionForce = iDynTree.Wrench()   # assume zero for fixed base, otherwise use e.g. imu data

                # compute inverse dynamics with idyntree (simulate)
                dynComp.inverseDynamics(torques, baseReactionForce)
                if self.useAPriori:
                    torqAP = torques.toNumPy()
                if self.iDynSimulate:
                    torq = torques.toNumPy()

            if self.robotranSimulate:
                # get dynamics from robotran equations
                torq = np.zeros(self.N_DOFS)
                pad = [0,0]

                invdynabar.invdynabar(torq, np.concatenate(([0], pad, pos)), np.concatenate(([0], pad, vel)),
                    np.concatenate(([0], pad, acc)), np.concatenate(([0], self.xStdModelAsBase)), d)

            start = self.N_DOFS*row
            # use symobolic regressor to get numeric regressor matrix
            if self.robotranRegressor:
                with identificationHelpers.Timer() as t:
                    YSym = np.empty((7,48))
                    pad = [0,0]  # symbolic code expects values for two more (static joints)
                    idinvbar.idinvbar(YSym, np.concatenate([[0], pad, pos]),
                        np.concatenate([[0], pad, vel]), np.concatenate([[0], pad, acc]), d)
                    #YSym = np.delete(YSym, (5,3,0), 1)   # remove unnecessary columns (numbers from generated code)
                    np.copyto(self.regressor_stack_sym[start:start+self.N_DOFS], YSym)
                sym_time += t.interval
            else:
                # get numerical regressor
                with identificationHelpers.Timer() as t:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)  # fixed base
                    self.generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromPyList(torq))

                    # get (standard) regressor
                    regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
                    knownTerms = iDynTree.VectorDynSize(self.N_OUT)    # what are known terms useable for?
                    if not self.generator.computeRegressor(regressor, knownTerms):
                        print "Error during numeric computation of regressor"

                    YStd = regressor.toNumPy()
                    # stack on previous regressors
                    np.copyto(self.regressor_stack[start:start+self.N_DOFS], YStd)
                num_time += t.interval

            np.copyto(self.torques_stack[start:start+self.N_DOFS], torq)
            if self.useAPriori:
                np.copyto(self.torquesAP_stack[start:start+self.N_DOFS], torqAP)

        if self.robotranRegressor:
            print('Symbolic regressors took %.03f sec.' % sym_time)
        else:
            print('Numeric regressors took %.03f sec.' % num_time)

    def getBaseParameters(self):
        """inverse stacked regressors and identify parameter vector."""

        # get subspace basis (for projection to base regressor/parameters)
        subspaceBasis = iDynTree.MatrixDynSize()
        if not self.generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
        # if not self.generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
            print "Error while computing basis matrix"

        # convert stacks
        self.YStd = self.regressor_stack

        if self.useAPriori:
            # get torque delta to identify with
            self.tau = self.torques_stack - self.torquesAP_stack
        else:
            self.tau = self.torques_stack
        B = subspaceBasis.toNumPy()

        print("YStd: {}".format(self.YStd.shape))
        print("tau: {}".format(self.tau.shape))

        if self.robotranRegressor:
            self.YBase = self.regressor_stack_sym
        else:
            # project regressor to base regressor, Y_base = Y_std*B
            self.YBase = np.dot(self.YStd, B)
            print("YBase: {}".format(self.YBase.shape))

        # invert equation to get parameter vector from measurements and model + system state values
        self.YBaseInv = np.linalg.pinv(self.YBase)
        print("YBaseInv: {}".format(self.YBaseInv.shape))

        # TODO: get jacobian and contact force for each contact frame (when iDynTree allows it)
        # in order to also use FT sensors in hands and feet
        # assuming zero external forces for fixed base on trunk
        # jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
        # self.generator.getFrameJacobian('arm', jacobian)

        self.xBase = np.dot(self.YBaseInv, self.tau.T) # - np.sum( YBaseInv*jacobian*contactForces )

        # project back to standard parameters
        if self.robotranRegressor:
            # delete some unimportant columns for getting the same amount of base params as iDynTree
            # so we can use the same projection back to the standard parameters
            # (is there a robotran way of doing this?)
            xBaseReduced = np.dot(np.linalg.pinv(np.delete(self.YBase, (5,3,1), 1)), self.tau.T)
            self.xStd = np.dot(B, xBaseReduced)
        else:
            self.xStd = np.dot(B, self.xBase)

        # get estimated parameters from estimated error (add a priori knowledge)
        if self.useAPriori:
            # TODO: something about the base params must be wrong
            if self.robotranRegressor:
                self.xBase = self.xBase + self.xStdModelAsBase   #both param vecs barycentric
            else:
                self.xBase = self.xBase + np.dot(B.T, self.xStdModel)   #both param vecs link relative linearized

            self.xStd = self.xStd + self.xStdModel

        # print "The base parameter vector {} is \n{}".format(xBase.shape, xBase)
        # print "The standard parameter vector {} is \n{}".format(self.xStd.shape, self.xStd)

        # thresholding
        # zero_threshold = 0.0001
        # low_values_indices = np.absolute(self.xStd) < zero_threshold
        # self.xStd[low_values_indices] = self.xStdModel[low_values_indices] # replace close to zeros with cad values

        self.num_base_params = self.xBase.size

    def estimateTorques(self):
        """ get torque estimations, prepare for plotting """

        if self.robotranRegressor:
            if self.estimateWith is 'base':
                tauEst = np.dot(self.YBase, self.xBase)
            elif self.estimateWith is 'model':
                tauEst = np.dot(self.YBase, self.xStdModelAsBase)
            elif self.estimateWith is 'std':
                print("Error: I don't have a standard regressor with symbolic equations.")
                sys.exit(-1)
            else:
                print("unknown type of parameters: {}".format(self.estimateWith))
        else:
            # estimate torques again with regressor and parameters
            if self.estimateWith is 'model':
                tauEst = np.dot(self.YStd, self.xStdModel) # idyntree standard regressor and parameters from URDF model
            elif self.estimateWith is 'base':
                tauEst = np.dot(self.YBase, self.xBase)   # idyntree base regressor and identified base parameters
            elif self.estimateWith is 'std':
                tauEst = np.dot(self.YStd, self.xStd)    # idyntree standard regressor and estimated standard parameters
            else:
                print("unknown type of parameters: {}".format(self.estimateWith))

        # put estimated torques in list of np vectors for plotting (NUM_SAMPLES*N_DOFSx1) -> (NUM_SAMPLESxN_DOFS)
        self.tauEstimated = np.reshape(tauEst, (self.num_samples, self.N_DOFS))

        if self.simulate:
            if self.useAPriori:
                # use original measurements, not delta
                tau = self.torques_stack

            self.tauMeasured = np.reshape(self.tau, (self.num_samples, self.N_DOFS))
        else:
            self.tauMeasured = self.measurements['torques'][self.start_offset:, :]

    def getEssentialParameters(self):
        """
        iteratively get essential parameters from previously identified base parameters.

        based on Gautier et al., Identification of Consistent Standard Dynamic Parameters (...), 2013
        """
        self.estimateTorques()

        if self.estimateWith is not 'base':
            print("Warning: not getting essential parameters, need to be estimating with base params")
            return

        r_sigma = 15    #target ratio of parameters' relative std deviation
        cnt = 0
        not_essential_idx = list()

        while 1:
            # get standard deviation of measurement and modeling error \sigma_{rho}^2
            # TODO: use these for weighting of linear least squares identification of base parameters
            sigma_rho = np.square(np.linalg.norm(self.tauMeasured-self.tauEstimated))/(self.num_samples-self.num_base_params)
            if cnt is 0:
                self.sigma_rho_base = sigma_rho.copy()

            # get standard deviation \sigma_{x} (of the estimated parameter vector x)
            C_xx = np.dot(sigma_rho, (np.linalg.inv(np.dot(self.YBase.T, self.YBase))))
            sigma_x = np.diag(C_xx)

            # get relative standard deviation
            p_sigma_x = np.sqrt(sigma_x)
            for i in range(0, p_sigma_x.size):
                if np.linalg.norm(self.xBase[i]) != 0:
                    p_sigma_x[i] /= np.linalg.norm(self.xBase[i])

            ratio = np.max(p_sigma_x)/np.min(p_sigma_x)
            print "min-max ratio of relative stddevs: {}".format(ratio)

            #while condition moved here
            if ratio < r_sigma:
                break

            #cancel parameter with largest deviation
            param_idx = np.argmax(p_sigma_x)
            not_essential_idx.append(param_idx)
            self.xBase[param_idx] = 0
            self.num_base_params -= 1
            #self.xBase = np.delete(self.xBase, param_idx)
            #self.num_base_params = self.xBase.size
            #self.YBase = np.delete(self.YBase, param_idx, 1)

            # get new estimations with updated parameters
            self.estimateTorques()
            cnt+=1

        self.essentialIdx = [x for x in range(0,self.N_PARAMS) if x not in not_essential_idx]
        print "Got {} essential parameters".format(len(self.essentialIdx))

    def output(self):
        """Do some pretty printing of parameters."""
        # TODO: write params to file/urdf file

        from colorama import Fore, Back, Style

        # convert params to COM-relative instead of frame origin-relative (linearized parameters)
        if not self.robotranRegressor:
            self.helpers.paramsLink2Bary(self.xStd)
        self.helpers.paramsLink2Bary(self.xStdModel)

        # collect values for parameters
        description = self.generator.getDescriptionOfParameters()
        idx_p = 0
        lines = list()
        print Style.RESET_ALL
        for d in description.replace(r'Parameter ', '# ').replace(r'first moment', 'center').split('\n'):
            new = self.xStd[idx_p]
            old = self.xStdModel[idx_p]
            diff = old - new
            #print beginning of each link block in green
            if idx_p % 10 == 0:
                d = Fore.GREEN + d + Style.RESET_ALL
            lines.append((old, new, diff, d))
            idx_p+=1
            if idx_p == len(self.xStd):
                break

        column_widths = [15, 15, 7, 45]   # widths of the columns
        precisions = [8, 8, 3, 0]         # numerical precision

        # print column header
        template = ''
        for w in range(0, len(column_widths)):
            template += '|{{{}:{}}}'.format(w, column_widths[w])
        print template.format("Model", "Approx", "Error", "Description")

        # print values/description
        template = ''
        for w in range(0, len(column_widths)):
            if(type(lines[0][w]) == str):
                # strings don't have precision
                template += '|{{{}:{}}}'.format(w, column_widths[w])
            else:
                template += '|{{{}:{}.{}f}}'.format(w, column_widths[w], precisions[w])
        for l in lines:
            print template.format(*l)

    def plot(self):
        """Display some torque plots."""

        colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.84705882,  0.        ,  0.1254902 ],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]

        datasets = [
                    ([self.tauMeasured], 'Measured Torques'),
                    ([self.tauEstimated], 'Estimated Torques'),
                   ]
        #print "torque diff: {}".format(self.tauMeasured - self.tauEstimated)

        T = self.measurements['times'][self.start_offset:]
        for (data, title) in datasets:
            plt.figure()
            plt.title(title)
            for i in range(0, self.N_DOFS):
                for d_i in range(0, len(data)):
                    l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                    plt.plot(T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
            leg = plt.legend(loc='best', fancybox=True, fontsize=10)
            leg.draggable()
        plt.show()
        self.measurements.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--measurements', required=True, type=str, help='the file to load the measurements from')
    parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
    parser.add_argument('--explain', help='whether to explain parameters', action='store_true')
    parser.set_defaults(plot=False, explain=True)
    args = parser.parse_args()

    identification = Identification(args.model, args.measurements)
    identification.computeRegressors()
    identification.getBaseParameters()
    identification.getEssentialParameters()

    if(args.explain):
        identification.output()
    if(args.plot):
        identification.plot()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            # open ipdb when an exception happens
            import sys, ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
