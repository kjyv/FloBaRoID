#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys
import numpy as np
import numpy.linalg as la
#import numexpr as ne
import scipy.linalg as sla

import math
import matplotlib; matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from IPython import embed

# numeric regression
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
import identificationHelpers

# symbolic regression
from robotran.left_arm import idinvbar, invdynabar, delidinvbar

# Referenced papers:
# Gautier, 2013: Identification of Consistent Standard Dynamic Parameters of Industrial Robots
# Gautier, 1990: Numerical Calculation of the base Inertial Parameters of Robots
# Pham, 1991: Essential Parameters of Robots

# TODO: load full model and programatically cut off chain from certain joints/links to allow
# subtree identification
# TODO: write params to file/urdf file (ideally don't rewrite from model but only change values
# in original xml tree)
# TODO: add/use contact forces

class Identification(object):
    def __init__(self, urdf_file, measurements_file):
        ## options

        # number of samples to use
        # (Khalil recommends about 500 times number of parameters to identify)
        self.start_offset = 200  #how many samples from the begginning of the measurements are skipped

        self.skip_samples = 2    #how many values to skip before using the next sample

        # use robotran symbolic regressor to estimate torques (else iDynTree)
        self.robotranRegressor = 0

        # simulate torques from target values, don't use both
        self.iDynSimulate = 1 # simulate torque using idyntree (instead of reading measurements)
        self.robotranSimulate = 0 # simulate torque using robotran (instead of reading measurements)
        self.addNoise = 0   #add some artificial zero-mean white noise to the 'measured' torques

        # using which parameters to estimate torques for validation. Set to one of
        # ['base', 'std', 'std_direct', 'model']
        self.estimateWith = 'std'

        # use known CAD parameters as a priori knowledge, generates (more) consistent std parameters
        self.useAPriori = 1

        # use weighted least squares(WLS) instead of ordinary least squares
        self.useWLS = 0

        # whether to identify and use direct standard with essential parameters
        self.useEssentialParams = 1

        if self.useAPriori:
            print("using a priori parameter data")
        if self.robotranRegressor:
            print("using robotran regressor")
            if self.useEssentialParams:
                print("can't get essential parameters with robotran regressor, aborting.")
                sys.exit(-1)
            if self.estimateWith in ['std_direct']:
                print("can't get standard parameters directly with robotran regressor, aborting.")
                sys.exit(-1)
        if self.iDynSimulate:
            print("using iDynTree to simulate robot dynamics")
        if self.robotranSimulate:
            print("using robotran to simulate robot dynamics")
        self.simulate = self.iDynSimulate or self.robotranSimulate
        if not self.simulate:
            print("using torque measurement data")
        print("estimating torques using {} parameters".format(self.estimateWith))
        if self.useWLS:
            print("using weighted least squares")
        if self.useEssentialParams:
            print("identifying essential parameters")

        ## end options

        with identificationHelpers.Timer() as t:
            #almost zero threshold for SVD and QR
            self.min_tol = 1e-3

            self.URDF_FILE = urdf_file
            self.measurements = np.load(measurements_file)
            self.num_samples = (self.measurements['positions'].shape[0]-self.start_offset)/(self.skip_samples+1)
            print 'loaded {} measurement samples (using {})'.format(
                self.measurements['positions'].shape[0], self.num_samples)

            # create generator instance and load model
            self.generator = iDynTree.DynamicsRegressorGenerator()
            self.generator.loadRobotAndSensorsModelFromFile(self.URDF_FILE)
            print 'loaded model {}'.format(self.URDF_FILE)

            # define what regressor type to use and options for it
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

            self.regressor_stack = np.zeros(shape=(self.N_DOFS*self.num_samples, self.N_PARAMS))
            if self.robotranRegressor:
                self.regressor_stack_sym = np.zeros(shape=(self.N_DOFS*self.num_samples, 45))
            self.torques_stack = np.zeros(shape=(self.N_DOFS*self.num_samples))
            self.torquesAP_stack = np.zeros(shape=(self.N_DOFS*self.num_samples))

            self.helpers = identificationHelpers.IdentificationHelpers(self.N_PARAMS)

        print("Initialization took %.03f sec." % t.interval)

    def computeRegressors(self):
        """compute regressors for each time step of the measurement data, stack them"""

        sym_time = 0
        num_time = 0
        simulate_time = 0

        with identificationHelpers.Timer() as t:
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
            if self.estimateWith is 'model':
                self.xStd = self.xStdModel

            if self.robotranSimulate or self.robotranRegressor:
                # get urdf model parameters as base parameters (for robotran inverse kinematics)
                xStdModelBary = self.xStdModel.copy()
                self.helpers.paramsLink2Bary(xStdModelBary)
                m = np.zeros(self.N_DOFS+3)   #masses
                l = np.zeros((4, self.N_DOFS+3))  #com positions
                inert = np.zeros((10, 10))   #inertias
                for i in range(0, self.N_DOFS+1):
                    m[i+2] = xStdModelBary[i*10]
                    l[1, i+2] = xStdModelBary[i*10+1]
                    l[2, i+2] = xStdModelBary[i*10+2]
                    l[3, i+2] = xStdModelBary[i*10+3]
                    inert[1, i+2] = xStdModelBary[i*10+4]     #xx w.r.t. com
                    inert[2, i+2] = xStdModelBary[i*10+5]     #xy w.r.t. com
                    inert[3, i+2] = xStdModelBary[i*10+6]     #xz w.r.t. com
                    inert[4, i+2] = xStdModelBary[i*10+5]     #yx
                    inert[5, i+2] = xStdModelBary[i*10+7]     #yy w.r.t. com
                    inert[6, i+2] = xStdModelBary[i*10+8]     #yz w.r.t. com
                    inert[7, i+2] = xStdModelBary[i*10+6]     #zx
                    inert[8, i+2] = xStdModelBary[i*10+8]     #zy
                    inert[9, i+2] = xStdModelBary[i*10+9]     #zz w.r.t. com

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
                    trans = j.getRestTransform(l1, l2)
                    p = trans.getPosition().toNumPy()
                    d[1:4, i+2] = p

                # convert to base parameters with robotran equations
                self.xStdModelAsBase = np.zeros(48)
                delidinvbar.delidinvbar(self.xStdModelAsBase, m=None, l=None, In=None, d=None)
                self.xStdModelAsBaseFull = self.xStdModelAsBase.copy()
                self.xStdModelAsBase = np.delete(self.xStdModelAsBase, (5,3,0), 0)

            self.tauEstimated = list()
            self.tauMeasured = list()
        print("Init for computing regressors took %.03f sec." % t.interval)

        """loop over measurements records (skip some values from the start)
           and get regressors for each system state"""
        for row in range(0, self.num_samples):
            # TODO: this takes multiple seconds because of lazy loading, try preload or use other
            # data format
            with identificationHelpers.Timer() as t:
                m_idx = self.start_offset+(row*(self.skip_samples)+row)
                if self.simulate:
                    pos = self.measurements['target_positions'][m_idx]
                    vel = self.measurements['target_velocities'][m_idx]
                    acc = self.measurements['target_accelerations'][m_idx]
                else:
                    # read measurements
                    pos = self.measurements['positions'][m_idx]
                    vel = self.measurements['velocities'][m_idx]
                    acc = self.measurements['accelerations'][m_idx]
                    torq = self.measurements['torques'][m_idx]

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
                    #pos[0]+=np.deg2rad(20)
                    pos[1]-=np.deg2rad(41)

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

                    invdynabar.invdynabar(torq, np.concatenate(([0], pad, pos)),
                                          np.concatenate(([0], pad, vel)),
                                          np.concatenate(([0], pad, acc)),
                                          np.concatenate(([0], self.xStdModelAsBaseFull)), d=None)
                if self.addNoise:
                    torq += np.random.normal(0,1.0)*(torq*0.01)
            simulate_time += t.interval

            start = self.N_DOFS*row
            # use symobolic regressor to get numeric regressor matrix (base)
            if self.robotranRegressor:
                with identificationHelpers.Timer() as t:
                    YSym = np.zeros((7,45))
                    pad = [0,0]  # symbolic code expects values for two more (static joints)
                    idinvbar.idinvbar(YSym, np.concatenate([[0], pad, pos]),
                                      np.concatenate([[0], pad, vel]),
                                      np.concatenate([[0], pad, acc]), d=None)
                    np.copyto(self.regressor_stack_sym[start:start+self.N_DOFS], YSym)
                sym_time += t.interval
            else:
                # get numerical regressor (std)
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

            with identificationHelpers.Timer() as t:
                np.copyto(self.torques_stack[start:start+self.N_DOFS], torq)
                if self.useAPriori:
                    np.copyto(self.torquesAP_stack[start:start+self.N_DOFS], torqAP)

                if self.useAPriori:
                    # get torque delta to identify with
                    self.tau = self.torques_stack - self.torquesAP_stack
                else:
                    self.tau = self.torques_stack
            simulate_time+=t.interval

        if not self.robotranRegressor:
            self.YStd = self.regressor_stack

        print('Simulation for regressors took %.03f sec.' % simulate_time)

        if self.robotranRegressor:
            print('Symbolic regressors took %.03f sec.' % sym_time)
        else:
            print('Numeric regressors took %.03f sec.' % num_time)

    def getBaseRegressoriDynTree(self):
        """get base regressor and identifiable basis matrix with iDynTree (SVD)"""

        with identificationHelpers.Timer() as t:
            # get subspace basis (for projection to base regressor/parameters)
            subspaceBasis = iDynTree.MatrixDynSize()
            if not self.generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
            # if not self.generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
                print "Error while computing basis matrix"

            self.B = subspaceBasis.toNumPy()

            print("tau: {}".format(self.tau.shape)),

            if self.robotranRegressor:
                self.YBase = self.regressor_stack_sym
            else:
                print("YStd: {}".format(self.YStd.shape)),
                # project regressor to base regressor, Y_base = Y_std*B
                self.YBase = np.dot(self.YStd, self.B)
            print("YBase: {}".format(self.YBase.shape))

            self.num_base_params = self.YBase.shape[1]
        print("Getting the base regressor (iDynTree) took %.03f sec." % t.interval)

    def getRandomRegressors(self, n_samples = 1000, fixed_base = True):
        import random

        R = np.array((self.N_OUT, self.N_PARAMS))
        for i in range(0, n_samples):
            # set random system state
            q = iDynTree.VectorDynSize.fromPyList((np.random.rand(self.N_DOFS)*np.pi).tolist())
            dq = iDynTree.VectorDynSize.fromPyList((np.random.rand(self.N_DOFS)*np.pi).tolist())
            ddq = iDynTree.VectorDynSize.fromPyList((np.random.rand(self.N_DOFS)*np.pi).tolist())

            if fixed_base:
                self.generator.setRobotState(q,dq,ddq, self.gravity_twist)
            else:
                base_acceleration = iDynTree.Twist()
                base_acceleration.zero()
                #base_acceleration = random values...
                self.generator.setRobotState(q,dq,ddq, self.gravity_twist, base_acceleration)  # fixed base

            #self.generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromPyList(torq))

            # get regressor
            regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
            knownTerms = iDynTree.VectorDynSize(self.N_OUT)
            if not self.generator.computeRegressor(regressor, knownTerms):
                print "Error during numeric computation of regressor"

            A = regressor.toNumPy()

            if i==0:
                R = A.T.dot(A)
            else:
                R += A.T.dot(A)

        return R

    def getBaseRegressorQR(self):
        """get base regressor and identifiable basis matrix with QR decomposition

        gets independent columns (non-unique choice) each with its dependent ones, i.e.
        those std parameter indices that form each of the base parameters (including the linear factors)
        """
        with identificationHelpers.Timer() as t:
            #using random regressor gives us structural base params, not dependent on excitation
            #QR of transposed gives us basis of column space of original matrix
            #TODO: this can be loaded from file if model structure doesn't change
            Yrand = self.getRandomRegressors()
            Q,R,P = sla.qr(Yrand.T, pivoting=True, mode='economic')

            #get rank
            r = np.where(np.abs(R.diagonal()) > self.min_tol)[0].size
            self.num_base_params = r

            #get basis projection matrix
            self.B = Q[:, 0:r]

            #get regressor for base parameters
            if self.robotranRegressor:
                self.YBase = self.regressor_stack_sym
            else:
                print("YStd: {}".format(self.YStd.shape)),
                # project regressor to base regressor, Y_base = Y_std*B
                self.YBase = np.dot(self.YStd, self.B)
            print("YBase: {}".format(self.YBase.shape))

            # seems we have to do QR again for column space dependencies
            Q,R,P = sla.qr(self.YStd, pivoting=True, mode='economic')
            self.Q, self.R, self.P = Q,R,P

            #create permuation matrix out of vector
            self.Pp = np.zeros((P.size, P.size))
            for i in P:
                self.Pp[i, P[i]] = 1

            # get the choice of indices of independent columns of the regressor matrix (std params -> base params)
            self.independent_cols = P[0:r]

            # get column dependency matrix (what dependent columns are combined in each base parameter with what factor)
            # j (independent column) = (value at i,j) * i (dependent column)
            ind = self.independent_cols.size
            R1 = R[0:ind,0:ind]
            R2 = R[:ind, ind:]
            self.linear_deps = la.inv(R1).dot(R2)

        print("Getting the base regressor (QR) took %.03f sec." % t.interval)

    def identifyBaseParameters(self, YBase=None, tau=None):
        """use previously computed regressors and identify base parameter vector."""

        if YBase is None:
            YBase = self.YBase
        if tau is None:
            tau = self.tau

        # TODO: get jacobian and contact force for each contact frame (when added to iDynTree)
        # in order to also use FT sensors in hands and feet
        # assuming zero external forces for fixed base on trunk
        # jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
        # self.generator.getFrameJacobian('arm', jacobian)

        # invert equation to get parameter vector from measurements and model + system state values
        #self.YBaseInv = la.pinv(self.YBase)
        #print("YBaseInv: {}".format(self.YBaseInv.shape))
        #self.xBase = np.dot(self.YBaseInv, self.tau.T) # - np.sum( YBaseInv*jacobian*contactForces )

        self.xBase = la.lstsq(YBase, tau)[0]
        #print "The base parameter vector {} is \n{}".format(self.xBase.shape, self.xBase)

        if self.useWLS:
            """add weighting with standard dev of estimation error on base regressor and params."""
            self.estimateTorques('base')
            # get standard deviation of measurement and modeling error \sigma_{rho}^2
            # for each joint subsystem (rho is assumed zero mean independent noise)
            self.sigma_rho = np.square(la.norm(self.tauMeasured-self.tauEstimated, axis=0))/ \
                                  (self.num_samples-self.num_base_params)

            # repeat stddev values for each measurement block (n_joints * num_samples)
            # along the diagonal of G
            G = np.diag(np.tile(self.sigma_rho, self.num_samples))

            # get standard deviation \sigma_{x} (of the estimated parameter vector x)
            #C_xx = la.norm(self.sigma_rho)*(la.inv(self.YBase.T.dot(self.YBase)))
            #sigma_x = np.sqrt(np.diag(C_xx))

            # weight Y and tau with deviations, identify params
            YBase = G.dot(self.YBase)
            tau = G.dot(self.tau)
            self.useWLS = 0
            self.identifyBaseParameters(YBase, tau)
            self.useWLS = 1

    def getBaseParamsFromParamError(self):
        if self.robotranRegressor:
            self.xBase = self.xBase + self.xStdModelAsBase   #both param vecs barycentric
        else:
            self.xBase = self.xBase + np.dot(self.B.T, self.xStdModel)   #both param vecs link relative linearized

    def getStdFromBase(self):
        # Note: assumes that xBase is still in error form if using a priori
        # i.e. don't call after getBaseParamsFromParamError

        # project back to standard parameters
        self.xStd = np.dot(self.B, self.xBase)

        # get estimated parameters from estimated error (add a priori knowledge)
        if self.useAPriori:
            self.xStd = self.xStd + self.xStdModel

        # print "The standard parameter vector {} is \n{}".format(self.xStd.shape, self.xStd)

    def estimateTorques(self, estimateWith=None):
        """ get torque estimations, prepare for plotting """

        with identificationHelpers.Timer() as t:
            if not estimateWith:
                #use global parameter choice if none is given specifically
                estimateWith = self.estimateWith
            if self.robotranRegressor:
                if estimateWith is 'base':
                    tauEst = np.dot(self.YBase, self.xBase)
                elif estimateWith is 'model':
                    tauEst = np.dot(self.YBase, self.xStdModelAsBase)
                elif estimateWith in ['std', 'std_direct']:
                    print("Error: I don't have a standard regressor from symbolic equations.")
                    sys.exit(-1)
                else:
                    print("unknown type of parameters: {}".format(self.estimateWith))
            else:
                # estimate torques again with regressor and parameters
                if estimateWith is 'model':
                    tauEst = np.dot(self.YStd, self.xStdModel) # idyntree standard regressor and parameters from URDF model
                elif estimateWith is 'base':
                    tauEst = np.dot(self.YBase, self.xBase)   # idyntree base regressor and identified base parameters
                elif estimateWith in ['std', 'std_direct']:
                    tauEst = np.dot(self.YStd, self.xStd)    # idyntree standard regressor and estimated standard parameters
                else:
                    print("unknown type of parameters: {}".format(self.estimateWith))

            # reshape torques into one column per DOF for plotting (NUM_SAMPLES*N_DOFSx1) -> (NUM_SAMPLESxN_DOFS)
            self.tauEstimated = np.reshape(tauEst, (self.num_samples, self.N_DOFS))

            self.sample_end = self.measurements['positions'].shape[0]
            if self.skip_samples > 0: self.sample_end -= (self.skip_samples+1)

            if self.simulate:
                if self.useAPriori:
                    tau = self.torques_stack    # use original measurements, not delta
                else:
                    tau = self.tau
                self.tauMeasured = np.reshape(tau, (self.num_samples, self.N_DOFS))
            else:
                self.tauMeasured = self.measurements['torques'][self.start_offset:self.sample_end:self.skip_samples+1, :]

        #print("torque estimation took %.03f sec." % t.interval)

    def getBaseEssentialParameters(self):
        """
        iteratively get essential parameters from previously identified base parameters.
        (goal is to get similar influence of all parameters, i.e. decrease sensitivity to errors,
        estimation with similar accuracy)

        based on Pham, 1991 and Gautier, 2013
        but with new stop criterium
        """

        # TODO: look at p_sigma_x ratios and why they get larger again
        with identificationHelpers.Timer() as t:
            not_essential_idx = list()
            #r_sigma = 21    #target ratio of parameters' relative std deviation
            ratio = 0

            self.xBase_orig = self.xBase.copy()
            while 1:
                # get new torque estimation to calc error norm (new estimation with updated parameters)
                self.estimateTorques('base')

                # get standard deviation of measurement and modeling error \sigma_{rho}^2
                sigma_rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))/(self.num_samples-self.num_base_params)

                # get standard deviation \sigma_{x} (of the estimated parameter vector x)
                C_xx = sigma_rho*(la.inv(np.dot(self.YBase.T, self.YBase)))
                sigma_x = np.diag(C_xx)

                # get relative standard deviation
                p_sigma_x = np.sqrt(sigma_x)
                for i in range(0, p_sigma_x.size):
                    if np.abs(self.xBase[i]) != 0:
                        p_sigma_x[i] /= np.abs(self.xBase[i])

                old_ratio = ratio
                ratio = np.max(p_sigma_x)/np.min(p_sigma_x)
                print "min-max ratio of relative stddevs: {}".format(ratio)
                # while loop condition moved to here
                #if ratio < r_sigma:
                #if ratio >= old_ratio and old_ratio != 0:
                if ratio == old_ratio and old_ratio != 0:
                    break

                #cancel the parameter with largest deviation
                param_idx = np.argmax(p_sigma_x)
                not_essential_idx.append(param_idx)
                self.xBase[param_idx] = 0

            self.xBase = self.xBase_orig
            #self.sigma_x = sigma_x
            #self.sigma_rho = sigma_rho
            self.baseEssentialIdx = [x for x in range(0,self.num_base_params) if x not in not_essential_idx]
            self.num_essential_params = len(self.baseEssentialIdx)
            print "Got {} essential parameters".format(self.num_essential_params)

        print("Getting base essential parameters took %.03f sec." % t.interval)

    def getStdEssentialParameters(self):
        """
        Find essential standard parameters from previously determined base essential parameters.
        """

        with identificationHelpers.Timer() as t:
            # get the choice of indices into the std params of the independent columns
            # of those, only select the std parameters that are essential
            self.stdEssentialIdx = self.independent_cols[self.baseEssentialIdx]

            # it seems we only want to identify the independent components among the base params,
            # values look better at least (paper is not clear about it)
            # intuitively, also the dependent ones should be essential as the linear combination is
            # used to identify and calc the error
            """
            # also get the ones that are linearly dependent on them -> base params
            dependents = []
            #to_delete = []
            for i in range(0,self.linear_deps.shape[0]):
                for j in range(0,self.linear_deps.shape[1]):
                    if np.abs(self.linear_deps[i,j]) > self.min_tol:
                        orgColi = self.P[self.independent_cols.size+i]
                        orgColj = self.P[j]
                        if orgColi not in dependents:
                             dependents.append(orgColi)
                        #orgColj has dependents, remove from stdEssentialIdx
                        #to_delete.append(orgColj)

                        #print(
                        #    '''col {} in W2(col {} in a) is a linear combination of col {} in W1 (col {} in a)'''\
                        #   .format(i, orgColi, j, orgColj))
            #self.stdEssentialIdx = np.concatenate((self.stdEssentialIdx, dependents))
            """
            # try to only identify those that are fully identifiable?
            #np.delete(self.stdEssentialIdx, to_delete, 0)
            self.stdNonEssentialIdx = [x for x in range(0, self.N_PARAMS) if x not in self.stdEssentialIdx]

            # get \hat{x_e}, set zeros for non-essential params
            self.xStdEssential = self.xStdModel.copy()
            self.xStdEssential[self.stdNonEssentialIdx] = 0

    def getNonsingularRegressor(self):
        with identificationHelpers.Timer() as t:
            U, s, VH = la.svd(self.YStd, full_matrices=False)
            V = VH.T
            nb = self.num_base_params
            st = self.N_PARAMS

            # non-singular YStd, called W_st in Gautier, 2013
            self.YStdHat = self.YStd - U[:, nb:st].dot(np.diag(s[nb:st])).dot(V[:,nb:st].T)
        print("Getting non-singular regressor took %.03f sec." % t.interval)

    def identifyStandardParameters(self):
        """Identify standard parameters directly with non-singular standard regressor."""
        with identificationHelpers.Timer() as t:
            self.YStdHatInv = la.pinv(self.YStdHat)
            x_tmp = np.dot(self.YStdHatInv, self.tau)

            if self.useAPriori:
                self.xStd = self.xStdModel + x_tmp
            else:
                self.xStd = x_tmp
        print("Identifying std parameters took %.03f sec." % t.interval)

    def identifyStandardEssentialParameters(self):
        """Identify standard essential parameters directly with non-singular standard regressor."""
        with identificationHelpers.Timer() as t:
            # weighting with previously determined essential params
            # calculates V_1e, U_1e etc. (Gautier, 2013)
            Y = self.YStd.dot(np.diag(self.xStdEssential))
            U_, s_, VH_ = la.svd(Y, full_matrices=False)
            ne = self.num_essential_params  #nr. of essential params among base params
            V_1 = VH_.T[:, 0:ne]
            U_1 = U_[:, 0:ne]
            s_1_inv = la.inv(np.diag(s_[0:ne]))
            x_tmp = np.diag(self.xStdEssential).dot(V_1).dot(s_1_inv).dot(U_1.T).dot(self.tau)

            if self.useAPriori:
                self.xStd = self.xStdModel + x_tmp
            else:
                self.xStd = x_tmp

        print("Identifying std essential parameters took %.03f sec." % t.interval)

    def output(self):
        """Do some pretty printing of parameters."""

        import colorama
        from colorama import Fore, Back, Style
        colorama.init(autoreset=True)

        if not self.useEssentialParams:
            self.stdEssentialIdx = range(0, self.N_PARAMS)
            self.stdNonEssentialIdx = []

        # convert params to COM-relative instead of frame origin-relative (linearized parameters)
        xStd = self.xStd
        if not self.robotranRegressor:
            xStd = self.helpers.paramsLink2Bary(self.xStd)
        xStdModel = self.helpers.paramsLink2Bary(self.xStdModel)

        # collect values for parameters
        description = self.generator.getDescriptionOfParameters()
        idx_p = 0
        lines = list()
        for d in description.replace(r'Parameter ', '# ').replace(r'first moment', 'center').split('\n'):
            new = xStd[idx_p]
            old = xStdModel[idx_p]
            diff = new - old
            #print beginning of each link block in green
            if idx_p % 10 == 0:
                d = Fore.GREEN + d
            lines.append((old, new, diff, d))
            idx_p+=1
            if idx_p == len(xStd):
                break

        column_widths = [15, 15, 7, 45]   # widths of the columns
        precisions = [8, 8, 4, 0]         # numerical precision

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
        idx_p = 0
        for l in lines:
            t = template.format(*l)
            if idx_p in self.stdNonEssentialIdx:
                t = Style.DIM + t
            if idx_p in self.stdEssentialIdx:
                t = Style.BRIGHT + t
            print t
            idx_p+=1
        print Style.RESET_ALL

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
        ymin = np.min([self.tauMeasured, self.tauEstimated]) - 5
        ymax = np.max([self.tauMeasured, self.tauEstimated]) + 5

        T = self.measurements['times'][self.start_offset:self.sample_end:self.skip_samples+1]
        for (data, title) in datasets:
            plt.figure()
            plt.ylim([ymin, ymax])
            plt.title(title)
            for i in range(0, self.N_DOFS):
                for d_i in range(0, len(data)):
                    l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                    plt.plot(T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
            leg = plt.legend(loc='best', fancybox=True, fontsize=10)
            leg.draggable()
        plt.show()
        self.measurements.close()

    def printMemUsage(self):
        import humanize
        print "Memory usage:"
        for v in self.__dict__.keys():
            if type(self.__dict__[v]).__module__ == np.__name__:
                print "{}: {} ".format( v, (humanize.naturalsize(self.__dict__[v].nbytes, binary=True)) ),
        print "\n"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--measurements', required=True, type=str, help='the file to load the measurements from')
    parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
    parser.add_argument('--explain', help='whether to explain parameters', action='store_true')
    parser.set_defaults(plot=False, explain=False)
    args = parser.parse_args()

    identification = Identification(args.model, args.measurements)
    identification.computeRegressors()

    if identification.useEssentialParams:
        identification.getBaseRegressorQR()
        identification.identifyBaseParameters()
        identification.getBaseEssentialParameters()
        identification.getStdEssentialParameters()
        identification.getNonsingularRegressor()
        identification.identifyStandardEssentialParameters()
        if identification.useAPriori:
            identification.getBaseParamsFromParamError()
    else:
        if identification.estimateWith in ['base', 'std']:
            identification.getBaseRegressoriDynTree()
            identification.identifyBaseParameters()
            identification.getStdFromBase()
            if identification.useAPriori:
                identification.getBaseParamsFromParamError()
        elif identification.estimateWith is 'std_direct':
            identification.getBaseRegressoriDynTree()
            identification.getNonsingularRegressor()
            identification.identifyStandardParameters()

    #identification.printMemUsage()

    if args.explain:
        identification.output()
    if args.plot:
        identification.estimateTorques()
        identification.plot()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            # open ipdb when an exception happens
            import ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)

