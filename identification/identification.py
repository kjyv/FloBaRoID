#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys
import math
import numpy as np
import numpy.linalg as la
#import numexpr as ne
import scipy.linalg as sla
import scipy.stats as stats

import matplotlib #; matplotlib.use('qt4agg')
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
# TODO: write params to file/urdf file, give explicit option for that
# TODO: add/use contact forces
# TODO: save random regressor to file next to urdf

class Identification(object):
    def __init__(self, urdf_file, measurements_files, regressor_file):
        ## options

        # determine number of samples to use
        # (Khalil recommends about 500 times number of parameters to identify...)
        self.start_offset = 0  #how many samples from the begginning of the measurements are skipped

        self.skip_samples = 4   #how many values to skip before using the next sample

        # use robotran symbolic regressor to estimate torques (else iDynTree)
        self.robotranRegressor = 0

        # simulate torques from target values, don't use both
        self.iDynSimulate = 0 # simulate torque using idyntree (instead of reading measurements)
        self.robotranSimulate = 0 # simulate torque using robotran (instead of reading measurements)
        self.addNoise = 0   #add some artificial zero-mean white noise to the 'measured' torques

        # using which parameters to estimate torques for validation. Set to one of
        # ['base', 'std', 'std_direct', 'urdf']
        self.estimateWith = 'std'

        # use known CAD parameters as a priori knowledge, generates (more) consistent std parameters
        self.useAPriori = 1

        # use weighted least squares(WLS) instead of ordinary least squares
        self.useWLS = 0

        # whether to identify and use direct standard with essential parameters
        self.useEssentialParams = 1

        # whether to take out masses from essential params to be identified because they are e.g.
        # well known or introduce problems
        self.dontIdentifyMasses = 0

        self.outputBarycentric = 0

        self.showMemUsage = 0
        self.show_random_regressor = 0

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

            self.measurements = {}
            for fn in measurements_files:
                m = np.load(fn[0])
                mv = {}
                for k in m.keys():
                    mv[k] = m[k]
                    if not self.measurements.has_key(k):
                        #first file
                        self.measurements[k] = m[k]
                    else:
                        if k == 'times':
                            mv[k] = m[k] - m[k][0] + (m[k][1]-m[k][0]) #let values start with first time diff
                            mv[k] = mv[k] + self.measurements[k][-1] #add after previous times
                        #following files, append data
                        self.measurements[k] = np.concatenate((self.measurements[k], mv[k]), axis=0)

                m.close()

            self.num_samples = (self.measurements['positions'].shape[0]-self.start_offset)/(self.skip_samples+1)
            print 'loaded {} measurement samples (using {})'.format(
                self.measurements['positions'].shape[0], self.num_samples)

            # create generator instance and load model
            self.generator = iDynTree.DynamicsRegressorGenerator()
            self.generator.loadRobotAndSensorsModelFromFile(self.URDF_FILE)

            # load also with new model class for some functions
            self.model = iDynTree.Model()
            iDynTree.modelFromURDF(self.URDF_FILE, self.model)
            print 'loaded model {}'.format(self.URDF_FILE)

            # define what regressor type to use and options for it
            regrXml = '''
            <regressor>
              <jointTorqueDynamics>
                <allJoints/>
              </jointTorqueDynamics>
            </regressor>'''

            if regressor_file:
                with open(regressor_file, 'r') as file:
                   regrXml = file.read()
            self.generator.loadRegressorStructureFromString(regrXml)

            # TODO: this and the following are not dependent on joints specified in regressor!
            self.N_DOFS = self.generator.getNrOfDegreesOfFreedom()
            print '# DOFs: {}'.format(self.N_DOFS)

            # Get the number of outputs of the regressor
            # (should be #links - #fakeLinks)
            self.N_OUT = self.generator.getNrOfOutputs()
            print '# outputs: {}'.format(self.N_OUT)

            # get initial inertia params (from urdf)
            self.N_PARAMS = self.generator.getNrOfParameters()
            print '# params: {}'.format(self.N_PARAMS)

            self.N_LINKS = self.generator.getNrOfLinks()
            print '# links: {} ({} fake)'.format(self.N_LINKS, self.generator.getNrOfFakeLinks())

            self.link_names = []
            for i in range(0, self.N_LINKS):
                self.link_names.append(self.model.getLinkName(i))
            print '({})'.format(self.link_names)

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
            if self.estimateWith is 'urdf':
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

                # get relative link positions from params
                # # (could also just get them from xStdModelBary...)
                d = np.zeros((4,10))  # should be 3 x 7, but invdynabar is funny and uses matlab indexing
                for i in range(1, self.N_DOFS+1):
                    j = self.model.getJoint(i-1)
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
            # TODO: this takes multiple seconds because of lazy loading, try preload
            # or use other data format
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
                    # 1 LShLat  |  -42deg    |  1
                    # rest      |  0deg      |  1
                    #pos[0]+=np.deg2rad(20)
                    pos[1]-=np.deg2rad(42)

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
                    torq += np.random.normal(0.0,0.1)
            simulate_time += t.interval

            #...still in sample loop

            if self.useAPriori and math.isnan(torqAP[0]) :
                #print "torques contain nans. Please investigate"
                #embed()
                # possibly just a very small number in C that gets converted to nan?
                torqAP[:] = 0

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
                    #self.generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromPyList(torq))

                    # get (standard) regressor
                    regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
                    knownTerms = iDynTree.VectorDynSize(self.N_OUT)    # what are known terms useable for?
                    if not self.generator.computeRegressor(regressor, knownTerms):
                        print "Error during numeric computation of regressor"

                    YStd = regressor.toNumPy()
                    # stack on previous regressors
                    np.copyto(self.regressor_stack[start:start+self.N_DOFS], YStd)
                num_time += t.interval

            # stack results onto matrices of previous timesteps
            np.copyto(self.torques_stack[start:start+self.N_DOFS], torq)
            if self.useAPriori:
                np.copyto(self.torquesAP_stack[start:start+self.N_DOFS], torqAP)

        with identificationHelpers.Timer() as t:
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

    def getBaseRegressorSVD(self):
        """get base regressor and identifiable basis matrix with iDynTree (SVD)"""

        with identificationHelpers.Timer() as t:
            # get subspace basis (for projection to base regressor/parameters)
            if False:
                subspaceBasis = iDynTree.MatrixDynSize()
                if not self.generator.computeFixedBaseIdentifiableSubspace(subspaceBasis):
                # if not self.generator.computeFloatingBaseIdentifiableSubspace(subspaceBasis):
                    print "Error while computing basis matrix"

                self.B = subspaceBasis.toNumPy()
            else:
                Yrand = self.getRandomRegressors(2000)
                #A = iDynTree.MatrixDynSize(self.N_PARAMS, self.N_PARAMS)
                #self.generator.generate_random_regressors(A, False, True, 2000)
                #Yrand = A.toNumPy()
                U, s, Vh = la.svd(Yrand, full_matrices=False)
                r = np.sum(s>self.min_tol)
                self.B = -Vh.T[:, 0:r]
                self.num_base_params = r

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

    def getRandomRegressors(self, fixed_base = True, n_samples=None):
        """
        Utility function for generating a random regressor for numerical base parameter calculation
        Given n_samples, the Y (n_samples*getNrOfOutputs() X getNrOfParameters() ) regressor is
        obtained by stacking the n_samples generated regressors This function returns Y^T Y
        (getNrOfParameters() X getNrOfParameters() ) (that share the row space with Y)
        """
        import random

        if not n_samples:
            n_samples = 2000 #self.N_DOFS * 1000
        R = np.array((self.N_OUT, self.N_PARAMS))
        regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
        knownTerms = iDynTree.VectorDynSize(self.N_OUT)
        for i in range(0, n_samples):
            # set random system state

            """
            # TODO: conceal to joint limits from urdf
            q_lim_pos = np.array([ 2.96705972839,  2.09439510239,  2.96705972839,  2.09439510239,
                                   2.96705972839,  2.09439510239,  2.96705972839])
            #q_lim_pos.fill(np.pi)
            q_lim_neg = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239,
                                  -2.96705972839, -2.09439510239, -2.96705972839])
            #q_lim_neg.fill(np.pi)
            dq_lim = np.array([1.91986217719, 1.91986217719, 2.23402144255, 2.23402144255,
                               3.56047167407, 3.21140582367, 3.21140582367])
            #dq_lim.fill(np.pi)

            q = iDynTree.VectorDynSize.fromPyList(((np.random.rand(self.N_DOFS)-0.5)*2*q_lim_pos).tolist())
            dq = iDynTree.VectorDynSize.fromPyList(((np.random.rand(self.N_DOFS)-0.5)*2*dq_lim).tolist())
            ddq = iDynTree.VectorDynSize.fromPyList(((np.random.rand(self.N_DOFS)-0.5)*2*np.pi).tolist())
            """

            q = iDynTree.VectorDynSize.fromPyList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())
            dq = iDynTree.VectorDynSize.fromPyList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())
            ddq = iDynTree.VectorDynSize.fromPyList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())

            # TODO: handle for fixed dofs (set vel and acc to zero)

            if fixed_base:
                self.generator.setRobotState(q,dq,ddq, self.gravity_twist)
            else:
                base_acceleration = iDynTree.Twist()
                base_acceleration.zero()
                #TODO: base_acceleration = random values...
                self.generator.setRobotState(q,dq,ddq, self.gravity_twist, base_acceleration)

            # get regressor
            if not self.generator.computeRegressor(regressor, knownTerms):
                print "Error during numeric computation of regressor"

            A = regressor.toNumPy()

            # add to previous regressors, linear dependency doesn't change
            # (if too many, saturation or accuracy problems?)
            if i==0:
                R = A.T.dot(A)
            else:
                R += A.T.dot(A)

        if self.show_random_regressor:
            plt.imshow(R, interpolation='nearest')
            plt.show()

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
            Yrand = self.getRandomRegressors(n_samples=4000)
            Qt,Rt,Pt = sla.qr(Yrand.T, pivoting=True, mode='economic')

            #get rank
            r = np.where(np.abs(Rt.diagonal()) > self.min_tol)[0].size
            self.num_base_params = r

            #get basis projection matrix
            self.B = Qt[:, 0:r]

            #get regressor for base parameters
            if self.robotranRegressor:
                self.YBase = self.regressor_stack_sym
            else:
                print("YStd: {}".format(self.YStd.shape)),
                # project regressor to base regressor, Y_base = Y_std*B
                self.YBase = np.dot(self.YStd, self.B)
            print("YBase: {}".format(self.YBase.shape))

            # seems we have to do QR again for column space dependencies
            Q,R,P = sla.qr(Yrand, pivoting=True, mode='economic')
            self.Q, self.R, self.P = Q,R,P

            #create permuation matrix out of vector
            self.Pp = np.zeros((P.size, P.size))
            for i in P:
                self.Pp[i, P[i]] = 1

            # get the choice of indices of independent columns of the regressor matrix (std params -> base params)
            self.independent_cols = P[0:r]

            # get column dependency matrix (with what factor are columns of each base parameter dependent)
            # i (independent column) = (value at i,j) * j (dependent column index among the others)
            R1 = R[0:r, 0:r]
            R2 = R[0:r, r:]
            self.linear_deps = la.inv(R1).dot(R2)

        print("Getting the base regressor (QR) took %.03f sec." % t.interval)

    def identifyBaseParameters(self, YBase=None, tau=None):
        """use previously computed regressors and identify base parameter vector using ordinary or weighted least squares."""

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

        self.xBaseModel = np.dot(self.B.T, self.xStdModel)

        if self.useWLS:
            """add weighting with standard dev of estimation error on base regressor and params."""
            self.estimateRegressorTorques('base')
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
            self.xBase = self.xBase + self.xBaseModel   #both param vecs link relative linearized

    def getStdFromBase(self):
        # Note: assumes that xBase is still in error form if using a priori
        # i.e. don't call after getBaseParamsFromParamError

        # project back to standard parameters
        self.xStd = np.dot(self.B, self.xBase)

        # get estimated parameters from estimated error (add a priori knowledge)
        if self.useAPriori:
            self.xStd = self.xStd + self.xStdModel

        # print "The standard parameter vector {} is \n{}".format(self.xStd.shape, self.xStd)

    def estimateRegressorTorques(self, estimateWith=None):
        """ get torque estimations using regressors, prepare for plotting """

        with identificationHelpers.Timer() as t:
            if not estimateWith:
                #use global parameter choice if none is given specifically
                estimateWith = self.estimateWith
            if self.robotranRegressor:
                if estimateWith is 'base':
                    tauEst = np.dot(self.YBase, self.xBase)
                elif estimateWith is 'urdf':
                    tauEst = np.dot(self.YBase, self.xStdModelAsBase)
                elif estimateWith in ['std', 'std_direct']:
                    print("Error: I don't have a standard regressor from symbolic equations.")
                    sys.exit(-1)
                else:
                    print("unknown type of parameters: {}".format(self.estimateWith))
            else:
                # estimate torques with idyntree regressor and different params
                if estimateWith is 'urdf':
                    tauEst = np.dot(self.YStd, self.xStdModel)
                elif estimateWith is 'base_essential':
                    tauEst = np.dot(self.YBase, self.xBase_essential)
                elif estimateWith is 'base':
                    tauEst = np.dot(self.YBase, self.xBase)
                elif estimateWith in ['std', 'std_direct']:
                    tauEst = np.dot(self.YStd, self.xStd)
                else:
                    print("unknown type of parameters: {}".format(self.estimateWith))

            # reshape torques into one column per DOF for plotting (NUM_SAMPLES*N_DOFSx1) -> (NUM_SAMPLESxN_DOFS)
            self.tauEstimated = np.reshape(tauEst, (self.num_samples, self.N_DOFS))

            self.sample_end = self.measurements['positions'].shape[0]
            if self.skip_samples > 0: self.sample_end -= (self.skip_samples)

            if self.simulate:
                if self.useAPriori:
                    tau = self.torques_stack    # use original measurements, not delta
                else:
                    tau = self.tau
                self.tauMeasured = np.reshape(tau, (self.num_samples, self.N_DOFS))
            else:
                self.tauMeasured = self.measurements['torques'][self.start_offset:self.sample_end:self.skip_samples+1, :]

            self.T = self.measurements['times'][self.start_offset:self.sample_end:self.skip_samples+1]

        #print("torque estimation took %.03f sec." % t.interval)

    def estimateValidationTorques(self, file):
        """ calculate torques of trajectory from validation measurements and identified params """
        # TODO: get identified params directly into idyntree (new KinDynComputations class does not have
        # inverse dynamics yet, so has to go over new urdf file now)

        v_data = np.load(file)
        dynComp = iDynTree.DynamicsComputations();

        self.helpers.replaceParamsInURDF(self.URDF_FILE, self.xStd, self.link_names)
        dynComp.loadRobotModelFromFile(self.URDF_FILE + '.tmp')
        gravity = iDynTree.SpatialAcc();
        gravity.zero()
        gravity.setVal(2, -9.81);

        self.tauEstimated = None
        for m_idx in range(0, v_data['positions'].shape[0], self.skip_samples+1):
            # read measurements
            pos = v_data['positions'][m_idx]
            vel = v_data['velocities'][m_idx]
            acc = v_data['accelerations'][m_idx]
            torq = v_data['torques'][m_idx]

            # system state for iDynTree
            q = iDynTree.VectorDynSize.fromPyList(pos)
            dq = iDynTree.VectorDynSize.fromPyList(vel)
            ddq = iDynTree.VectorDynSize.fromPyList(acc)

            # calc torques with iDynTree dynamicsComputation class
            dynComp.setRobotState(q, dq, ddq, gravity)

            torques = iDynTree.VectorDynSize(self.N_DOFS)
            baseReactionForce = iDynTree.Wrench()   # assume zero for fixed base, otherwise use e.g. imu data

            # compute inverse dynamics with idyntree (simulate)
            dynComp.inverseDynamics(torques, baseReactionForce)
            if self.tauEstimated is None:
                self.tauEstimated = torques.toNumPy()
            else:
                self.tauEstimated = np.vstack((self.tauEstimated, torques.toNumPy()))

        if self.skip_samples > 0:
            self.tauMeasured = v_data['torques'][::self.skip_samples]
            self.T = v_data['times'][::self.skip_samples]
        else:
            self.tauMeasured = v_data['torques']
            self.T = v_data['times']

    def getBaseEssentialParameters(self):
        """
        iteratively get essential parameters from previously identified base parameters.
        (goal is to get similar influence of all parameters, i.e. decrease condition number by throwing
        out parameters that are too sensitive to errors. The remaining params should be estimated with
        similar accuracy)

        based on Pham, 1991 and Gautier, 2013 and Jubien, 2014
        """

        with identificationHelpers.Timer() as t:
            # if columns are deleted or just cancelled by setting to zero
            delete_columns = 0

            # keep current values
            xBase_orig = self.xBase.copy()
            YBase_orig = self.YBase.copy()

            # add a priori info, xBase includes only parameter diffs
            self.xBase += self.xBaseModel

            # count how many params were canceled
            b_c = 0

            # list of param indices to keep the original indices when deleting columns
            base_idx = range(0,self.num_base_params)
            not_essential_idx = list()
            ratio = 0

            # get initial errors of estimation
            self.estimateRegressorTorques('base')
            error = np.mean(self.tauMeasured-self.tauEstimated, axis=1)
            k2, p = stats.normaltest(error)

            use_f_test = p > 0.05  #5%
            if use_f_test:
                print("error is normal distributed")
                pure_error = np.sum(np.square( (self.tauMeasured.T - np.mean(self.tauMeasured, axis=1)).T ))
                print "pure_error: {}".format(pure_error / (self.num_samples*self.N_DOFS - self.num_samples))
                error_norm_start = np.square(la.norm(error))
            else:
                print("error is not normal distributed (p={}), can't use f-test".format(p))
                F = 0

            rho_start = np.square(la.norm(self.tauMeasured-self.tauEstimated))

            # start removing non-essential parameters
            while 1:
                # get new torque estimation to calc error norm (new estimation with updated parameters)
                self.estimateRegressorTorques('base')

                # get standard deviation of measurement and modeling error \sigma_{rho}^2
                rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))
                sigma_rho = rho/(self.num_samples-self.num_base_params)

                # get standard deviation \sigma_{x} (of the estimated parameter vector x)
                C_xx = sigma_rho*(la.inv(np.dot(self.YBase.T, self.YBase)))

                # TODO: since also side diagonals carry information on how this param influences other
                # params, try using norm of columns instead of diagonal elements
                #sigma_x = np.linalg.norm(C_xx, axis=1)
                sigma_x = np.diag(C_xx)

                # get relative standard deviation
                p_sigma_x = np.sqrt(sigma_x)
                for i in range(0, p_sigma_x.size):
                    if np.abs(self.xBase[i]) != 0:
                        p_sigma_x[i] /= np.abs(self.xBase[i])

                old_ratio = ratio
                ratio = np.max(p_sigma_x)/np.min(p_sigma_x)

                if use_f_test:
                    # use f-test to determine if model reduction can be accepted or not
                    lack_of_fit = self.N_DOFS*np.sum( np.square( (np.mean(self.tauMeasured, axis=1) - self.tauEstimated.T).T) )
                    print "lack_of_fit: {}".format(lack_of_fit / (self.num_samples - (self.num_base_params-b_c)))

                    #lack-of-fit
                    #F = ( lack_of_fit / (self.num_samples - (self.num_base_params-b_c))) /  \
                    #    ( pure_error / (self.num_samples*self.N_DOFS - self.num_samples))

                    #f-test from janot
                    error_norm = np.square(la.norm(np.mean(self.tauMeasured-self.tauEstimated, axis=1)))
                    F = ((error_norm - error_norm_start) / (self.num_base_params - b_c)) /  \
                        (error_norm_start / (self.num_samples-self.num_base_params))

                print "min-max ratio of relative stddevs: {}, F: {}".format(ratio, F)

                # while loop condition moved to here
                #if ratio == old_ratio and old_ratio != 0:
                #if ratio == old_ratio and old_ratio != 0 or ratio < 20:
                if use_f_test and F > stats.f.ppf(0.95, self.num_base_params, self.num_base_params-b_c):    #alpha = 5%
                    break
                if not use_f_test and ratio < 25:
                    break

                #cancel the parameter with largest deviation
                param_idx = np.argmax(p_sigma_x)
                #get its index among the base params (otherwise it doesnt take deletion into account)
                param_base_idx = base_idx[param_idx]
                if param_base_idx not in not_essential_idx:
                    not_essential_idx.append(param_base_idx)
                else:
                    # TODO: if parameter was set to zero and still has the largest std deviation,
                    # something is weird..?
                    print("param {} already canceled before, stopping".format(param_base_idx))
                    break

                if delete_columns:
                    self.xBase = np.delete(self.xBase, param_idx, 0)
                    base_idx = np.delete(base_idx, param_idx, 0)
                    self.YBase = np.delete(self.YBase, param_idx, 1)
                else:
                    self.xBase[param_idx] = 0
                b_c += 1

            # get indices of the essential base params
            self.baseNonEssentialIdx = not_essential_idx
            self.baseEssentialIdx = [x for x in range(0,self.num_base_params) if x not in not_essential_idx]
            self.num_essential_params = len(self.baseEssentialIdx)

            # leave previous base params and regressor unchanged
            if delete_columns:
                self.xBase_essential = np.zeros_like(xBase_orig)
                self.xBase_essential[self.baseEssentialIdx] = self.xBase.copy()
            else:
                self.xBase_essential = self.xBase.copy()
            self.xBase = xBase_orig
            self.YBase = YBase_orig

            print "Got {} essential parameters".format(self.num_essential_params)

        print("Getting base essential parameters took %.03f sec." % t.interval)

    def getStdEssentialParameters(self):
        """
        Find essential standard parameters from previously determined base essential parameters.
        """

        with identificationHelpers.Timer() as t:
            # get the choice of indices into the std params of the independent columns.
            # Of those, only select the std parameters that are essential
            self.stdEssentialIdx = self.independent_cols[self.baseEssentialIdx]

            # intuitively, also the dependent columns should be essential as the linear combination
            # is used to identify and calc the error
            useDependents = 0
            useCADWeighting = 1
            if useDependents:
                # also get the ones that are linearly dependent on them -> base params
                dependents = []
                #to_delete = []
                for i in range(0, self.linear_deps.shape[0]):
                    for j in range(0,self.linear_deps.shape[1]):
                        if np.abs(self.linear_deps[i,j]) > 0.1:
                            dep_org_col = self.P[self.independent_cols.size+j]
                            indep_org_col = self.P[i]
                            if dep_org_col not in dependents and indep_org_col in self.stdEssentialIdx:
                                dependents.append(dep_org_col)
                            #indep_org_col has dependents, remove from stdEssentialIdx to get fully identifiable
                            #to_delete.append(indep_org_col)

                            #print(
                            #    ('''col {} in W2(col {} in a) is a linear combination of col {} in W1''' +\
                            #    '''(col {} in a) with factor {}''')\
                            #   .format(i, dep_org_col, j, indep_org_col, self.linear_deps[i,j]))
                #print self.stdEssentialIdx
                #print len(dependents)
                print dependents
                self.stdEssentialIdx = np.concatenate((self.stdEssentialIdx, dependents))

            #np.delete(self.stdEssentialIdx, to_delete, 0)

            #remove mass params if present
            if self.dontIdentifyMasses:
                ps = range(0,self.N_PARAMS, 10)
                self.stdEssentialIdx = np.fromiter((x for x in self.stdEssentialIdx if x not in ps), int)

            self.stdNonEssentialIdx = [x for x in range(0, self.N_PARAMS) if x not in self.stdEssentialIdx]

            ## get \hat{x_e}, set zeros for non-essential params
            if useDependents or useCADWeighting:
                # we don't really know what the weights are if we have more std essential than base
                # essentials, so use CAD/previous params for weighting
                self.xStdEssential = self.xStdModel.copy()

                # set essential but zero cad values to small values that are in possible range of those parameters
                # so something can be estimated
                #self.xStdEssential[np.where(self.xStdEssential == 0)[0]] = .1
                idx = 0
                for p in self.xStdEssential:
                    if p == 0:
                        v = 0.1
                        p_start = idx/10*10
                        if idx % 10 in [1,2,3]:   #com value
                            v = np.mean(self.xStdModel[p_start + 1:p_start + 4]) * 0.1
                        elif idx % 10 in [4,5,6,7,8,9]:  #inertia value
                            inertia_range = np.array([4,5,6,7,8,9])+p_start
                            v = np.mean(self.xStdModel[np.where(self.xStdModel[inertia_range] != 0)[0]+p_start+4]) * 0.1
                        if v == 0: v = 0.1
                        self.xStdEssential[idx] = v
                        #print idx, idx % 10, v
                    idx += 1

                # cancel non-essential std params so they are not identified
                self.xStdEssential[self.stdNonEssentialIdx] = 0
            else:
                # weighting using base essential params (like in Gautier, 2013)
                # (paper is not specifying if using absolute base params or identified errors)
                self.xStdEssential = np.zeros_like(self.xStdModel)
                self.xStdEssential[self.stdEssentialIdx] = \
                        self.xBase_essential[self.baseEssentialIdx] \
                        + self.xBaseModel[self.baseEssentialIdx]

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

        print("Identifying %s std essential parameters took %.03f sec." % (len(self.stdEssentialIdx), t.interval))

    def output(self):
        """Do some pretty printing of parameters."""

        import colorama
        from colorama import Fore, Back, Style
        colorama.init(autoreset=True)

        if not self.useEssentialParams:
            self.stdEssentialIdx = range(0, self.N_PARAMS)
            self.stdNonEssentialIdx = []

        # convert params to COM-relative instead of frame origin-relative (linearized parameters)
        if self.outputBarycentric:
            if not self.robotranRegressor:
              xStd = self.helpers.paramsLink2Bary(self.xStd)
            xStdModel = self.helpers.paramsLink2Bary(self.xStdModel)
            print("Barycentric (relative to COM) Standard Parameters")
        else:
            xStd = self.xStd
            xStdModel = self.xStdModel
            print("Linear (relative to Frame) Standard Parameters")

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

        ## print base params
        if self.estimateWith in ['urdf', 'std_direct']:
            return

        print("Base Parameters and Corresponding standard columns")
        if not self.useEssentialParams:
            baseEssentialIdx = range(0, self.N_PARAMS)
            baseNonEssentialIdx = []
            xBase_essential = self.xBase
        else:
            baseEssentialIdx = self.baseEssentialIdx
            baseNonEssentialIdx = self.baseNonEssentialIdx
            xBase_essential = self.xBase_essential

        # collect values for parameters
        lines = list()
        for idx_p in range(0,self.num_base_params):
            #if xBase_essential[idx_p] != 0:
            #    new = xBase_essential[idx_p]
            #else:
            new = self.xBase[idx_p]
            old = self.xBaseModel[idx_p]
            diff = new - old

            deps = np.where(np.abs(self.linear_deps[idx_p, :])>0.1)[0]
            dep_factors = self.linear_deps[idx_p, deps]

            param_columns = ' |{}|'.format(self.independent_cols[idx_p])
            if len(deps):
                param_columns += " deps:"
            for p in range(0, len(deps)):
                param_columns += ' {:.4f}*|{}|'.format(dep_factors[p], self.P[self.num_base_params:][deps[p]])

            lines.append((old, new, diff, param_columns))

        column_widths = [15, 15, 7, 30]   # widths of the columns
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
            if idx_p in baseNonEssentialIdx:
                t = Style.DIM + t
            if idx_p in baseEssentialIdx:
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
            ([self.tauMeasured - self.tauEstimated], 'Estimation Error'),
            #([self.measurements['positions'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Positions'),
            #([self.measurements['velocities'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Vels'),
            #([self.measurements['accelerations'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Accls'),
        ]

        # scale all figures to same ranges and add some margin
        ymin = np.min([self.tauMeasured, self.tauEstimated])
        ymin += ymin * 0.05
        ymax = np.max([self.tauMeasured, self.tauEstimated])
        ymax += ymax * 0.05

        for (data, title) in datasets:
            plt.figure()
            plt.ylim([ymin, ymax])
            plt.title(title)
            for i in range(0, self.N_DOFS):
                for d_i in range(0, len(data)):
                    l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                    plt.plot(self.T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
            leg = plt.legend(loc='best', fancybox=True, fontsize=10)
            leg.draggable()
        plt.show()
        #self.measurements.close()

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
    parser.add_argument('--measurements', required=True, nargs='*', action='append', type=str,
                        help='the file(s) to load the measurements from')
    parser.add_argument('--verification', required=False, type=str,
                        help='the file to load the verification trajectory from')
    parser.add_argument('--regressor', required=False, type=str,
                        help='the file containing the regressor structure(for the iDynTree generator).\
                              Identifies on all joints if not specified.')
    parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
    parser.add_argument('--explain', help='whether to explain parameters', action='store_true')
    parser.set_defaults(plot=False, explain=False, regressor=None)
    args = parser.parse_args()

    identification = Identification(args.model, args.measurements, args.regressor)
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
        identification.getBaseRegressorQR()
        if identification.estimateWith in ['base', 'std']:
            identification.identifyBaseParameters()
            identification.getStdFromBase()
            if identification.useAPriori:
                identification.getBaseParamsFromParamError()
        elif identification.estimateWith is 'std_direct':
            identification.getNonsingularRegressor()
            identification.identifyStandardParameters()

    if identification.showMemUsage:
        identification.printMemUsage()

    if args.explain:
        identification.output()
    if args.plot:
        if args.verification:
            identification.estimateValidationTorques(args.verification)
        else:
            identification.estimateRegressorTorques()
        identification.plot()

if __name__ == '__main__':
   # import ipdb
   # import traceback
    #try:
    main()
    '''
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            # open ipdb when an exception happens
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
    '''
