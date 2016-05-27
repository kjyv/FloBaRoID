#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys
import math
import numpy as np
import numpy.linalg as la
#import numexpr as ne
import scipy as sp
from scipy import signal
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
# Jubien, 2014: Dynamic identification of the Kuka LWR robot using motor torques and joint torque
# sensors data

# TODO: add/use contact forces
# TODO: load full model and programatically cut off chain from certain joints/links to allow
# subtree identification

class Identification(object):
    def __init__(self, urdf_file, urdf_file_real, measurements_files, regressor_file):
        ## options

        # determine number of samples to use
        # (Khalil recommends about 500 times number of parameters to identify...)
        self.startOffset = 100  #how many samples from the beginning of the (first) measurement are skipped
        self.skipSamples = 4    #how many values to skip before using the next sample

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

        # whether only "good" data is being selected or simply all is used
        self.selectBlocksFromMeasurements = 0
        self.block_size = 250  # needs to be at least as much as parameters so regressor is square or higher

        # whether to filter the regressor columns
        # (cutoff frequency is system dependent)
        # doesn't make any difference though it seems
        self.filterRegressor = 0

        # use weighted least squares(WLS) instead of ordinary least squares
        # needs small condition number, otherwise might amplify some parameters too much
        self.useWLS = 0

        # whether to identify and use direct standard with essential parameters
        self.useEssentialParams = 0

        # whether to take out masses from essential params to be identified because they are e.g.
        # well known or introduce problems
        self.dontIdentifyMasses = 0

        # some investigation output
        self.outputBarycentric = 0     #output all values in barycentric (e.g. urdf) form
        self.showMemUsage = 0          #print used memory for different variables
        self.showTiming = 0            #show times various steps have taken
        self.showRandomRegressor = 0   #show 2d plot of random regressor
        self.showErrorHistogram = 0    #show estimation error distribution
        self.showEssentialSteps = 0    #stop after every reduction step and show values

        # output options
        self.showStandardParams = 1
        self.showBaseParams = 0

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
            self.urdf_file_real = urdf_file_real

            # load data from multiple files and concatenate, fix timing
            self.measurements = {}
            for fa in measurements_files:
                for fn in fa:
                    m = np.load(fn)
                    mv = {}
                    for k in m.keys():
                        mv[k] = m[k]
                        if not self.measurements.has_key(k):
                            #first file
                            if k == 'times':
                                self.measurements[k] = m[k][self.startOffset:]
                            else:
                                self.measurements[k] = m[k][self.startOffset:, :]
                        else:
                            #following files, append data
                            if k == 'times':
                                mv[k] = m[k] - m[k][0] + (m[k][1]-m[k][0]) #let values start with first time diff
                                mv[k] = mv[k] + self.measurements[k][-1] #add after previous times
                                self.measurements[k] = np.concatenate((self.measurements[k], mv[k][self.startOffset:]),
                                                                      axis=0)
                            else:
                                self.measurements[k] = np.concatenate((self.measurements[k], mv[k][self.startOffset:, :]),
                                                                      axis=0)
                    m.close()

            self.num_loaded_samples = self.measurements['positions'].shape[0]
            self.num_used_samples = self.num_loaded_samples/(self.skipSamples+1)
            print 'loaded {} measurement samples (using {})'.format(
                self.num_loaded_samples, self.num_used_samples)

            # create data that identification is working on (subset of all measurements)
            self.samples = {}
            self.block_pos = 0
            if self.selectBlocksFromMeasurements:
                # fill with starting block
                for k in self.measurements.keys():
                    if k == 'times':
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.block_size]
                    else:
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.block_size, :]

                self.old_condition_number = 1e60
                self.last_largest_value = 1e60

                self.num_selected_samples = self.samples['positions'].shape[0]
                self.num_used_samples = self.num_selected_samples/(self.skipSamples+1)
            else:
                # simply use all data
                self.samples = self.measurements

            self.usedBlocks = list()
            self.unusedBlocks = list()

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

            self.N_LINKS = self.generator.getNrOfLinks()-self.generator.getNrOfFakeLinks()
            print '# links: {} ({} fake)'.format(self.N_LINKS+self.generator.getNrOfFakeLinks(),
                                                 self.generator.getNrOfFakeLinks())

            self.link_names = []
            for i in range(0, self.N_LINKS):
                self.link_names.append(self.model.getLinkName(i))
            print '({})'.format(self.link_names)

            self.jointNames = [self.generator.getDescriptionOfDegreeOfFreedom(dof) for dof in range(0, self.N_DOFS)]
            self.helpers = identificationHelpers.IdentificationHelpers(self.N_PARAMS)

        if self.showTiming:
            print("Initialization took %.03f sec." % t.interval)

    def hasMoreSamples(self):
        """ tell if there are more samples to be added to the date used for identification """

        if not self.selectBlocksFromMeasurements:
            return False

        if self.block_pos + self.block_size >= self.num_loaded_samples:
            return False

        return True

    def updateNumSamples(self):
        self.num_selected_samples = self.samples['positions'].shape[0]
        self.num_used_samples = self.num_selected_samples/(self.skipSamples+1)

    def removeLastSampleBlock(self):
        print "removing block starting at {}".format(self.block_pos)
        for k in self.measurements.keys():
            self.samples[k] = np.delete(self.samples[k], range(self.num_selected_samples - self.block_size,
                                                               self.num_selected_samples), axis=0)
        self.updateNumSamples()
        print "we now have {} samples selected (using {})".format(self.num_selected_samples, self.num_used_samples)

    def nextSampleBlock(self):
        """ fill samples with next measurements block """

        # advance to next block or end of data
        self.block_pos += self.block_size

        if self.block_pos + self.block_size > self.num_loaded_samples:
            self.block_size = self.num_loaded_samples - self.block_pos

        print "adding next block: {}/{}".format(self.block_pos, self.num_loaded_samples)

        for k in self.measurements.keys():
            if k == 'times':
                mv = self.measurements[k][self.block_pos:self.block_pos + self.block_size]
                #fix time offsets
                mv = mv - mv[0] + (mv[1]-mv[0]) #let values start with first time diff
                mv = mv + self.samples[k][-1] #add after previous times
                self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
            else:
                mv = self.measurements[k][self.block_pos:self.block_pos + self.block_size,:]
                self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
        self.updateNumSamples()

    def checkBlockImprovement(self):
        """ check if we want to add a new data block to the already selected ones """
        # possible criteria for minimization:
        # * condition number of (base) regressor (+variations)
        # * largest per link condition number gets smaller (some are really huge though and are not
        # getting smaller with most new data)
        # * estimation error gets smaller (same data or validation)
        # ratio of min/max rel std devs

        # use condition number of regressor
        #new_condition_number = la.cond(self.YBase.dot(np.diag(self.xBaseModel)))   #weighted with a priori
        new_condition_number = la.cond(self.YBase)

        # get condition number for each of the links
        linkConds = list()
        for i in range(0, self.N_LINKS):
            base_columns = [j for j in range(0, self.num_base_params) if self.independent_cols[j] in range(i*10, i*10+9)]

            for j in range(0, self.num_base_params):
                for dep in np.where(np.abs(self.linear_deps[j, :])>0.1)[0]:
                    if dep in range(i*10, i*10+9):
                        base_columns.append(j)
            if not len(base_columns):
                linkConds.append(99999)
            else:
                linkConds.append(la.cond(self.YBase[:, base_columns]))

            #linkConds[i] = la.cond(self.YStd[:, i*10:i*10+9])
        print("Condition numbers of link sub-regressor: [{}]\n".format(linkConds))

        """
        # use largest link condition number
        largest_idx = np.argmax(linkConds)  #2+np.argmax(linkConds[2:7])
        new_condition_number = linkConds[largest_idx]
        """

        # use validation error
        # new_condition_number = self.val_error

        # use std dev ratio
        """
        # get standard deviation of measurement and modeling error \sigma_{rho}^2
        rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))
        sigma_rho = rho/(self.num_used_samples-self.num_base_params)

        # get standard deviation \sigma_{x} (of the estimated parameter vector x)
        C_xx = sigma_rho*(la.inv(np.dot(self.YBase.T, self.YBase)))
        sigma_x = np.diag(C_xx)

        # get relative standard deviation
        p_sigma_x = np.sqrt(sigma_x)
        for i in range(0, p_sigma_x.size):
            if np.abs(self.xBase[i]) != 0:
                p_sigma_x[i] /= np.abs(self.xBase[i])

        new_condition_number = np.max(p_sigma_x)/np.min(p_sigma_x)
        """

        if new_condition_number > self.old_condition_number:
            print "not using block starting at {} (cond {})".format(self.block_pos, new_condition_number)
            self.removeLastSampleBlock()
            self.unusedBlocks.append(self.block_pos)
            is_better = False
        else:
            print "using block starting at {} (cond {})".format(self.block_pos, new_condition_number)
            self.usedBlocks.append(self.block_pos)
            self.old_condition_number = new_condition_number
            is_better = True

        return is_better


    def initRegressors(self):
        with identificationHelpers.Timer() as t:
            self.gravity_twist = iDynTree.Twist()
            self.gravity_twist.zero()
            self.gravity_twist.setVal(2, -9.81)

            if self.simulate or self.useAPriori:
                self.dynComp = iDynTree.DynamicsComputations();
                self.dynComp.loadRobotModelFromFile(self.URDF_FILE);
                self.gravity = iDynTree.SpatialAcc();
                self.gravity.zero()
                self.gravity.setVal(2, -9.81);

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

            # get model dependent projection matrix and linear column dependencies (i.e. base
            # groupings)
            self.getBaseRegressorQR()
            self.res_error = 100

        if self.showTiming:
            print("Init for computing regressors took %.03f sec." % t.interval)

    def computeRegressors(self):
        """compute regressors for each time step of the measurement data, stack them"""

        sym_time = 0
        num_time = 0
        simulate_time = 0

        self.regressor_stack = np.zeros(shape=(self.N_DOFS*self.num_used_samples, self.N_PARAMS))
        if self.robotranRegressor:
            self.regressor_stack_sym = np.zeros(shape=(self.N_DOFS*self.num_used_samples, 45))
        self.torques_stack = np.zeros(shape=(self.N_DOFS*self.num_used_samples))
        self.torquesAP_stack = np.zeros(shape=(self.N_DOFS*self.num_used_samples))

        """loop over measurements records (skip some values from the start)
           and get regressors for each system state"""
        for row in range(0, self.num_used_samples):
            with identificationHelpers.Timer() as t:
                m_idx = row*(self.skipSamples)+row
                if self.simulate:
                    pos = self.samples['target_positions'][m_idx]
                    vel = self.samples['target_velocities'][m_idx]
                    acc = self.samples['target_accelerations'][m_idx]
                else:
                    # read samples
                    pos = self.samples['positions'][m_idx]
                    vel = self.samples['velocities'][m_idx]
                    acc = self.samples['accelerations'][m_idx]
                    torq = self.samples['torques'][m_idx]

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
                    self.dynComp.setRobotState(q, dq, ddq, self.gravity)

                    torques = iDynTree.VectorDynSize(self.N_DOFS)
                    baseReactionForce = iDynTree.Wrench()   # assume zero for fixed base, otherwise use e.g. imu data

                    # compute inverse dynamics with idyntree (simulate)
                    self.dynComp.inverseDynamics(torques, baseReactionForce)
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
                    torq += np.random.randn(self.N_DOFS)*0.03
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

        if self.showTiming:
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
                Yrand = self.getRandomRegressors(5000)
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
            print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))

            self.num_base_params = self.YBase.shape[1]
        if self.showTiming:
            print("Getting the base regressor (iDynTree) took %.03f sec." % t.interval)

    def getRandomRegressors(self, fixed_base = True, n_samples=None):
        """
        Utility function for generating a random regressor for numerical base parameter calculation
        Given n_samples, the Y (n_samples*getNrOfOutputs() X getNrOfParameters() ) regressor is
        obtained by stacking the n_samples generated regressors This function returns Y^T Y
        (getNrOfParameters() X getNrOfParameters() ) (that share the row space with Y)
        """

        regr_filename = self.URDF_FILE + '.regressor.npz'
        generate_new = False
        try:
            regr_file = np.load(regr_filename)
            R = regr_file['R']
            n = regr_file['n']
            print("loaded random regressor from {}".format(regr_filename))
            if n != n_samples:
                generate_new = True
            #TODO: save and check timestamp of urdf file, if newer regenerate
        except IOError, KeyError:
            generate_new = True

        if generate_new:
            print("generating random regressor")
            import random

            if not n_samples:
                n_samples = self.N_DOFS * 1000
            R = np.array((self.N_OUT, self.N_PARAMS))
            regressor = iDynTree.MatrixDynSize(self.N_OUT, self.N_PARAMS)
            knownTerms = iDynTree.VectorDynSize(self.N_OUT)
            for i in range(0, n_samples):
                # set random system state

                # TODO: restrict to joint limits from urdf (these are kuka lwr4)
                """
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

            np.savez(regr_filename, R=R, n=n_samples)

        if self.showRandomRegressor:
            plt.imshow(R, interpolation='nearest')
            plt.show()

        return R

    def getBaseRegressorQR(self):
        """get base regressor and identifiable basis matrix with QR decomposition

        gets independent columns (non-unique choice) each with its dependent ones, i.e.
        those std parameter indices that form each of the base parameters (including the linear factors)
        """
        #using random regressor gives us structural base params, not dependent on excitation
        #QR of transposed gives us basis of column space of original matrix
        Yrand = self.getRandomRegressors(n_samples=5000)
        Qt,Rt,Pt = sla.qr(Yrand.T, pivoting=True, mode='economic')

        #get rank
        r = np.where(np.abs(Rt.diagonal()) > self.min_tol)[0].size
        self.num_base_params = r

        #get basis projection matrix
        self.B = Qt[:, 0:r]

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

    def getBaseRegressor(self):
        with identificationHelpers.Timer() as t:
            #get regressor for base parameters
            if self.robotranRegressor:
                self.YBase = self.regressor_stack_sym
            else:
                print("YStd: {}".format(self.YStd.shape)),
                # project regressor to base regressor, Y_base = Y_std*B
                self.YBase = np.dot(self.YStd, self.B)
            print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))

            if self.filterRegressor:
                order = 6  #Filter order
                fs = 200.0   #Sampling freq #TODO: get from measurements
                fc = fs/2 - 10 #90.0   #Cut-off frequency (Hz)
                b, a = sp.signal.butter(order, fc / (fs/2), btype='low', analog=False)
                for j in range(0, self.num_base_params):
                    for i in range(0, self.N_DOFS):
                        self.YBase[i::self.N_DOFS, j] = sp.signal.filtfilt(b, a, self.YBase[i::self.N_DOFS, j])

                """
                # Plot the frequency and phase response of the filter
                w, h = sp.signal.freqz(b, a, worN=8000)
                plt.subplot(2, 1, 1)
                plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
                plt.plot(fc, 0.5*np.sqrt(2), 'ko')
                plt.axvline(fc, color='k')
                plt.xlim(0, 0.5*fs)
                plt.title("Lowpass Filter Frequency Response")
                plt.xlabel('Frequency [Hz]')

                plt.subplot(2,1,2)
                h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
                plt.plot(w, h_Phase)
                plt.ylabel('Phase (radians)')
                plt.xlabel(r'Frequency (Hz)')
                plt.title(r'Phase response')
                plt.subplots_adjust(hspace=0.5)
                plt.grid()
                """

        if self.showTiming:
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
            """
            additionally do weighted least squares IDIM-WLS, cf. Poignet, Gautier, 2000 and Gautier, 1997.
            adds weighting with standard dev of estimation error on base regressor and params.
            """

            # get estimation once with previous ordinary LS solution parameters
            self.estimateRegressorTorques('base')

            # get standard deviation of measurement and modeling error \sigma_{rho}^2
            # for each joint subsystem (rho is assumed zero mean independent noise)
            self.sigma_rho = np.square(la.norm(self.tauMeasured-self.tauEstimated, axis=0))/ \
                                  (self.num_used_samples-self.num_base_params)

            # repeat stddev values for each measurement block (n_joints * num_samples)
            # along the diagonal of G
            # G = np.diag(np.repeat(1/self.sigma_rho, self.num_used_samples))
            import scipy.sparse as sparse
            G = sparse.spdiags(np.repeat(1/self.sigma_rho, self.num_used_samples), 0, self.N_DOFS*self.num_used_samples, self.N_DOFS*self.num_used_samples)
            #G = sparse.spdiags(np.tile(1/self.sigma_rho, self.num_used_samples), 0, self.N_DOFS*self.num_used_samples, self.N_DOFS*self.num_used_samples)

            # get standard deviation \sigma_{x} (of the estimated parameter vector x)
            #C_xx = la.norm(self.sigma_rho)*(la.inv(self.YBase.T.dot(self.YBase)))
            #sigma_x = np.sqrt(np.diag(C_xx))

            # weight Y and tau with deviations, identify params
            YBase = G.dot(self.YBase)
            tau = G.dot(self.tau)
            print("Condition number of WLS YBase: {}".format(la.cond(YBase)))

            # get identified values using weighted matrices without weighing them again
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
            self.tauEstimated = np.reshape(tauEst, (self.num_used_samples, self.N_DOFS))

            self.sample_end = self.samples['positions'].shape[0]
            if self.skipSamples > 0: self.sample_end -= (self.skipSamples)

            if self.simulate:
                if self.useAPriori:
                    tau = self.torques_stack    # use original measurements, not delta
                else:
                    tau = self.tau
                self.tauMeasured = np.reshape(tau, (self.num_used_samples, self.N_DOFS))
            else:
                self.tauMeasured = self.samples['torques'][0:self.sample_end:self.skipSamples+1, :]

            self.T = self.samples['times'][0:self.sample_end:self.skipSamples+1]

        #print("torque estimation took %.03f sec." % t.interval)

    def estimateValidationTorques(self, file):
        """ calculate torques of trajectory from validation measurements and identified params """
        # TODO: get identified params directly into idyntree (new KinDynComputations class does not have
        # inverse dynamics yet, so we have to go over a new urdf file now)
        import os

        v_data = np.load(file)
        dynComp = iDynTree.DynamicsComputations();

        self.helpers.replaceParamsInURDF(input_urdf=self.URDF_FILE, output_urdf=self.URDF_FILE + '.tmp',
                                         new_params=self.xStd, link_names=self.link_names)
        dynComp.loadRobotModelFromFile(self.URDF_FILE + '.tmp')
        os.remove(self.URDF_FILE + '.tmp')

        gravity = iDynTree.SpatialAcc();
        gravity.zero()
        gravity.setVal(2, -9.81);

        self.tauEstimatedValidation = None
        for m_idx in range(0, v_data['positions'].shape[0], self.skipSamples+1):
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
            if self.tauEstimatedValidation is None:
                self.tauEstimatedValidation = torques.toNumPy()
            else:
                self.tauEstimatedValidation = np.vstack((self.tauEstimatedValidation, torques.toNumPy()))

        if self.skipSamples > 0:
            self.tauMeasuredValidation = v_data['torques'][::self.skipSamples+1]
            self.T = v_data['times'][::self.skipSamples+1]
        else:
            self.tauMeasuredValidation = v_data['torques']
            self.T = v_data['times']

        self.val_error = la.norm(self.tauEstimatedValidation-self.tauMeasuredValidation)*100/la.norm(self.tauMeasuredValidation)
        print("Validation error: {}%".format(self.val_error))

    def getBaseEssentialParameters(self):
        """
        iteratively get essential parameters from previously identified base parameters.
        (goal is to get similar influence of all parameters, i.e. decrease condition number by throwing
        out parameters that are too sensitive to errors. The remaining params should be estimated with
        similar accuracy)

        based on Pham, 1991; Gautier, 2013 and Jubien, 2014
        """

        with identificationHelpers.Timer() as t:
            # use mean least squares (actually median least abs) to determine when the error
            # introduced by model reduction gets too large
            use_error_criterion = 1

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

            def error_func(inst):
                rho = inst.tauMeasured-inst.tauEstimated   #error of median over all joints
                #rho = np.mean(inst.tauMeasured-inst.tauEstimated, axis=1)
                #rho = np.square(la.norm(inst.tauMeasured-inst.tauEstimated))
                return rho

            def mse_func(error, inst):
                mse = np.sum(np.abs(error)/inst.tauMeasured.shape[0])
                return mse

            error_start = error_func(self)

            if self.showErrorHistogram:
                h = plt.hist(error_start, 50)
                plt.title("error probability")
                plt.draw()

            #range of measured torque
            #torq_range = np.max(self.tauMeasured, axis=0)+(-1*np.min(self.tauMeasured, axis=0))

            #get starting mean squared error
            #mse_start = mse_func(error_start, self)
            #mse_percent_start = mse_start/np.median(torq_range)
            pham_percent_start = la.norm(self.tauEstimated-self.tauMeasured)*100/la.norm(self.tauMeasured)
            print("starting percentual error {}".format(pham_percent_start))

            k2, p = stats.normaltest(error_start, axis=0)
            if np.mean(p) > 0.05:
                print("error is normal distributed")
            else:
                print("error is not normal distributed (p={})".format(p))

            rho_start = np.square(la.norm(self.tauMeasured-self.tauEstimated))
            p_sigma_x = 0

            run_once = 0
            # start removing non-essential parameters
            while 1:
                # get new torque estimation to calc error norm (new estimation with updated parameters)
                self.estimateRegressorTorques('base')

                # get standard deviation of measurement and modeling error \sigma_{rho}^2
                rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))
                sigma_rho = rho/(self.num_used_samples-self.num_base_params)

                # get standard deviation \sigma_{x} (of the estimated parameter vector x)
                C_xx = sigma_rho*(la.inv(np.dot(self.YBase.T, self.YBase)))
                sigma_x = np.diag(C_xx)

                # get relative standard deviation
                prev_p_sigma_x = p_sigma_x
                p_sigma_x = np.sqrt(sigma_x)
                for i in range(0, p_sigma_x.size):
                    if np.abs(self.xBase[i]) != 0:
                        p_sigma_x[i] /= np.abs(self.xBase[i])

                print("{} params|".format(self.num_base_params-b_c)),

                old_ratio = ratio
                ratio = np.max(p_sigma_x)/np.min(p_sigma_x)
                print "min-max ratio of relative stddevs: {},".format(ratio),

                print("cond(YBase):{},".format(la.cond(self.YBase))),
                #error = error_func(self)

                    #mse = mse_func(error, self)
                    #allow 5% of mean/median of measured value range
                    #mse_meas_torq = mse/np.median(torq_range)
                    #mse_percent = mse_meas_torq
                    #error_increase = mse_percent - mse_percent_start

                    pham_percent = la.norm(self.tauEstimated-self.tauMeasured)*100/la.norm(self.tauMeasured)
                    error_increase_pham = pham_percent - pham_percent_start

                    #print("% mse of torq limits {},".format(mse_max_torq)),
                    #print("% mse of measured range {},".format(mse_meas_torq)),
                    print("error increase {}").format(error_increase_pham)

                # while loop condition moved to here
                # TODO: consider to only stop when under ratio
                # if error is to large at that point, advise to get more/better data
                if ratio < 20:
                    break
                if use_error_criterion and error_increase_pham > 3.5:
                    break

                if run_once and self.showEssentialSteps:
                    # get indices of the essential base params
                    self.baseNonEssentialIdx = not_essential_idx
                    self.baseEssentialIdx = [x for x in range(0,self.num_base_params) if x not in not_essential_idx]
                    self.num_essential_params = len(self.baseEssentialIdx)
                    self.xBase_essential = np.zeros_like(xBase_orig)
                    self.xBase_essential[self.baseEssentialIdx] = self.xBase
                    self.p_sigma_x = p_sigma_x

                    old_showStd = self.showStandardParams
                    old_showBase = self.showStandardParams
                    self.showStandardParams = 1
                    self.showBaseParams = 1
                    self.output()
                    self.showStandardParams = old_showStd
                    self.showBaseParams = old_showBase

                    print base_idx, np.argmax(p_sigma_x)
                    print self.baseNonEssentialIdx
                    raw_input("Press return...")
                else:
                    run_once = 1

                #cancel the parameter with largest deviation
                param_idx = np.argmax(p_sigma_x)
                #get its index among the base params (otherwise it doesnt take deletion into account)
                param_base_idx = base_idx[param_idx]
                if param_base_idx not in not_essential_idx:
                    not_essential_idx.append(param_base_idx)
                else:
                    # if parameter was set to zero and still has the largest std deviation,
                    # something is weird..? (probably only happens when not deleting columns)
                    print("param {} already canceled before, stopping".format(param_base_idx))
                    break

                self.prev_xBase = self.xBase.copy()
                self.xBase = np.delete(self.xBase, param_idx, 0)
                base_idx = np.delete(base_idx, param_idx, 0)
                self.YBase = np.delete(self.YBase, param_idx, 1)
                b_c += 1

            not_essential_idx.pop()
            print("essential rel stddevs: {}".format(prev_p_sigma_x))
            self.p_sigma_x = prev_p_sigma_x

            # get indices of the essential base params
            self.baseNonEssentialIdx = not_essential_idx
            self.baseEssentialIdx = [x for x in range(0,self.num_base_params) if x not in not_essential_idx]
            self.num_essential_params = len(self.baseEssentialIdx)

            # leave previous base params and regressor unchanged
            self.xBase_essential = np.zeros_like(xBase_orig)
            self.xBase_essential[self.baseEssentialIdx] = self.prev_xBase
            self.YBase = YBase_orig
            self.xBase = xBase_orig

            print "Got {} essential parameters".format(self.num_essential_params)

        if self.showTiming:
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
            useCADWeighting = 0   # usually produces exact same result, but might be good for some tests
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

            # remove mass params if present
            if self.dontIdentifyMasses:
                ps = range(0,self.N_PARAMS, 10)
                self.stdEssentialIdx = np.fromiter((x for x in self.stdEssentialIdx if x not in ps), int)

            # add some other params for testing (useCADWeighting and maybe increase change num_essential_params)
            #self.stdEssentialIdx = np.concatenate((self.stdEssentialIdx, [19,22,21,25,26,27,31,33,34,35,38,41,42,44,45,46,48,49,55,56,57,58,63,64,66,68,70,75,76,79]))

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
        if self.showTiming:
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
        if self.showTiming:
            print("Identifying std parameters took %.03f sec." % t.interval)

    def identifyStandardEssentialParameters(self):
        """Identify standard essential parameters directly with non-singular standard regressor."""
        with identificationHelpers.Timer() as t:
            # weighting with previously determined essential params
            # calculates V_1e, U_1e etc. (Gautier, 2013)
            Y = self.YStd.dot(np.diag(self.xStdEssential))   #= W_st^e
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

        if self.showTiming:
            print("Identifying %s std essential parameters took %.03f sec." % (len(self.stdEssentialIdx), t.interval))


    def estimateParameters(self):
        print("doing identification on {} samples".format(self.num_used_samples)),

        self.computeRegressors()
        if self.useEssentialParams:
            self.getBaseRegressor()
            self.identifyBaseParameters()
            self.getBaseEssentialParameters()
            self.getStdEssentialParameters()
            self.getNonsingularRegressor()
            self.identifyStandardEssentialParameters()
            if self.useAPriori:
                self.getBaseParamsFromParamError()
        else:
            self.getBaseRegressor()
            if self.estimateWith in ['base', 'std']:
                self.identifyBaseParameters()
                self.getStdFromBase()
                if self.useAPriori:
                    self.getBaseParamsFromParamError()
            elif self.estimateWith is 'std_direct':
                self.getNonsingularRegressor()
                self.identifyStandardParameters()

    def output(self, summary_only=False):
        """Do some pretty printing of parameters."""

        import colorama
        from colorama import Fore, Back, Style
        colorama.init(autoreset=False)

        if not self.useEssentialParams:
            self.stdEssentialIdx = range(0, self.N_PARAMS)
            self.stdNonEssentialIdx = []

        #if requested, load params from other urdf for comparison
        if self.urdf_file_real:
            dc = iDynTree.DynamicsComputations()
            dc.loadRobotModelFromFile(self.urdf_file_real)
            tmp = iDynTree.VectorDynSize(self.N_PARAMS)
            dc.getModelDynamicsParameters(tmp)
            xStdReal = tmp.toNumPy()
            xBaseReal = np.dot(self.B.T, xStdReal)

        if self.showStandardParams:
            # convert params to COM-relative instead of frame origin-relative (linearized parameters)
            if self.outputBarycentric:
                if not self.robotranRegressor:
                  xStd = self.helpers.paramsLink2Bary(self.xStd)
                xStdModel = self.helpers.paramsLink2Bary(self.xStdModel)
                if not summary_only:
                    print("Barycentric (relative to COM) Standard Parameters")
            else:
                xStd = self.xStd
                xStdModel = self.xStdModel
                if not summary_only:
                    print("Linear (relative to Frame) Standard Parameters")

            # collect values for parameters
            description = self.generator.getDescriptionOfParameters()
            idx_p = 0   #count (std) params
            idx_ep = 0  #count essential params
            lines = list()
            sum_diff_r_pc_ess = 0
            sum_diff_r_pc_all = 0
            sum_pc_delta_all = 0
            sum_pc_delta_ess = 0
            for d in description.replace(r'Parameter ', '# ').replace(r'first moment', 'center').split('\n'):
                #print beginning of each link block in green
                if idx_p % 10 == 0:
                    d = Fore.GREEN + d

                #get some error values for each parameter
                approx = xStd[idx_p]
                apriori = xStdModel[idx_p]

                if self.urdf_file_real:
                    real = xStdReal[idx_p]
                    # set real params that are 0 to some small value
                    #if real == 0: real = 0.01

                    # get error percentage (new to real)
                    # so if 100% are the real value, how big is the error
                    diff_real = approx - real
                    if real != 0:
                        diff_r_pc = (100*diff_real)/real
                    else:
                        diff_r_pc = (100*diff_real)/0.01

                    #add to final error percent sum
                    sum_diff_r_pc_all += np.abs(diff_r_pc)
                    if idx_p in self.stdEssentialIdx:
                        sum_diff_r_pc_ess += np.abs(diff_r_pc)

                    # get error percentage (new to apriori)
                    diff_apriori = apriori - real
                    #if apriori == 0: apriori = 0.01
                    if diff_apriori != 0: pc_delta = np.abs((100/diff_apriori)*diff_real)
                    else: pc_delta = 0
                    sum_pc_delta_all += pc_delta
                    if idx_p in self.stdEssentialIdx:
                        sum_pc_delta_ess += pc_delta
                else:
                    # get percentage difference between apriori and identified values
                    # (shown when real values are not known)
                    diff = approx - apriori
                    if apriori != 0:
                        diff_pc = (100*diff)/apriori
                    else:
                        diff_pc = (100*diff)/0.01

                #values for each line
                if self.useEssentialParams and idx_ep < self.num_essential_params and idx_p in self.stdEssentialIdx:
                    sigma = self.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if self.urdf_file_real:
                    vals = [real, apriori, approx, diff_real, np.abs(diff_r_pc), pc_delta, sigma, d]
                else:
                    vals = [apriori, approx, diff, diff_pc, sigma, d]
                lines.append(vals)

                if self.useEssentialParams and idx_p in self.stdEssentialIdx:
                    idx_ep+=1
                idx_p+=1
                if idx_p == len(xStd):
                    break

            if self.urdf_file_real:
                column_widths = [13, 13, 13, 7, 7, 7, 6, 45]
                precisions = [8, 8, 8, 4, 1, 1, 3, 0]
            else:
                column_widths = [13, 13, 7, 7, 6, 45]
                precisions = [8, 8, 4, 1, 3, 0]

            if not summary_only:
                # print column header
                template = ''
                for w in range(0, len(column_widths)):
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                if self.urdf_file_real:
                    print template.format("'Real'", "A priori", "Approx", "Change", "%e", u"Δ%e".encode('utf-8'), u"%σ".encode('utf-8'), "Description")
                else:
                    print template.format("A priori", "Approx", "Change", "%e", u"%σ".encode('utf-8'), "Description")

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
                    print t,
                    idx_p+=1
                    print Style.RESET_ALL
                print "\n"

        ### print base params
        if self.showBaseParams and self.estimateWith not in ['urdf', 'std_direct']:
            print("Base Parameters and Corresponding standard columns")
            if not self.useEssentialParams:
                baseEssentialIdx = range(0, self.num_base_params)
                baseNonEssentialIdx = []
                xBase_essential = self.xBase
            else:
                baseEssentialIdx = self.baseEssentialIdx
                baseNonEssentialIdx = self.baseNonEssentialIdx
                xBase_essential = self.xBase_essential

            # collect values for parameters
            idx_ep = 0
            lines = list()
            for idx_p in range(0, self.num_base_params):
                if self.useEssentialParams: # and xBase_essential[idx_p] != 0:
                    new = xBase_essential[idx_p]
                else:
                    new = self.xBase[idx_p]
                old = self.xBaseModel[idx_p]
                diff = new - old
                if self.urdf_file_real:
                    real = xBaseReal[idx_p]
                    error = new - real

                # collect linear dependencies for this param
                deps = np.where(np.abs(self.linear_deps[idx_p, :])>0.1)[0]
                dep_factors = self.linear_deps[idx_p, deps]

                param_columns = ' |{}|'.format(self.independent_cols[idx_p])
                if len(deps):
                    param_columns += " deps:"
                for p in range(0, len(deps)):
                    param_columns += ' {:.4f}*|{}|'.format(dep_factors[p], self.P[self.num_base_params:][deps[p]])

                if self.useEssentialParams and idx_p in self.baseEssentialIdx:
                    sigma = self.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if self.urdf_file_real:
                    lines.append((idx_p, real, old, new, diff, error, sigma, param_columns))
                else:
                    lines.append((old, new, diff, sigma, param_columns))

                if self.useEssentialParams and idx_p in self.baseEssentialIdx:
                    idx_ep+=1

            if self.urdf_file_real:
                column_widths = [3, 13, 13, 13, 7, 7, 6, 30]   # widths of the columns
                precisions = [0, 8, 8, 8, 4, 4, 3, 0]         # numerical precision
            else:
                column_widths = [13, 13, 7, 6, 30]   # widths of the columns
                precisions = [8, 8, 4, 3, 0]         # numerical precision

            if not summary_only:
                # print column header
                template = ''
                for w in range(0, len(column_widths)):
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                if self.urdf_file_real:
                    print template.format("\#", "Real", "Model", "Approx", "Change", "Error", u"%σ".encode('utf-8'), "Description")
                else:
                    print template.format("Model", "Approx", "Change", u"%σ".encode('utf-8'), "Description")

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
                    elif idx_p in baseEssentialIdx:
                        t = Style.BRIGHT + t
                    if self.showEssentialSteps and l[-2] == np.max(self.p_sigma_x):
                        t = Fore.CYAN + t
                    print t,
                    idx_p+=1
                    print Style.RESET_ALL

        if self.selectBlocksFromMeasurements:
            print "used blocks: {}".format(self.usedBlocks)
            print "unused blocks: {}".format(self.unusedBlocks)
            print "condition number: {}".format(la.cond(self.YBase))

        if self.showStandardParams:
            print("Per-link physical consistency (a priori): {}".format(self.helpers.checkPhysicalConsistency(self.xStdModel)))
            print("Per-link physical consistency (identified): {}".format(self.helpers.checkPhysicalConsistency(self.xStd)))

        if self.urdf_file_real:
            if self.showStandardParams:
                if self.useEssentialParams:
                    print("Mean relative error of essential std params: {}%".\
                            format(sum_diff_r_pc_ess/len(self.stdEssentialIdx)))
                print("Mean relative error of all std params: {}%".format(sum_diff_r_pc_all/len(self.xStd)))

                if self.useEssentialParams:
                    print("Mean error delta (apriori error vs approx error) of essential std params: {}%".\
                            format(sum_pc_delta_ess/len(self.stdEssentialIdx)))
                print("Mean error delta (apriori error vs approx error) of all std params: {}%".\
                        format(sum_pc_delta_all/len(self.xStd)))

        self.res_error = la.norm(self.tauEstimated-self.tauMeasured)*100/la.norm(self.tauMeasured)
        self.estimateRegressorTorques(estimateWith='urdf')
        self.apriori_error = la.norm(self.tauEstimated-self.tauMeasured)*100/la.norm(self.tauMeasured)

        print("Relative residual error (torque prediction): {}% vs. A priori error: {}%".\
                format(self.res_error, self.apriori_error))

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
            #([self.samples['positions'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Positions'),
            #([self.samples['velocities'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Vels'),
            #([self.samples['accelerations'][self.start_offset:self.sample_end:self.skip_samples+1]], 'Accls'),
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
            for d_i in range(0, len(data)):
                if len(data[d_i].shape) > 1:
                    for i in range(0, data[d_i].shape[1]):
                        l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                        plt.plot(self.T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
                else:
                    plt.plot(self.T, data[d_i], label=title, color=colors[0], alpha=1-(d_i/2.0))

            leg = plt.legend(loc='best', fancybox=True, fontsize=10)
            leg.draggable()
        plt.show()
        #self.samples.close()

    def printMemUsage(self):
        import humanize
        total = 0
        print "Memory usage:"
        for v in self.__dict__.keys():
            if type(self.__dict__[v]).__module__ == np.__name__:
                size = self.__dict__[v].nbytes
                total += size
                print "{}: {} ".format( v, (humanize.naturalsize(size, binary=True)) ),
        print "- total: {}".format(humanize.naturalsize(total, binary=True))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('-m', '--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--model_real', required=False, type=str, help='the file to load the model params for\
                        comparison from')
    parser.add_argument('-o', '--model_output', required=False, type=str, help='the file to save the identified params to')

    parser.add_argument('--measurements', required=True, nargs='+', action='append', type=str,
                        help='the file(s) to load the measurements from')

    parser.add_argument('--validation', required=False, type=str,
                        help='the file to load the validation trajectory from')

    parser.add_argument('--regressor', required=False, type=str,
                        help='the file containing the regressor structure(for the iDynTree generator).\
                              Identifies on all joints if not specified.')

    parser.add_argument('--plot', help='whether to plot measurements', action='store_true')
    parser.add_argument('-e', '--explain', help='whether to explain identified parameters', action='store_true')
    parser.set_defaults(plot=False, explain=False, regressor=None, model_real=None)
    args = parser.parse_args()

    idf = Identification(args.model, args.model_real, args.measurements, args.regressor)
    idf.initRegressors()

    if idf.selectBlocksFromMeasurements:
        old_essential_option = idf.useEssentialParams
        idf.useEssentialParams = 0
        # loop over input blocks and select good ones
        while 1:
            idf.estimateParameters()
            idf.estimateRegressorTorques()
            #if args.validation: idf.estimateValidationTorques(args.validation)
            if(idf.checkBlockImprovement()):
                idf.output(summary_only=True)

            if idf.hasMoreSamples():
                idf.nextSampleBlock()
            else:
                break
        idf.useEssentialParams = old_essential_option

    print("estimating output parameters...")
    idf.estimateParameters()
    idf.estimateRegressorTorques()

    if idf.showMemUsage: idf.printMemUsage()
    if args.model_output:
        idf.helpers.replaceParamsInURDF(input_urdf=args.model, output_urdf=args.model_output, \
                                        new_params=idf.xStd, link_names=idf.link_names)

    if args.explain: idf.output()
    if args.validation: idf.estimateValidationTorques(args.validation)
    if args.plot: idf.plot()

if __name__ == '__main__':
   # import ipdb
   # import traceback
    #try:
    main()
    print "\n"

    '''
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            # open ipdb when an exception happens
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
    '''
