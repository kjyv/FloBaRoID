#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys

# math
import numpy as np
import numpy.linalg as la
import scipy
import scipy.linalg as sla
import scipy.stats as stats

import sympy
from sympy import Symbol, symbols, solve, Eq, Matrix, BlockMatrix, Identity, sympify, eye
version = int(sympy.__version__.replace('.','')[:3])
is_old_sympy = (version <= 74 and not sympy.__version__.startswith('1'))

# plotting
import matplotlib
import matplotlib.pyplot as plt

# kinematics, dynamics and URDF reading
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

# submodules
from model import Model
from data import Data
from output import OutputConsole
import helpers

import optimization
from optimization import LMI_PSD, LMI_PD
import lmi_sdp

# stuff
from colorama import Fore, Back, Style
from IPython import embed

# Referenced papers:
# Gautier, 2013: Identification of Consistent Standard Dynamic Parameters of Industrial Robots
# Gautier, 1991: Numerical Calculation of the base Inertial Parameters of Robots
# Pham, 1991: Essential Parameters of Robots
# Zag et. al, 1991: Application of the Weighted Least Squares Parameter Estimation Method to the
# Robot Calibration
# Venture et al, 2009: A numerical method for choosing motions with optimal excitation properties
# for identification of biped dynamics
# Jubien, 2014: Dynamic identification of the Kuka LWR robot using motor torques and joint torque
# sensors data
# Sousa, 2014: Physical feasibility of robot base inertial parameter identification: A linear matrix
# inequality approach

# TODO: add/use contact forces / floating base
# TODO: add friction identification
# TODO: load full model and programatically cut off chain from certain joints/links to allow
# subtree identification
# TODO: use experiment config files, read options and data paths from there
# TODO: allow visual filtering and data selection (take raw data as input)

class Identification(object):
    def __init__(self, urdf_file, urdf_file_real, measurements_files, regressor_file, validation_file):
        ## options
        self.opt = {}

        # determine number of samples to use
        # (Khalil recommends about 500 times number of parameters to identify...)
        self.opt['start_offset'] = 400  #how many samples from the beginning of the (first) measurement are skipped
        self.opt['skip_samples'] = 2    #how many values to skip before using the next sample

        # simulate torques from target values, don't use both
        self.opt['iDynSimulate'] = 0 # simulate torque for measured angles etc using idyntree (instead of reading data)
        self.opt['addNoise'] = 0 #0.05  #additional percentage of zero-mean white noise for simulated or measured torques

        # use previously known CAD parameters to identify parameter error, estimates parameters closer to
        # known ones (taken from URDF file)
        # for some methods, this gives parameters that are more likely to be consistent
        # (no effect for SDP constrained solutions)
        self.opt['useAPriori'] = 1

        ####

        # whether only "good" data is being selected or simply all is used
        self.opt['selectBlocksFromMeasurements'] = 0
        self.opt['block_size'] = 250  # needs to be at least as much as parameters so regressor is square or higher

        ####

        # constrain std params to physical consistent space to only achieve physical consistent parameters
        # (currently this also does the estimation, so previously selecting another method has no effect)
        # if only torque estimation is desired, not using this self.option might give a better model
        # accuracy with approriate parameters
        self.opt['useConsistencyConstraints'] = 1

        # constrain parameters for links more than a certain condition number to the a priori values
        # (to prevent very big changes for parameters that are not expressed in the data)
        self.opt['noChange'] = 1
        self.opt['noChangeThresh'] = 200

        # restrict COM to smallest enclosing box of STL Mesh (taken from <visual> in URDF)
        self.opt['restrictCOMtoHull'] = 1
        # set extra scaling for mesh (e.g. if it is clear that COM will not be at outer border of
        # geometry or that initial CAD data is too large)
        self.opt['hullScaling'] = 1.0

        # constrain overall mass
        self.opt['limitOverallMass'] = 0
        # if overall mass is set, limit to this value. If None, limit to overall a priori mass +- 30%
        self.opt['limitMassVal'] = None #16

        # enforce the same upper limit for each link mass
        self.opt['limitMassValPerLink'] = None #3

        # or enforce staying around the a priori masses (only set this or a combination of the other
        # two mass limiting self.options to prevent constraint conflicts!)
        self.opt['limitMassToApriori'] = 1
        self.opt['limitMassAprioriBoundary'] = 1.0     #percentage of CAD value in both +- directions

        # whether to take out masses to be identified because they are e.g.
        # well known or introduce problems
        # (essential params or when using feasability constraints)
        self.opt['dontIdentifyMasses'] = 0

        ####

        # whether to identify and use direct standard with essential parameters
        self.opt['useEssentialParams'] = 0

        # whether to include linear dependent columns in essential params or not
        self.opt['useDependents'] = 1

        # use weighted least squares(WLS) instead of ordinary least squares
        # needs small condition number, otherwise might amplify some parameters too much as the
        # covariance estimation can be off (also assumes that error is zero mean and normal
        # distributed)
        self.opt['useWLS'] = 0

        # whether to filter the regressor columns
        # (cutoff frequency is system dependent)
        # mostly not improving results
        self.opt['filterRegressor'] = 0

        ####

        # how to output plots and other stuff ['matplotlib', 'html']
        self.opt['outputModule'] = 'html'

        # self.options for console output
        self.opt['showMemUsage'] = 0          #print used memory for different variables
        self.opt['showTiming'] = 0            #show times various steps have taken
        self.opt['showRandomRegressor'] = 0   #show 2d plot of random regressor
        self.opt['showErrorHistogram'] = 0    #show estimation error distribution
        self.opt['showEssentialSteps'] = 0    #stop after every reduction step and show values
        self.opt['outputBarycentric'] = 0     #output all values in barycentric (e.g. urdf) form
        self.opt['showStandardParams'] = 1
        self.opt['showBaseParams'] = 1

        ####

        #some experiments

        # orthogonalize basis matrix (uglier linear relationships, should not change results)
        self.opt['orthogonalizeBasis'] = 0

        #project a priori to solution subspace
        self.opt['projectToAPriori'] = 0

        # which parameters to use when estimating torques for validation. Set to one of
        # ['base', 'std', 'std_direct', 'urdf']
        self.opt['estimateWith'] = 'std'

        #### end self.options

        self.opt['min_tol'] = 1e-5    # almost zero threshold for SVD and QR

        # load model description and initialize
        self.model = Model(self.opt, urdf_file, regressor_file)

        # load measurements
        self.data = Data(self.opt)
        self.data.init_from_files(measurements_files)

        self.tauEstimated = list()
        self.tauMeasured = list()

        self.paramHelpers = helpers.ParamHelpers(self.model.num_params)
        self.urdfHelpers = helpers.URDFHelpers(self.paramHelpers, self.model.link_names)

        self.res_error = 100
        self.urdf_file_real = urdf_file_real
        self.validation_file = validation_file


    def identifyBaseParameters(self, YBase=None, tau=None):
        """use previously computed regressors and identify base parameter vector using ordinary or weighted least squares."""

        if YBase is None:
            YBase = self.model.YBase
        if tau is None:
            tau = self.model.tau

        # TODO: get jacobian and contact force for each contact frame (when added to iDynTree)
        # in order to also use FT sensors in hands and feet
        # assuming zero external forces for fixed base on trunk
        # jacobian = iDynTree.MatrixDynSize(6,6+N_DOFS)
        # self.generator.getFrameJacobian('arm', jacobian)

        # in case B is not an orthogonal base (B.T != B^-1), we have to use pinv instead of T
        # (using QR on B yields orthonormal base if necessary)
        # in general, pinv is always working correctly
        self.model.xBaseModel = np.dot(self.model.Binv, self.model.xStdModel)

        # invert equation to get parameter vector from measurements and model + system state values
        self.model.YBaseInv = la.pinv(self.model.YBase)
        self.model.xBase = np.dot(self.model.YBaseInv, self.model.tau.T) # - np.sum( YBaseInv*jacobian*contactForces )

        #damped least squares
        #from scipy.sparse.linalg import lsqr
        #self.model.xBase = lsqr(YBase, tau, damp=10)[0]

        #ordinary least squares with numpy method
        #self.model.xBase = la.lstsq(YBase, tau)[0]

        if self.opt['useWLS']:
            """
            additionally do weighted least squares IDIM-WLS, cf. Zak, 1991 and Gautier, 1997.
            adds weighting with standard dev of estimation error on base regressor and params.
            """

            # get estimation once with previous ordinary LS solution parameters
            self.estimateRegressorTorques('base')

            # get standard deviation of measurement and modeling error \sigma_{rho}^2
            # for each joint subsystem (rho is assumed zero mean independent noise)
            self.sigma_rho = np.square(sla.norm(self.tauEstimated))/ \
                                       (self.data.num_used_samples-self.model.num_base_params)

            # repeat stddev values for each measurement block (n_joints * num_samples)
            # along the diagonal of G
            # G = np.diag(np.repeat(1/self.sigma_rho, self.num_used_samples))
            G = scipy.sparse.spdiags(np.repeat(1/self.sigma_rho, self.data.num_used_samples), 0,
                               self.model.N_DOFS*self.data.num_used_samples, self.model.N_DOFS*self.data.num_used_samples)
            #G = scipy.sparse.spdiags(np.tile(1/self.sigma_rho, self.num_used_samples), 0,
            #                   self.N_DOFS*self.num_used_samples, self.N_DOFS*self.num_used_samples)

            # get standard deviation \sigma_{x} (of the estimated parameter vector x)
            #C_xx = la.norm(self.sigma_rho)*(la.inv(self.YBase.T.dot(self.YBase)))
            #sigma_x = np.sqrt(np.diag(C_xx))

            # weight Y and tau with deviations, identify params
            YBase = G.dot(self.model.YBase)
            tau = G.dot(self.model.tau)
            print("Condition number of WLS YBase: {}".format(la.cond(YBase)))

            # get identified values using weighted matrices without weighing them again
            self.opt['useWLS'] = 0
            self.identifyBaseParameters(YBase, tau)
            self.opt['useWLS'] = 1

    def getBaseParamsFromParamError(self):
        self.model.xBase += self.model.xBaseModel   #both param vecs link relative linearized

        if self.opt['useEssentialParams']:
            self.xBase_essential[self.baseEssentialIdx] += self.model.xBaseModel[self.baseEssentialIdx]

    def findStdFromBaseParameters(self):
        # Note: assumes that xBase is still in error form if using a priori
        # i.e. don't call after getBaseParamsFromParamError

        # project back to standard parameters
        self.model.xStd = self.model.B.dot(self.model.xBase)

        # get estimated parameters from estimated error (add a priori knowledge)
        if self.opt['useAPriori']:
            self.model.xStd += self.model.xStdModel
        elif self.opt['projectToAPriori']:
            #add a priori parameters projected on non-identifiable subspace
            self.model.xStd += (np.eye(self.model.B.shape[0])-self.model.B.dot(self.model.Binv)).dot(self.model.xStdModel)

            #do projection algebraically
            #for each identified base param,
            base_deps_vals = []
            for idx in range(0,self.model.num_base_params):
                base_deps_vals.append(Eq(self.model.base_deps[idx], self.model.xBase[idx]))

            prev_eq = base_deps_vals[0].lhs - base_deps_vals[0].rhs
            prev_eq2 = prev_eq.copy()
            for eq in base_deps_vals[1:]:
                prev_eq2 -= eq.lhs - eq.rhs

            print "solution space:", prev_eq2

            #get some point in the affine subspace (set all but one var then solve)
            p_on_eq = []
            rns = np.random.rand(len(self.model.param_syms)-1)
            #rns = np.zeros(len(syms)-1)
            #rns[0] = 1
            eq = prev_eq2.subs(list(zip(self.model.param_syms, rns)))   #replace vars with values
            p_on_eq[0:len(rns)] = rns   #collect values
            p_on_eq.append(solve(eq, self.model.param_syms[len(self.model.param_syms)-1])[0])   #solve for remaining (last) symbol
            print("p_on_eq\t", np.array(p_on_eq, dtype=np.float64))
            pham_percent = sla.norm(self.model.YStd.dot(p_on_eq))*100/sla.norm(self.tauMeasured)
            print(pham_percent)


    def estimateRegressorTorques(self, estimateWith=None):
        """ get torque estimations using regressors, prepare for plotting """

        with helpers.Timer() as t:
            if not estimateWith:
                #use global parameter choice if none is given specifically
                estimateWith = self.opt['estimateWith']
            # estimate torques with idyntree regressor and different params
            if estimateWith is 'urdf':
                tauEst = np.dot(self.model.YStd, self.model.xStdModel)
            elif estimateWith is 'base_essential':
                tauEst = np.dot(self.model.YBase, self.xBase_essential)
            elif estimateWith is 'base':
                tauEst = np.dot(self.model.YBase, self.model.xBase)
            elif estimateWith in ['std', 'std_direct']:
                tauEst = np.dot(self.model.YStd, self.model.xStd)
            else:
                print("unknown type of parameters: {}".format(self.opt['estimateWith']))

            # reshape torques into one column per DOF for plotting (NUM_SAMPLES*N_DOFSx1) -> (NUM_SAMPLESxN_DOFS)
            self.tauEstimated = np.reshape(tauEst, (self.data.num_used_samples, self.model.N_DOFS))

        #print("torque estimation took %.03f sec." % t.interval)

    def estimateValidationTorques(self):
        """ calculate torques of trajectory from validation measurements and identified params """
        # TODO: get identified params directly into idyntree (new KinDynComputations class does not
        # have inverse dynamics yet, so we have to go over a new urdf file now)
        import os

        v_data = np.load(self.validation_file)
        dynComp = iDynTree.DynamicsComputations();

        self.urdfHelpers.replaceParamsInURDF(input_urdf=self.model.urdf_file,
                                             output_urdf=self.model.urdf_file + '.tmp',
                                             new_params=self.model.xStd, link_names=self.model.link_names)
        dynComp.loadRobotModelFromFile(self.model.urdf_file + '.tmp')
        os.remove(self.model.urdf_file + '.tmp')

        gravity = iDynTree.SpatialAcc();
        gravity.zero()
        gravity.setVal(2, -9.81);

        self.tauEstimatedValidation = None
        for m_idx in range(0, v_data['positions'].shape[0], self.opt['skip_samples']+1):
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

            torques = iDynTree.VectorDynSize(self.model.N_DOFS)
            baseReactionForce = iDynTree.Wrench()   # assume zero for fixed base, otherwise use e.g. imu data

            # compute inverse dynamics with idyntree (simulate)
            dynComp.inverseDynamics(torques, baseReactionForce)
            if self.tauEstimatedValidation is None:
                self.tauEstimatedValidation = torques.toNumPy()
            else:
                self.tauEstimatedValidation = np.vstack((self.tauEstimatedValidation, torques.toNumPy()))

        if self.opt['skip_samples'] > 0:
            self.tauMeasuredValidation = v_data['torques'][::self.opt['skip_samples']+1]
            self.Tv = v_data['times'][::self.opt['skip_samples']+1]
        else:
            self.tauMeasuredValidation = v_data['torques']
            self.Tv = v_data['times']

        self.val_error = la.norm(self.tauEstimatedValidation-self.tauMeasuredValidation) \
                            *100/la.norm(self.tauMeasuredValidation)
        print("Validation error (std params): {}%".format(self.val_error))

    def findBaseEssentialParameters(self):
        """
        iteratively get essential parameters from previously identified base parameters.
        (goal is to get similar influence of all parameters, i.e. decrease condition number by throwing
        out parameters that are too sensitive to errors. The remaining params should be estimated with
        similar accuracy)

        based on Pham, 1991; Gautier, 2013 and Jubien, 2014
        """

        with helpers.Timer() as t:
            # use mean least squares (actually median least abs) to determine when the error
            # introduced by model reduction gets too large
            use_error_criterion = 0

            # keep current values
            xBase_orig = self.model.xBase.copy()
            YBase_orig = self.model.YBase.copy()

            # count how many params were canceled
            b_c = 0

            # list of param indices to keep the original indices when deleting columns
            base_idx = range(0, self.model.num_base_params)
            not_essential_idx = list()
            ratio = 0

            # get initial errors of estimation
            self.estimateRegressorTorques('base')

            if not self.opt['useAPriori']:
                tauDiff = self.tauMeasured - self.tauEstimated
            else:
                tauDiff = self.tauEstimated

            def error_func(tauDiff):
                #rho = tauDiff
                rho = np.mean(tauDiff, axis=1)
                #rho = np.square(la.norm(tauDiff, axis=1))
                return rho

            error_start = error_func(tauDiff)

            k2, p = stats.normaltest(error_start, axis=0)
            if np.mean(p) > 0.05:
                print("error is normal distributed")
            else:
                print("error is not normal distributed (p={})".format(p))

            if self.opt['showErrorHistogram']:
                h = plt.hist(error_start, 50)
                plt.title("error probability")
                plt.draw()
                plt.show()

            pham_percent_start = sla.norm(tauDiff)*100/sla.norm(self.tauMeasured)
            print("starting percentual error {}".format(pham_percent_start))

            rho_start = np.square(sla.norm(tauDiff))
            p_sigma_x = 0

            has_run_once = 0
            # start removing non-essential parameters
            while 1:
                # get new torque estimation to calc error norm (new estimation with updated parameters)
                self.estimateRegressorTorques('base')

                # get standard deviation of measurement and modeling error \sigma_{rho}^2
                rho = np.square(sla.norm(tauDiff))
                sigma_rho = rho/(self.data.num_used_samples-self.model.num_base_params)

                # get standard deviation \sigma_{x} (of the estimated parameter vector x)
                C_xx = sigma_rho*(sla.inv(np.dot(self.model.YBase.T, self.model.YBase)))
                sigma_x = np.diag(C_xx)

                # get relative standard deviation
                prev_p_sigma_x = p_sigma_x
                p_sigma_x = np.sqrt(sigma_x)
                for i in range(0, p_sigma_x.size):
                    if self.model.xBase[i] != 0:
                        p_sigma_x[i] /= np.abs(self.model.xBase[i])

                print("{} params|".format(self.model.num_base_params-b_c)),

                old_ratio = ratio
                ratio = np.max(p_sigma_x)/np.min(p_sigma_x)
                print "min-max ratio of relative stddevs: {},".format(ratio),

                print("cond(YBase):{},".format(la.cond(self.model.YBase))),

                if not self.opt['useAPriori']:
                    tauDiff = self.tauMeasured - self.tauEstimated
                else:
                    tauDiff = self.tauEstimated
                pham_percent = sla.norm(tauDiff)*100/sla.norm(self.tauMeasured)
                error_increase_pham = pham_percent_start - pham_percent
                print("error delta {}").format(error_increase_pham)

                # while loop condition moved to here
                # TODO: consider to only stop when under ratio and
                # if error is to large at that point, advise to get more/better data
                if ratio < 21:
                    break
                if use_error_criterion and error_increase_pham > 3.5:
                    break

                if has_run_once and self.opt['showEssentialSteps']:
                    # put some values into global variable for output
                    self.baseNonEssentialIdx = not_essential_idx
                    self.baseEssentialIdx = [x for x in range(0,self.model.num_base_params) if x not in not_essential_idx]
                    self.num_essential_params = len(self.baseEssentialIdx)
                    self.xBase_essential = np.zeros_like(xBase_orig)

                    # take current xBase with reduced parameters as essentials to display
                    self.xBase_essential[self.baseEssentialIdx] = self.model.xBase

                    self.p_sigma_x = p_sigma_x

                    old_showStd = self.opt['showStandardParams']
                    old_showBase = self.opt['showBaseParams']
                    self.opt['showStandardParams'] = 0
                    self.opt['showBaseParams'] = 1
                    OutputConsole.render(self)
                    self.opt['showStandardParams'] = old_showStd
                    self.opt['showBaseParams'] = old_showBase

                    print base_idx, np.argmax(p_sigma_x)
                    print self.baseNonEssentialIdx
                    raw_input("Press return...")
                else:
                    has_run_once = 1

                #cancel the parameter with largest deviation
                param_idx = np.argmax(p_sigma_x)
                #get its index among the base params (otherwise it doesnt take deletion into account)
                param_base_idx = base_idx[param_idx]
                if param_base_idx not in not_essential_idx:
                    not_essential_idx.append(param_base_idx)

                self.prev_xBase = self.model.xBase.copy()
                self.model.xBase = np.delete(self.model.xBase, param_idx, 0)
                base_idx = np.delete(base_idx, param_idx, 0)
                self.model.YBase = np.delete(self.model.YBase, param_idx, 1)

                # re-estimate parameters with reduced regressor
                self.identifyBaseParameters()

                b_c += 1

            not_essential_idx.pop()
            print("essential rel stddevs: {}".format(prev_p_sigma_x))
            self.p_sigma_x = prev_p_sigma_x

            # get indices of the essential base params
            self.baseNonEssentialIdx = not_essential_idx
            self.baseEssentialIdx = [x for x in range(0,self.model.num_base_params) if x not in not_essential_idx]
            self.num_essential_params = len(self.baseEssentialIdx)

            # leave previous base params and regressor unchanged
            self.xBase_essential = np.zeros_like(xBase_orig)
            self.xBase_essential[self.baseEssentialIdx] = self.prev_xBase
            self.model.YBase = YBase_orig
            self.model.xBase = xBase_orig

            print "Got {} essential parameters".format(self.num_essential_params)

        if self.opt['showTiming']:
            print("Getting base essential parameters took %.03f sec." % t.interval)

    def findStdFromBaseEssParameters(self):
        """
        Find essential standard parameters from previously determined base essential parameters.
        """

        with helpers.Timer() as t:
            # get the choice of indices into the std params of the independent columns.
            # Of those, only select the std parameters that are essential
            self.stdEssentialIdx = self.model.independent_cols[self.baseEssentialIdx]

            # intuitively, also the dependent columns should be essential as the linear combination
            # is used to identify and calc the error
            useCADWeighting = 0   # usually produces exact same result, but might be good for some tests
            if self.opt['useDependents']:
                # also get the ones that are linearly dependent on them -> base params
                dependents = []
                #to_delete = []
                for i in range(0, self.model.linear_deps.shape[0]):
                    if i in self.baseEssentialIdx:
                        for s in self.model.base_deps[i].free_symbols:
                            idx = self.model.param_syms.index(s)
                            if idx not in dependents:
                                dependents.append(idx)

                #print self.stdEssentialIdx
                #print len(dependents)
                print dependents
                self.stdEssentialIdx = np.concatenate((self.stdEssentialIdx, dependents))

            #np.delete(self.stdEssentialIdx, to_delete, 0)

            # remove mass params if present
            if self.opt['dontIdentifyMasses']:
                ps = range(0,self.model.num_params, 10)
                self.stdEssentialIdx = np.fromiter((x for x in self.stdEssentialIdx if x not in ps), int)

            self.stdNonEssentialIdx = [x for x in range(0, self.model.num_params) if x not in self.stdEssentialIdx]

            ## get \hat{x_e}, set zeros for non-essential params
            if self.opt['useDependents'] or useCADWeighting:
                # we don't really know what the weights are if we have more std essential than base
                # essentials, so use CAD/previous params for weighting
                self.xStdEssential = self.model.xStdModel.copy()

                # set essential but zero cad values to small values that are in possible range of those parameters
                # so something can be estimated
                #self.xStdEssential[np.where(self.xStdEssential == 0)[0]] = .1
                idx = 0
                for p in self.xStdEssential:
                    if p == 0:
                        v = 0.1
                        p_start = idx/10*10
                        if idx % 10 in [1,2,3]:   #com value
                            v = np.mean(self.model.xStdModel[p_start + 1:p_start + 4]) * 0.1
                        elif idx % 10 in [4,5,6,7,8,9]:  #inertia value
                            inertia_range = np.array([4,5,6,7,8,9])+p_start
                            v = np.mean(self.model.xStdModel[np.where(self.model.xStdModel[inertia_range] != 0)[0]+p_start+4]) * 0.1
                        if v == 0: v = 0.1
                        self.xStdEssential[idx] = v
                        #print idx, idx % 10, v
                    idx += 1

                # cancel non-essential std params so they are not identified
                self.xStdEssential[self.stdNonEssentialIdx] = 0
            else:
                # weighting using base essential params (like in Gautier, 2013)
                self.xStdEssential = np.zeros_like(self.model.xStdModel)
                #if self.opt['useAPriori']:
                #    self.xStdEssential[self.stdEssentialIdx] = self.xBase_essential[self.baseEssentialIdx] \
                #        + self.xBaseModel[self.baseEssentialIdx]
                #else:
                self.xStdEssential[self.stdEssentialIdx] = self.xBase_essential[self.baseEssentialIdx]

    def identifyStandardParameters(self):
        """Identify standard parameters directly with non-singular standard regressor."""
        with helpers.Timer() as t:
            U, s, VH = la.svd(self.model.YStd, full_matrices=False)
            nb = self.model.num_base_params

            #identify standard parameters directly
            V_1 = VH.T[:, 0:nb]
            U_1 = U[:, 0:nb]
            s_1 = np.diag(s[0:nb])
            s_1_inv = la.inv(s_1)
            W_st_pinv = V_1.dot(s_1_inv).dot(U_1.T)
            W_st = la.pinv(W_st_pinv)

            x_est = W_st_pinv.dot(self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_est
            else:
                self.model.xStd = x_est

            """
            st = self.model.num_params
            # non-singular YStd, called W_st in Gautier, 2013
            self.YStdHat = self.YStd - U[:, nb:st].dot(np.diag(s[nb:st])).dot(V[:,nb:st].T)
            self.YStdHatInv = la.pinv(self.YStdHat)
            x_tmp = np.dot(self.YStdHatInv, self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_tmp
            else:
                self.model.xStd = x_tmp
            """
        if self.opt['showTiming']:
            print("Identifying std parameters directly took %.03f sec." % t.interval)

    def identifyStandardEssentialParameters(self):
        """Identify standard essential parameters directly with non-singular standard regressor."""
        with helpers.Timer() as t:
            # weighting with previously determined essential params
            # calculates V_1e, U_1e etc. (Gautier, 2013)
            Yst_e = self.model.YStd.dot(np.diag(self.xStdEssential))   #= W_st^e
            Ue, se, VHe = sla.svd(Yst_e, full_matrices=False)
            ne = self.num_essential_params  #nr. of essential params among base params
            V_1e = VHe.T[:, 0:ne]
            U_1e = Ue[:, 0:ne]
            s_1e_inv = sla.inv(np.diag(se[0:ne]))
            W_st_e_pinv = np.diag(self.xStdEssential).dot(V_1e.dot(s_1e_inv).dot(U_1e.T))
            W_st_e = la.pinv(W_st_e_pinv)

            x_tmp = W_st_e_pinv.dot(self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_tmp
            else:
                self.model.xStd = x_tmp

        if self.opt['showTiming']:
            print("Identifying %s std essential parameters took %.03f sec." % (len(self.stdEssentialIdx), t.interval))

    def initSDP_LMIs(self):
        # initialize LMI matrices to set physical consistency constraints for SDP solver
        # based on Sousa, 2014 and corresponding code (https://github.com/cdsousa/IROS2013-Feas-Ident-WAM7)

        def skew(v):
            return Matrix([ [     0, -v[2],  v[1] ],
                            [  v[2],     0, -v[0] ],
                            [ -v[1],  v[0],   0 ] ])
        I = Identity
        S = skew

        #correct LMIs if estimating error instead of absolute values
        if self.opt['useAPriori']:
            #LMI contraints need be corrected with a priori knowledge
            apriori = self.model.xStdModel
            #in order to write no-change constraints, compare to zero (changes)
            compare = np.zeros_like(self.model.xStdModel)
        else:
            #dont correct as we're estimating absolute values
            apriori = np.zeros_like(self.model.xStdModel)
            #compare values relative to CAD
            compare = self.model.xStdModel

        # create LMI matrices (symbols) for each link
        # so that mass is positive, inertia matrix is positive definite
        # (each matrix is later on used to be either >0 or >=0)
        D_inertia_blocks = []
        for i in range(0, self.model.N_LINKS):
            m = self.model.mass_syms[i] + apriori[i*10]
            l = Matrix([ self.model.param_syms[i*10+1] + apriori[i*10+1],
                         self.model.param_syms[i*10+2] + apriori[i*10+2],
                         self.model.param_syms[i*10+3] + apriori[i*10+3] ] )
            L = Matrix([ [self.model.param_syms[i*10+4+0] + apriori[i*10+4+0],
                          self.model.param_syms[i*10+4+1] + apriori[i*10+4+1],
                          self.model.param_syms[i*10+4+2] + apriori[i*10+4+2]],
                         [self.model.param_syms[i*10+4+1] + apriori[i*10+4+1],
                          self.model.param_syms[i*10+4+3] + apriori[i*10+4+3],
                          self.model.param_syms[i*10+4+4] + apriori[i*10+4+4]],
                         [self.model.param_syms[i*10+4+2] + apriori[i*10+4+2],
                          self.model.param_syms[i*10+4+4] + apriori[i*10+4+4],
                          self.model.param_syms[i*10+4+5] + apriori[i*10+4+5]]
                       ])

            Di = BlockMatrix([[L,    S(l).T],
                              [S(l), I(3)*m]])
            D_inertia_blocks.append(Di.as_explicit().as_mutable())

        # add conditions for triangle inequality of inertia tensor diagonal
        # TODO: these need to be valid for parameters expressed about COM, not link frame
        # also they have to be the eigenvalues of the inertia tensor instead of diagonal elements
        '''
        for i in range(0, self.model.N_LINKS):
            ixx = self.model.param_syms[i*10+4]
            iyy = self.model.param_syms[i*10+7]
            izz = self.model.param_syms[i*10+9]
            D_inertia_blocks.append(Matrix([ixx+iyy-izz]))
            D_inertia_blocks.append(Matrix([ixx+izz-iyy]))
            D_inertia_blocks.append(Matrix([iyy+izz-ixx]))
        '''
        D_other_blocks = []

        linkConds = self.model.getSubregressorsConditionNumbers()
        robotmass_apriori = 0
        for i in range(0, self.model.N_LINKS):
            robotmass_apriori+= self.model.xStdModel[i*10]  #count a priori link masses

            #for links that have too high condition number, don't change params
            if self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']:
                print Fore.YELLOW + 'skipping identification of link {}!'.format(i) + Fore.RESET
                # don't change mass
                D_other_blocks.append(Matrix([compare[i*10]+0.001 - self.model.mass_syms[i]]))
                D_other_blocks.append(Matrix([self.model.mass_syms[i]+0.001 - compare[i*10]]))

                # don't change COM
                D_other_blocks.append(Matrix([compare[i*10+1]+0.0001 - self.model.param_syms[i*10+1]]))
                D_other_blocks.append(Matrix([compare[i*10+2]+0.0001 - self.model.param_syms[i*10+2]]))
                D_other_blocks.append(Matrix([compare[i*10+3]+0.0001 - self.model.param_syms[i*10+3]]))

                D_other_blocks.append(Matrix([self.model.param_syms[i*10+1]+0.0001 - compare[i*10+1]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+2]+0.0001 - compare[i*10+2]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+3]+0.0001 - compare[i*10+3]]))

                # don't change inertia
                D_other_blocks.append(Matrix([compare[i*10+4]+0.0001 - self.model.param_syms[i*10+4]]))
                D_other_blocks.append(Matrix([compare[i*10+5]+0.0001 - self.model.param_syms[i*10+5]]))
                D_other_blocks.append(Matrix([compare[i*10+6]+0.0001 - self.model.param_syms[i*10+6]]))
                D_other_blocks.append(Matrix([compare[i*10+7]+0.0001 - self.model.param_syms[i*10+7]]))
                D_other_blocks.append(Matrix([compare[i*10+8]+0.0001 - self.model.param_syms[i*10+8]]))
                D_other_blocks.append(Matrix([compare[i*10+9]+0.0001 - self.model.param_syms[i*10+9]]))

                D_other_blocks.append(Matrix([self.model.param_syms[i*10+4]+0.0001 - compare[i*10+4]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+5]+0.0001 - compare[i*10+5]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+6]+0.0001 - compare[i*10+6]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+7]+0.0001 - compare[i*10+7]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+8]+0.0001 - compare[i*10+8]]))
                D_other_blocks.append(Matrix([self.model.param_syms[i*10+9]+0.0001 - compare[i*10+9]]))
            else:
                # all other links
                if self.opt['dontIdentifyMasses']:
                    D_other_blocks.append(Matrix([compare[i*10]+0.001 - self.model.mass_syms[i]]))
                    D_other_blocks.append(Matrix([self.model.mass_syms[i]+0.001 - compare[i*10]]))


        # constrain overall mass within bound
        if self.opt['limitOverallMass']:
            #use given overall mass else use overall mass from CAD
            if self.opt['limitMassVal']:
                robotmaxmass = self.limitMassVal
                robotmaxmass_ub = robotmaxmass * 1.05
                robotmaxmass_lb = robotmaxmass * 0.95
            else:
                robotmaxmass = robotmass_apriori
                # constrain with a bit of space around
                robotmaxmass_ub = robotmaxmass * 1.3
                robotmaxmass_lb = robotmaxmass * 0.7

            if self.opt['useAPriori']:
                D_other_blocks.append(Matrix([robotmaxmass_ub - (robotmass_apriori + sum(self.model.mass_syms))]))
                D_other_blocks.append(Matrix([(robotmass_apriori + sum(self.model.mass_syms)) - robotmaxmass_lb]))
            else:
                D_other_blocks.append(Matrix([robotmaxmass_ub - sum(self.model.mass_syms)])) #maximum mass
                D_other_blocks.append(Matrix([sum(self.model.mass_syms) - robotmaxmass_lb])) #minimum mass

        # constrain for each link separately
        if self.opt['limitMassValPerLink']:
            for i in range(self.model.N_LINKS):
                if not (self.noChange and linkConds[i] > self.noChangeThresh):
                    c = Matrix([self.limitMassValPerLink - (apriori[i*10] + self.model.mass_syms[i])])
                    D_other_blocks.append(c)
        elif self.opt['limitMassToApriori']:
            # constrain each mass to env of a priori value
            for i in range(self.model.N_LINKS):
                if not (self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']):
                    ub = Matrix([self.model.xStdModel[i*10]*(1+self.opt['limitMassAprioriBoundary']) -
                                (apriori[i*10] + self.model.mass_syms[i])])
                    lb = Matrix([(apriori[i*10] + self.model.mass_syms[i]) -
                                 self.model.xStdModel[i*10]*(1-self.opt['limitMassAprioriBoundary'])])
                    D_other_blocks.append(ub)
                    D_other_blocks.append(lb)

        if self.opt['restrictCOMtoHull']:
            link_cuboid_hulls = np.zeros((self.model.N_LINKS, 3, 2))
            for i in range(self.model.N_LINKS):
                if not (self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']):
                    link_cuboid_hulls[i] = np.array(self.urdfHelpers.getBoundingBox(self.model.urdf_file, i, self.opt['hullScaling']))
                    l = Matrix( self.model.param_syms[i*10+1:i*10+4])
                    m = self.model.mass_syms[i] + apriori[i*10]
                    link_cuboid_hull = link_cuboid_hulls[i]
                    for j in range(3):
                        ub = Matrix( [[  l[j]+apriori[i*10+1+j] - m*link_cuboid_hull[j][0] ]] )
                        lb = Matrix( [[ -l[j]-apriori[i*10+1+j] + m*link_cuboid_hull[j][1] ]] )
                        D_other_blocks.append( ub )
                        D_other_blocks.append( lb )

        """
        #friction constraints
        for i in range(dof):
            D_other_blocks.append( Matrix([rbt.rbtdef.fv[i]]) )
            D_other_blocks.append( Matrix([rbt.rbtdef.fc[i]]) )

        """
        D_blocks = D_inertia_blocks + D_other_blocks

        epsilon_safemargin = 1e-30
        #LMIs = list(map(LMI_PD, D_blocks))
        self.LMIs_marg = list(map(lambda lm: LMI_PSD(lm - epsilon_safemargin*eye(lm.shape[0])), D_blocks))


    def identifyStandardFeasibleParameters(self):
        # use SDP program to do OLS and constrain to physically feasible
        # space at the same time. Based on Sousa, 2014

        #build OLS matrix
        I = Identity
        delta = Matrix(self.model.param_syms)

        Q, R = la.qr(self.model.YBase)
        Q1 = Q[:, 0:self.model.num_base_params]
        #Q2 = Q[:, self.model.num_base_params:]
        rho1 = Q1.T.dot(self.model.tau)
        R1 = np.matrix(R[:self.model.num_base_params, :self.model.num_base_params])

        #projection matrix so that xBase = K*xStd
        #Sousa: K = Pb.T + Kd * Pd.T (Kd==self.model.linear_deps, self.P == [Pb Pd] ?)
        K = Matrix(self.model.Binv)

        if is_old_sympy:
            e_rho1 = Matrix(rho1).T - R1*K*delta
        else:
            e_rho1 = Matrix(rho1) - R1*K*delta

        rho2_norm_sqr = la.norm(self.model.tau - self.model.YBase.dot(self.model.xBase))**2
        u = Symbol('u')
        U_rho = BlockMatrix([[Matrix([u - rho2_norm_sqr]), e_rho1.T],
                             [e_rho1,       I(self.model.num_base_params)]])
        U_rho = U_rho.as_explicit()

        #add constraint LMIs
        lmis = [LMI_PSD(U_rho)] + self.LMIs_marg
        variables = [u] + list(delta)

        #solve SDP
        objective_func = u

        # try to use dsdp if a priori values are inconsistent (otherwise doesn't find solution)
        # it's probable still a bad solution
        if not self.paramHelpers.isPhysicalConsistent(self.model.xStdModel):
            print(Fore.RED+"a priori not consistent, but trying to use dsdp solver"+Fore.RESET)
            optimization.solve_sdp = optimization.cvxopt_dsdp5

        # start at CAD data, might increase convergence speed
        #if not self.opt['useAPriori'] and optimization.solve_sdp is optimization.dsdp5: prime = self.model.xStdModel
        #else: prime = None

        #solve SDP program (constrained OLS)
        solution = optimization.solve_sdp(objective_func, lmis, variables)

        u_star = solution[0,0]
        if u_star:
            print("found constrained solution with distance {} from OLS solution".format(u_star))
        delta_star = np.matrix(solution[1:])
        self.model.xStd = np.squeeze(np.asarray(delta_star))
        if self.opt['useAPriori']:
            self.model.xStd += self.model.xStdModel

    def findFeasibleStdFromStd(self, xStd):
        # correct any std solution to feasible std parameters
        if self.opt['useAPriori']:
            xStd -= self.model.xStdModel

        delta = Matrix(self.model.param_syms)
        I = Identity

        # correct a std solution to be feasible
        Pd = self.Pp[:, self.model.num_base_params:]
        delta_d = (Pd.T*delta)
        n_delta_d = len(delta_d)

        u = Symbol('u')
        U_delta = BlockMatrix([[Matrix([u]),       (xStd - delta).T],
                               [xStd - delta,    I(self.model.num_params)]])
        U_delta = U_delta.as_explicit()
        lmis = [LMI_PSD(U_delta)] + self.LMIs_marg
        variables = [u] + list(delta) # + list(delta_d)
        objective_func = u
        solution = optimization.solve_sdp(objective_func, lmis, variables)

        u_star = solution[0,0]
        if u_star:
            print("found constrained solution with distance {} from OLS solution".format(u_star))
        delta_star = np.matrix(solution[1:])
        xStd = np.squeeze(np.asarray(delta_star))

        if self.opt['useAPriori']:
            xStd += self.model.xStdModel
        return xStd

    def estimateParameters(self):
        print("doing identification on {} samples".format(self.data.num_used_samples)),

        self.model.computeRegressors(self.data)
        self.tauMeasured = self.model.tauMeasured

        if self.opt['useEssentialParams']:
            self.identifyBaseParameters()
            self.findBaseEssentialParameters()
            if self.opt['useAPriori']:
                self.getBaseParamsFromParamError()
            self.findStdFromBaseEssParameters()
            self.identifyStandardEssentialParameters()
        else:
            if self.opt['estimateWith'] in ['base', 'std']:
                self.identifyBaseParameters()
                if self.opt['useConsistencyConstraints']:
                    self.initSDP_LMIs()
                    self.identifyStandardFeasibleParameters()
                else:
                    self.findStdFromBaseParameters()
                if self.opt['useAPriori']:
                    self.getBaseParamsFromParamError()

            elif self.opt['estimateWith'] is 'std_direct':
                self.identifyStandardParameters()

    def plot(self):
        """Display some torque plots."""

        rel_time = self.model.T-self.model.T[0]
        if self.validation_file:
            rel_vtime = self.Tv-self.Tv[0]
        #TODO: allow plotting in subplots per joint, include raw values then
        datasets = [
            {'data': [self.tauMeasured], 'time': rel_time, 'title': 'Measured Torques',
                      'unified_scaling': True},
            {'data': [self.tauEstimated], 'time': rel_time, 'title': 'Estimated Torques',
                      'unified_scaling': True},
            {'data': [self.tauMeasured-self.tauEstimated], 'time': rel_time, 'title': 'Estimation Error',
                      'unified_scaling': False},
            {'data': [self.data.samples['positions'][0:self.model.sample_end:self.opt['skip_samples']+1]],
                      'time': rel_time, 'title': 'Positions'},
            {'data': [self.data.samples['velocities'][0:self.model.sample_end:self.opt['skip_samples']+1]],
                      'time': rel_time, 'title': 'Velocities'},
            {'data': [self.data.samples['accelerations'][0:self.model.sample_end:self.opt['skip_samples']+1]],
                      'time': rel_time, 'title': 'Accelerations'},
        ]

        if self.data.samples.has_key('positions_raw'):
            datasets[3]['data'].append(self.data.samples['positions_raw'][0:self.model.sample_end:self.opt['skip_samples']+1])
        if self.data.samples.has_key('velocities_raw'):
            datasets[4]['data'].append(self.data.samples['velocities_raw'][0:self.model.sample_end:self.opt['skip_samples']+1])

        if self.validation_file:
            datasets.append(
                {'data': [self.tauEstimatedValidation-self.tauMeasuredValidation],
                 'time': rel_vtime, 'title': 'Validation Error'},
                #([self.tauEstimatedValidation], rel_vtime, 'Validation Estimation'),
            )


        if self.opt['outputModule'] is 'matplotlib':
            from output import OutputMatplotlib
            output = OutputMatplotlib(datasets, self.model.jointNames)
            output.render()
        elif self.opt['outputModule'] is 'html':
            from output import OutputHTML
            output = OutputHTML(datasets, self.model.jointNames)
            output.render()
            #output.runServer()

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

    idf = Identification(args.model, args.model_real, args.measurements, args.regressor, args.validation)

    if idf.opt['selectBlocksFromMeasurements']:
        old_essential_option = idf.opt['useEssentialParams']
        idf.opt['useEssentialParams'] = 0

        old_feasible_option = idf.opt['useConsistencyConstraints']
        idf.opt['useConsistencyConstraints'] = 0
        # loop over input blocks and select good ones
        while 1:
            idf.estimateParameters()
            idf.data.getBlockStats(idf.model)
            idf.estimateRegressorTorques()
            OutputConsole.render(idf, summary_only=True)

            if idf.data.hasMoreSamples():
                idf.data.getNextSampleBlock()
            else:
                break

        idf.data.selectBlocks()
        idf.data.assembleSelectedBlocks()
        idf.opt['useEssentialParams'] = old_essential_option
        idf.opt['useConsistencyConstraints'] = old_feasible_option

    print("estimating output parameters...")
    idf.estimateParameters()
    idf.estimateRegressorTorques()

    if args.model_output:
        if idf.paramHelpers.isPhysicalConsistent(idf.model.xStd):
            print("can't create urdf file with estimated parameters since they are not physical consistent.")
        else:
            idf.urdfHelpers.replaceParamsInURDF(input_urdf=args.model, output_urdf=args.model_output, \
                                        new_params=idf.model.xStd, link_names=idf.model.link_names)

    if args.explain: OutputConsole.render(idf)
    if args.validation: idf.estimateValidationTorques()
    if args.plot: idf.plot()
    if idf.opt['showMemUsage']: idf.printMemUsage()

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
