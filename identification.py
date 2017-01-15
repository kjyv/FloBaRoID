#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import input
from builtins import zip
from builtins import range
from builtins import object
import sys

# math
import numpy as np
import numpy.linalg as la
import scipy
import scipy.linalg as sla
import scipy.stats as stats

import sympy
from sympy import Symbol, solve, Eq, Matrix, BlockMatrix, Identity, eye, zeros
from distutils.version import LooseVersion
is_old_sympy = LooseVersion(sympy.__version__) < LooseVersion('0.7.4')

# plotting
import matplotlib
import matplotlib.pyplot as plt

# kinematics, dynamics and URDF reading
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

# submodules
from identification.model import Model
from identification.data import Data
from identification.output import OutputConsole
from identification import helpers

from colorama import Fore, Back, Style

from identification import optimization
from identification.optimization import LMI_PSD, LMI_PD

from IPython import embed
np.core.arrayprint._line_width = 160

# Referenced papers:
# Gautier, 1991: Numerical Calculation of the base Inertial Parameters of Robots
# Pham, 1991: Essential Parameters of Robots
# Zak et. al, 1994: Application of the Weighted Least Squares Parameter Estimation Method to the
# Robot Calibration
# Venture et al, 2009: A numerical method for choosing motions with optimal excitation properties
# for identification of biped dynamics
# Jubien, 2014: Dynamic identification of the Kuka LWR robot using motor torques and joint torque
# sensors data
# Sousa, 2014: Physical feasibility of robot base inertial parameter identification: A linear matrix
# inequality approach


class Identification(object):
    def __init__(self, opt, urdf_file, urdf_file_real, measurements_files, regressor_file, validation_file):
        self.opt = opt

        ## some additional options (experiments)

        # orthogonalize basis matrix (some SDP estimations seem to work more stable that way)
        self.opt['orthogonalizeBasis'] = 1

        # in order ot get regressor and base equations, use basis projection matrix or use
        # permutation from QR directly (Gautier/Sousa method)
        self.opt['useBasisProjection'] = 1

        # use RBDL for simulation
        self.opt['useRBDL'] = 0

        #### end additional config flags


        # load model description and initialize
        self.model = Model(self.opt, urdf_file, regressor_file)

        # load measurements
        self.data = Data(self.opt)
        self.data.init_from_files(measurements_files)

        self.paramHelpers = helpers.ParamHelpers(self.model.num_inertial_params)
        self.urdfHelpers = helpers.URDFHelpers(self.paramHelpers, self.model.linkNames, self.opt)

        self.tauEstimated = list()
        self.res_error = 100
        self.urdf_file_real = urdf_file_real
        self.validation_file = validation_file


    def estimateRegressorTorques(self, estimateWith=None):
        """ get torque estimations using regressors, prepare for plotting """

        if not estimateWith:
            # use global parameter choice if none is given specifically
            estimateWith = self.opt['estimateWith']
        # estimate torques with idyntree regressor and different params
        if estimateWith == 'urdf':
            tauEst = np.dot(self.model.YStd, self.model.xStdModel)
        elif estimateWith == 'base_essential':
            tauEst = np.dot(self.model.YBase, self.xBase_essential)
        elif estimateWith == 'base':
            tauEst = np.dot(self.model.YBase, self.model.xBase)
        elif estimateWith in ['std', 'std_direct']:
            tauEst = np.dot(self.model.YStd, self.model.xStd)
        else:
            print("unknown type of parameters: {}".format(self.opt['estimateWith']))

        if self.opt['floatingBase']:
            fb = 6
            tauEst -= self.model.contactForcesSum
        else:
            fb = 0

        self.tauEstimated = np.reshape(tauEst, (self.data.num_used_samples, self.model.N_DOFS + fb))

        #if self.opt['floatingBase']:
        #    self.tauEstimated[:, 6:] -= self.model.contactForcesSum_2dim[:, 6:]

        if self.opt['showErrorHistogram'] == 1:
            error = np.mean(self.model.tauMeasured - self.tauEstimated, axis=1)
            plt.hist(error, 50)
            plt.title("error histogram")
            plt.draw()
            plt.show()
            # don't show again if we come here later
            self.opt['showErrorHistogram'] = 2

        # reshape torques into one column per DOF for plotting (NUM_SAMPLES*N_DOFSx1) -> (NUM_SAMPLESxN_DOFS)
        if estimateWith == 'urdf':
            self.tauAPriori = self.tauEstimated


    def estimateValidationTorques(self):
        """ calculate torques of trajectory from validation measurements and identified params """

        # TODO: get identified params directly into idyntree (new KinDynComputations class does not
        # have inverse dynamics yet, so we have to go over a new urdf file for now)
        import os

        v_data = np.load(self.validation_file)
        dynComp = iDynTree.DynamicsComputations()

        self.urdfHelpers.replaceParamsInURDF(input_urdf=self.model.urdf_file,
                                             output_urdf=self.model.urdf_file + '.tmp',
                                             new_params=self.model.xStd, link_names=self.model.linkNames)
        dynComp.loadRobotModelFromFile(self.model.urdf_file + '.tmp')
        os.remove(self.model.urdf_file + '.tmp')

        old_skip = self.opt['skipSamples']
        self.opt['skipSamples'] = 8

        self.tauEstimatedValidation = None
        for m_idx in range(0, v_data['positions'].shape[0], self.opt['skipSamples'] + 1):
            torques = self.model.simulateDynamicsIDynTree(v_data, m_idx, dynComp)

            if self.tauEstimatedValidation is None:
                self.tauEstimatedValidation = torques
            else:
                self.tauEstimatedValidation = np.vstack((self.tauEstimatedValidation, torques))

        if self.opt['skipSamples'] > 0:
            self.tauMeasuredValidation = v_data['torques'][::self.opt['skipSamples'] + 1]
            self.Tv = v_data['times'][::self.opt['skipSamples'] + 1]
        else:
            self.tauMeasuredValidation = v_data['torques']
            self.Tv = v_data['times']

        # add simulated base forces also to measurements
        if self.opt['floatingBase']:
            self.tauMeasuredValidation = np.concatenate((self.tauEstimatedValidation[:, :6], self.tauMeasuredValidation), axis=1)

        #TODO: add contact forces to estimation

        self.opt['skipSamples'] = old_skip

        self.val_error = sla.norm(self.tauEstimatedValidation - self.tauMeasuredValidation) \
                            *100/sla.norm(self.tauMeasuredValidation)
        print("Validation error (std params): {}%".format(self.val_error))


    def getBaseParamsFromParamError(self):
        self.model.xBase += self.model.xBaseModel   # both param vecs link relative linearized

        if self.opt['useEssentialParams']:
            self.xBase_essential[self.baseEssentialIdx] += self.model.xBaseModel[self.baseEssentialIdx]


    def findStdFromBaseParameters(self):
        '''find std parameter from base parameters (simply projection method)'''
        # Note: assumes that xBase is still in error form if using a priori
        # i.e. don't call after getBaseParamsFromParamError

        # project back to standard parameters
        if self.opt['useBasisProjection']:
            self.model.xStd = self.model.B.dot(self.model.xBase)
        else:
            self.model.xStd = la.pinv(self.model.K).dot(self.model.xBase)

        # get estimated parameters from estimated error (add a priori knowledge)
        if self.opt['useAPriori']:
            self.model.xStd += self.model.xStdModel

    def getStdDevForParams(self):
        if self.opt['useAPriori']:
            tauDiff = self.model.tauMeasured - self.tauEstimated
        else:
            tauDiff = self.tauEstimated

        if self.opt['floatingBase']: fb = 6
        else: fb = 0

        # get relative standard deviation of measurement and modeling error \sigma_{rho}^2
        r = self.data.num_used_samples*(self.model.N_DOFS+fb)
        rho = np.square(sla.norm(tauDiff))
        sigma_rho = rho/(r - self.model.num_base_params)

        # get standard deviation \sigma_{x} (of the estimated parameter vector x)
        C_xx = sigma_rho * (sla.inv(np.dot(self.model.YBase.T, self.model.YBase)))
        sigma_x = np.diag(C_xx)

        # get relative standard deviation
        p_sigma_x = np.sqrt(sigma_x)
        for i in range(0, p_sigma_x.size):
            if self.model.xBase[i] != 0:
                p_sigma_x[i] /= np.abs(self.model.xBase[i])

        return p_sigma_x

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
            base_idx = list(range(0, self.model.num_base_params))
            not_essential_idx = list()
            ratio = 0

            # get initial errors of estimation
            self.estimateRegressorTorques('base')

            if not self.opt['useAPriori']:
                tauDiff = self.model.tauMeasured - self.tauEstimated
            else:
                tauDiff = self.tauEstimated

            def error_func(tauDiff):
                #rho = tauDiff
                rho = np.mean(tauDiff, axis=1)
                #rho = np.square(la.norm(tauDiff, axis=1))
                return rho

            error_start = error_func(tauDiff)

            k2, p = stats.normaltest(error_start, axis=0)
            if self.opt['verbose']:
                if np.mean(p) > 0.05:
                    print("error is normal distributed")
                else:
                    print("error is not normal distributed (p={})".format(p))

            if not self.opt['useAPriori']:
                pham_percent_start = sla.norm(tauDiff) * 100 / sla.norm(self.tauEstimated)
            else:
                pham_percent_start = sla.norm(tauDiff) * 100 / sla.norm(self.model.tauMeasured)

            print("starting percentual error {}".format(pham_percent_start))

            #rho_start = np.square(sla.norm(tauDiff))
            p_sigma_x = 0

            has_run_once = 0
            # start removing non-essential parameters
            while 1:
                # get new torque estimation to calc error norm (new estimation with updated parameters)
                self.estimateRegressorTorques('base')

                prev_p_sigma_x = p_sigma_x
                p_sigma_x = self.getStdDevForParams()

                print("{} params|".format(self.model.num_base_params - b_c), end=' ')

                ratio = np.max(p_sigma_x) / np.min(p_sigma_x)
                print("min-max ratio of relative stddevs: {},".format(ratio), end=' ')

                print("cond(YBase):{},".format(la.cond(self.model.YBase)), end=' ')

                if not self.opt['useAPriori']:
                    tauDiff = self.model.tauMeasured - self.tauEstimated
                else:
                    tauDiff = self.tauEstimated
                pham_percent = sla.norm(tauDiff) * 100 / sla.norm(self.model.tauMeasured)
                error_increase_pham = pham_percent_start - pham_percent
                print("error delta {}".format(error_increase_pham))

                # while loop condition moved to here
                # TODO: consider to only stop when under ratio and
                # if error is to large at that point, advise to get more/better data
                if ratio < 30:
                    break
                if use_error_criterion and error_increase_pham > 3.5:
                    break

                if has_run_once and self.opt['showEssentialSteps']:
                    # put some values into global variable for output
                    self.baseNonEssentialIdx = not_essential_idx
                    self.baseEssentialIdx = [x for x in range(0, self.model.num_base_params) if x not in not_essential_idx]
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

                    print(base_idx, np.argmax(p_sigma_x))
                    print(self.baseNonEssentialIdx)
                    input("Press return...")
                else:
                    has_run_once = 1

                # cancel the parameter with largest deviation
                param_idx = np.argmax(p_sigma_x)
                # get its index among the base params (otherwise it doesnt take deletion into account)
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
            self.baseEssentialIdx = [x for x in range(0, self.model.num_base_params) if x not in not_essential_idx]
            self.num_essential_params = len(self.baseEssentialIdx)

            # leave previous base params and regressor unchanged
            self.xBase_essential = np.zeros_like(xBase_orig)
            self.xBase_essential[self.baseEssentialIdx] = self.prev_xBase
            self.model.YBase = YBase_orig
            self.model.xBase = xBase_orig

            print("Got {} essential parameters".format(self.num_essential_params))

        if self.opt['showTiming']:
            print("Getting base essential parameters took %.03f sec." % t.interval)


    def findStdFromBaseEssParameters(self):
        """ Find essential standard parameters from previously determined base essential parameters. """

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
            for i in range(0, self.model.base_deps.shape[0]):
                if i in self.baseEssentialIdx:
                    for s in self.model.base_deps[i].free_symbols:
                        idx = self.model.param_syms.index(s)
                        if idx not in dependents:
                            dependents.append(idx)

            #print self.stdEssentialIdx
            #print len(dependents)
            print(dependents)
            self.stdEssentialIdx = np.concatenate((self.stdEssentialIdx, dependents))

        #np.delete(self.stdEssentialIdx, to_delete, 0)

        # remove mass params if present
        if self.opt['dontIdentifyMasses']:
            ps = list(range(0, self.model.num_params, 10))
            self.stdEssentialIdx = np.fromiter((x for x in self.stdEssentialIdx if x not in ps), int)

        self.stdNonEssentialIdx = [x for x in range(0, self.model.num_params) if x not in self.stdEssentialIdx]

        # get \hat{x_e}, set zeros for non-essential params
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
                    p_start = idx // 10 * 10
                    if idx % 10 in [1,2,3]:   #com value
                        v = np.mean(self.model.xStdModel[p_start + 1:p_start + 4]) * 0.1
                    elif idx % 10 in [4,5,6,7,8,9]:  #inertia value
                        inertia_range = np.array([4,5,6,7,8,9])+p_start
                        v = np.mean(self.model.xStdModel[np.where(self.model.xStdModel[inertia_range] != 0)[0]+p_start+4]) * 0.1
                    if v == 0:
                        v = 0.1
                    self.xStdEssential[idx] = v
                    #print idx, idx % 10, v
                idx += 1
                if idx > self.model.num_inertial_params:
                    break

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


    def identifyBaseParameters(self, YBase=None, tau=None, id_only=False):
        """use previously computed regressors and identify base parameter vector using ordinary or
           weighted least squares."""

        if YBase is None:
            YBase = self.model.YBase
        if tau is None:
            tau = self.model.tau

        if self.opt['useBasisProjection']:
            self.model.xBaseModel = np.dot(self.model.xStdModel, self.model.B)
        else:
            self.model.xBaseModel = np.dot(self.model.Pb.T, self.model.xStdModel)

        # note: using pinv is only ok if low condition number, otherwise numerical issues can happen
        # should always try to avoid inversion if possible

        # invert equation to get parameter vector from measurements and model + system state values
        self.model.YBaseInv = la.pinv(YBase)

        if self.opt['floatingBase']:
            self.model.xBase = self.model.YBaseInv.dot(tau.T) + self.model.YBaseInv.dot(self.model.contactForcesSum)
        else:
            self.model.xBase = self.model.YBaseInv.dot(tau.T)

        """
        # ordinary least squares with numpy method (might be better in noisy situations)
        if self.opt['floatingBase']:
            self.model.xBase = la.lstsq(YBase, tau)[0] + self.model.YBaseInv.dot(self.model.contactForcesSum)
        else:
            self.model.xBase = la.lstsq(YBase, tau)[0]
        """

        # damped least squares
        #from scipy.sparse.linalg import lsqr
        #self.model.xBase = lsqr(YBase, tau, damp=10)[0]

        # stop here if called recursively
        if id_only:
            return

        if self.opt['showBaseParams']:
            # get estimation once with previous ordinary LS solution parameters
            self.estimateRegressorTorques('base')
            self.p_sigma_x = self.getStdDevForParams()

        if self.opt['useWLS']:
            """
            additionally do weighted least squares IDIM-WLS, cf. Zak, 1994, Gautier, 1997 and Khalil, 2007.
            adds weighting with relative standard dev of estimation error on OLS base regressor and params.
            (includes reducing effect of different units of parameters)
            """

            # get estimation once with previous ordinary LS solution parameters
            self.estimateRegressorTorques('base')
            self.p_sigma_x = self.getStdDevForParams()

            if self.opt['floatingBase']: fb = 6
            else: fb = 0
            r = self.data.num_used_samples*(self.model.N_DOFS+fb)

            '''
            if self.opt['useAPriori']:
                tauDiff = self.model.tauMeasured - self.tauEstimated
            else:
                tauDiff = self.tauEstimated

            # get standard deviation of measurement and modeling error \sigma_{rho}^2
            # for each joint subsystem (rho is assumed zero mean independent noise)
            self.sigma_rho = np.square(sla.norm(tauDiff)) / (r-self.model.num_base_params)
            '''
            # repeat stddev values for each measurement block (n_joints * num_samples)
            # along the diagonal of G
            # G = np.diag(np.repeat(1/self.sigma_rho, self.num_used_samples))
            #G = scipy.sparse.spdiags(np.tile(1/self.sigma_rho, self.num_used_samples), 0,
            #        self.N_DOFS*self.num_used_samples, self.N_DOFS*self.num_used_samples)
            #G = scipy.sparse.spdiags(np.repeat(1/np.sqrt(self.sigma_rho), self.data.num_used_samples), 0, r, r)
            G = scipy.sparse.spdiags(np.repeat(1/self.p_sigma_x, self.data.num_used_samples), 0, r, r)

            # weigh Y and tau with deviations
            self.model.YBase = G.dot(self.model.YBase)
            if self.opt['useAPriori']:
                #if identifying parameter error, weigh full tau
                self.model.tau = G.dot(self.model.torques_stack) - G.dot(self.model.torquesAP_stack)
            else:
                self.model.tau = G.dot(self.model.tau)
            if self.opt['verbose']:
                print("Condition number of WLS YBase: {}".format(la.cond(self.model.YBase)))

            # get identified values using weighted matrices without weighing them again
            self.identifyBaseParameters(self.model.YBase, tau, id_only=True)


    def identifyStandardParametersDirect(self):
        """Identify standard parameters directly with non-singular standard regressor."""

        with helpers.Timer() as t:
            U, s, VH = la.svd(self.model.YStd, full_matrices=False)
            nb = self.model.num_base_params

            # identify standard parameters directly
            V_1 = VH.T[:, 0:nb]
            U_1 = U[:, 0:nb]
            s_1 = np.diag(s[0:nb])
            s_1_inv = la.inv(s_1)
            W_st_pinv = V_1.dot(s_1_inv).dot(U_1.T)
            W_st = la.pinv(W_st_pinv)
            self.YStd_nonsing = W_st

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
            Yst_e = self.model.YStd.dot(np.diag(self.xStdEssential))   # = W_st^e
            Ue, se, VHe = sla.svd(Yst_e, full_matrices=False)
            ne = self.num_essential_params  # nr. of essential params among base params
            V_1e = VHe.T[:, 0:ne]
            U_1e = Ue[:, 0:ne]
            s_1e_inv = sla.inv(np.diag(se[0:ne]))
            W_st_e_pinv = np.diag(self.xStdEssential).dot(V_1e.dot(s_1e_inv).dot(U_1e.T))
            #W_st_e = la.pinv(W_st_e_pinv)

            x_tmp = W_st_e_pinv.dot(self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_tmp
            else:
                self.model.xStd = x_tmp

        if self.opt['showTiming']:
            print("Identifying %s std essential parameters took %.03f sec." % (len(self.stdEssentialIdx), t.interval))


    def initSDP_LMIs(self):
        ''' initialize LMI matrices to set physical consistency constraints for SDP solver
            based on Sousa, 2014 and corresponding code (https://github.com/cdsousa/IROS2013-Feas-Ident-WAM7)
        '''

        with helpers.Timer() as t:
            if self.opt['verbose']:
                print("Initializing LMIs...")

            def skew(v):
                return Matrix([ [     0, -v[2],  v[1] ],
                                [  v[2],     0, -v[0] ],
                                [ -v[1],  v[0],   0 ] ])
            I = Identity
            S = skew

            #compare values relative to apriori CAD parameters
            compare = self.model.xStdModel

            # create LMI matrices (symbols) for each link
            # so that mass is positive, inertia matrix is positive definite
            # (each matrix is later on used to be either >0 or >=0)
            D_inertia_blocks = []
            for i in range(0, self.model.N_LINKS):
                m = self.model.mass_syms[i]
                l = Matrix([ self.model.param_syms[i*10+1],
                             self.model.param_syms[i*10+2],
                             self.model.param_syms[i*10+3] ] )
                L = Matrix([ [self.model.param_syms[i*10+4+0],
                              self.model.param_syms[i*10+4+1],
                              self.model.param_syms[i*10+4+2]],
                             [self.model.param_syms[i*10+4+1],
                              self.model.param_syms[i*10+4+3],
                              self.model.param_syms[i*10+4+4]],
                             [self.model.param_syms[i*10+4+2],
                              self.model.param_syms[i*10+4+4],
                              self.model.param_syms[i*10+4+5]]
                           ])

                Di = BlockMatrix([[L,    S(l).T],
                                  [S(l), I(3)*m]])
                D_inertia_blocks.append(Di.as_explicit().as_mutable())

            D_other_blocks = []

            params_to_skip = []

            if self.opt['limitNonIdentifiable']:
                params_to_skip.extend(self.model.non_identifiable)

            self.constr_per_param = {}
            for i in range(self.model.num_params):
                self.constr_per_param[i] = []

            linkConds = self.model.getSubregressorsConditionNumbers()
            robotmass_apriori = 0
            for i in range(0, self.model.N_LINKS):
                robotmass_apriori+= self.model.xStdModel[i*10]  #count a priori link masses

                #for links that have too high condition number, don't change params
                if self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']:
                    print(Fore.YELLOW + 'skipping identification of link {} ({})!'.format(i, self.model.linkNames[i]) + Fore.RESET)
                    # don't change mass
                    params_to_skip.append(i*10)

                    # don't change COM
                    params_to_skip.append(i*10+1)
                    params_to_skip.append(i*10+2)
                    params_to_skip.append(i*10+3)

                    # don't change inertia
                    params_to_skip.append(i*10+4)
                    params_to_skip.append(i*10+5)
                    params_to_skip.append(i*10+6)
                    params_to_skip.append(i*10+7)
                    params_to_skip.append(i*10+8)
                    params_to_skip.append(i*10+9)
                else:
                    # constraints for all other links
                    if self.opt['dontIdentifyMasses']:
                        params_to_skip.append(i*10)

            # manual fixed params
            for p in self.opt['dontChangeParams']:
                params_to_skip.append(p)

            # create actual don't-change constraints
            for p in set(params_to_skip):
                D_other_blocks.append(Matrix([compare[p] - self.model.param_syms[p]]))
                D_other_blocks.append(Matrix([self.model.param_syms[p] - compare[p]]))
                self.constr_per_param[p].append('cad')

            # constrain overall mass within bounds
            if self.opt['limitOverallMass']:
                #use given overall mass else use overall mass from CAD
                if self.opt['limitMassVal']:
                    robotmaxmass = self.opt['limitMassVal']
                    robotmaxmass_ub = robotmaxmass * 1.01
                    robotmaxmass_lb = robotmaxmass * 0.99
                else:
                    robotmaxmass = robotmass_apriori
                    # constrain with a bit of space around
                    robotmaxmass_ub = robotmaxmass * 1.3
                    robotmaxmass_lb = robotmaxmass * 0.7

                D_other_blocks.append(Matrix([robotmaxmass_ub - sum(self.model.mass_syms)])) #maximum mass
                D_other_blocks.append(Matrix([sum(self.model.mass_syms) - robotmaxmass_lb])) #minimum mass

            # constrain for each link separately
            if self.opt['limitMassValPerLink']:
                for i in range(self.model.N_LINKS):
                    if not (self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']):
                        c = Matrix([self.opt['limitMassValPerLink'] - self.model.mass_syms[i]])
                        D_other_blocks.append(c)
                        self.constr_per_param[i*10].append('mF')
            elif self.opt['limitMassToApriori']:
                # constrain each mass to env of a priori value
                for i in range(self.model.N_LINKS):
                    if not (self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']):
                        ub = Matrix([compare[i*10]*(1+self.opt['limitMassAprioriBoundary']) -
                                    self.model.mass_syms[i]])
                        lb = Matrix([self.model.mass_syms[i] -
                                     compare[i*10]*(1-self.opt['limitMassAprioriBoundary'])])
                        D_other_blocks.append(ub)
                        D_other_blocks.append(lb)
                        self.constr_per_param[i*10].append('mA')

            if self.opt['restrictCOMtoHull']:
                link_cuboid_hulls = np.zeros((self.model.N_LINKS, 3, 2))
                for i in range(self.model.N_LINKS):
                    if not (self.opt['noChange'] and linkConds[i] > self.opt['noChangeThresh']):
                        link_cuboid_hulls[i] = np.array(self.urdfHelpers.getBoundingBox(self.model.urdf_file, i, self.opt['hullScaling']))
                        #print link_cuboid_hulls[i]*self.model.xStdModel[i*10]
                        l = Matrix( self.model.param_syms[i*10+1:i*10+4])
                        m = self.model.mass_syms[i]
                        link_cuboid_hull = link_cuboid_hulls[i]
                        for j in range(3):
                            ub = Matrix( [[  l[j] - m*link_cuboid_hull[j][0] ]] )
                            lb = Matrix( [[ -l[j] + m*link_cuboid_hull[j][1] ]] )
                            D_other_blocks.append( ub )
                            D_other_blocks.append( lb )
                            self.constr_per_param[i*10+1+j].append('hull')

            # symmetry constraints
            if self.opt['useSymmetryConstraints'] and self.opt['symmetryConstraints']:
                for (a, b, sign) in self.opt['symmetryConstraints']:
                    D_other_blocks.append(Matrix([self.model.param_syms[a] - sign*self.model.param_syms[b]]))
                    D_other_blocks.append(Matrix([sign*self.model.param_syms[b] - self.model.param_syms[a]]))
                    self.constr_per_param[a].append('sym')
                    self.constr_per_param[b].append('sym')

            if self.opt['identifyFriction']:
                # friction constraints
                # (only makes sense when no offsets on torque measurements, otherwise can be
                # negative)
                for i in range(self.model.N_DOFS):
                    #Fc > 0
                    D_other_blocks.append( Matrix([self.model.param_syms[self.model.num_inertial_params+i]]) )
                    #Fv > 0
                    D_other_blocks.append( Matrix([self.model.param_syms[self.model.num_inertial_params+self.model.N_DOFS+i]]) )
                    D_other_blocks.append( Matrix([self.model.param_syms[self.model.num_inertial_params+self.model.N_DOFS*2+i]]) )
                    self.constr_per_param[self.model.num_inertial_params+i].append('>0')
                    self.constr_per_param[self.model.num_inertial_params+self.model.N_DOFS+i].append('>0')
                    self.constr_per_param[self.model.num_inertial_params+self.model.N_DOFS*2+i].append('>0')

            self.D_blocks = D_inertia_blocks + D_other_blocks

            epsilon_safemargin = 1e-6
            #LMIs = list(map(LMI_PD, D_blocks))
            self.LMIs_marg = list([LMI_PSD(lm - epsilon_safemargin*eye(lm.shape[0])) for lm in self.D_blocks])

        if self.opt['showTiming']:
            print("Initializing LMIs took %.03f sec." % (t.interval))


    def identifyFeasibleStandardParameters(self):
        ''' use SDP optimzation to solve constrained OLS to find globally optimal physically
            feasible std parameters (not necessarily unique). Based on code from Sousa, 2014
        '''
        with helpers.Timer() as t:
            #if self.opt['useAPriori']:
            #    print("Please disable using a priori parameters when using constrained optimization.")
            #    sys.exit(1)

            if self.opt['verbose']:
                print("Preparing SDP...")

            #build OLS matrix
            I = Identity
            delta = Matrix(self.model.param_syms)

            Q, R = la.qr(self.model.YBase)
            Q1 = Q[:, 0:self.model.num_base_params]
            #Q2 = Q[:, self.model.num_base_params:]
            rho1 = Q1.T.dot(self.model.torques_stack)
            R1 = np.matrix(R[:self.model.num_base_params, :self.model.num_base_params])

            # get projection matrix so that xBase = K*xStd
            if self.opt['useBasisProjection']:
                K = Matrix(self.model.Binv)
            else:
                #Sousa: K = Pb.T + Kd * Pd.T (Kd==self.model.linear_deps, [Pb Pd] == self.model.Pp)
                #Pb = Matrix(self.model.Pb) #.applyfunc(lambda x: x.nsimplify())
                #Pd = Matrix(self.model.Pd) #.applyfunc(lambda x: x.nsimplify())
                K = Matrix(self.model.K) #(Pb.T + Kd * Pd.T)

            # OLS: minimize ||tau - Y*x_base||^2 (simplify)=> minimize ||rho1.T - R1*K*delta||^2
            # sub contact forces
            if self.opt['floatingBase']:
                contactForces = Q.T.dot(self.model.contactForcesSum)
            else:
                contactForces = zeros(self.model.num_base_params, 1)

            # minimize estimation error of to-be-found parameters delta
            # (regressor dot std variables projected to base - contatcs should be close to measured torques)
            if is_old_sympy:
                e_rho1 = Matrix(rho1).T - (R1*K*delta - contactForces)
            else:
                e_rho1 = Matrix(rho1) - (R1*K*delta - contactForces)

            rho2_norm_sqr = la.norm(self.model.torques_stack - self.model.YBase.dot(self.model.xBase))**2
            u = Symbol('u')
            U_rho = BlockMatrix([[Matrix([u - rho2_norm_sqr]), e_rho1.T],
                                 [e_rho1, I(self.model.num_base_params)]])
            U_rho = U_rho.as_explicit()

            if self.opt['verbose']:
                print("Add constraint LMIs")
            lmis = [LMI_PSD(U_rho)] + self.LMIs_marg
            variables = [u] + list(delta)

            #solve SDP
            objective_func = u

            if self.opt['verbose']:
                print("Solving constrained OLS as SDP")

            # start at CAD data, might increase convergence speed (atm only works with dsdp5,
            # otherwise returns primal as solution when failing)
            prime = self.model.xStdModel
            solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime)

            #try again with wider bounds and dsdp5 cmd line
            if state is not 'optimal' or not self.paramHelpers.isPhysicalConsistent(np.squeeze(np.asarray(solution[1:]))):
                print("Trying again with dsdp5 solver")
                optimization.solve_sdp = optimization.dsdp5
                solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime, wide_bounds=True)
                optimization.solve_sdp = optimization.cvxopt_conelp

            u_star = solution[0,0]
            if u_star:
                print("found std solution with distance {} from OLS solution".format(u_star))
            delta_star = np.matrix(solution[1:])
            self.model.xStd = np.squeeze(np.asarray(delta_star))

        if self.opt['showTiming']:
            print("Constrained SDP optimization took %.03f sec." % (t.interval))


    def identifyFeasibleStandardParametersDirect(self):
        ''' use SDP optimzation to solve constrained OLS to find globally optimal physically
            feasible std parameters. Based on code from Sousa, 2014, using direct regressor from Gautier, 2013
        '''
        with helpers.Timer() as t:
            #if self.opt['useAPriori']:
            #    print("Please disable using a priori parameters when using constrained optimization.")
            #    sys.exit(1)

            if self.opt['verbose']:
                print("Preparing SDP...")

            #build OLS matrix
            I = Identity
            delta = Matrix(self.model.param_syms)

            Q, R = la.qr(self.YStd_nonsing)
            Q1 = Q[:, 0:self.model.num_params]
            #Q2 = Q[:, self.model.num_base_params:]
            rho1 = Q1.T.dot(self.model.torques_stack)
            R1 = np.matrix(R[:self.model.num_params, :self.model.num_params])

            # OLS: minimize ||tau - Y*x_base||^2 (simplify)=> minimize ||rho1.T - R1*K*delta||^2
            # sub contact forces
            if self.opt['floatingBase']:
                contactForces = Q.T.dot(self.model.contactForcesSum)
            else:
                contactForces = zeros(self.model.num_params, 1)

            import time
            print("Step 1...", time.ctime())

            # minimize estimation error of to-be-found parameters delta
            # (regressor dot std variables projected to base - contatcs should be close to measured torques)
            if is_old_sympy:
                e_rho1 = Matrix(rho1).T - (R1*delta - contactForces)
            else:
                e_rho1 = Matrix(rho1) - (R1*delta - contactForces)

            print("Step 2...", time.ctime())

            # calc estimation error of previous OLS parameter solution
            rho2_norm_sqr = la.norm(self.model.torques_stack - self.model.YBase.dot(self.model.xBase))**2
            print("rho2_norm_sqr: ", rho2_norm_sqr)

            # (this is the slow part when matrices get bigger, BlockMatrix or as_explicit?)
            u = Symbol('u')
            U_rho = BlockMatrix([[Matrix([u - rho2_norm_sqr]), e_rho1.T],
                                 [e_rho1, I(self.model.num_params)]])
            print("Step 3...", time.ctime())
            U_rho = U_rho.as_explicit()
            print("Step 4...", time.ctime())

            if self.opt['verbose']:
                print("Add constraint LMIs")
            lmis = [LMI_PSD(U_rho)] + self.LMIs_marg
            variables = [u] + list(delta)

            #solve SDP
            objective_func = u

            if self.opt['verbose']:
                print("Solving constrained OLS as SDP")

            # start at CAD data, might increase convergence speed (atm only works with dsdp5,
            # otherwise returns primal as solution when failing)
            prime = self.model.xStdModel
            solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime)

            #try again with wider bounds and dsdp5 cmd line
            if state is not 'optimal':
                print("Trying again with dsdp5 solver")
                optimization.solve_sdp = optimization.dsdp5
                solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime, wide_bounds=True)
                optimization.solve_sdp = optimization.cvxopt_conelp

            u_star = solution[0,0]
            if u_star:
                print("found std solution with {} error increase from OLS solution".format(u_star))
            delta_star = np.matrix(solution[1:])
            self.model.xStd = np.squeeze(np.asarray(delta_star))

        if self.opt['showTiming']:
            print("Constrained SDP optimization took %.03f sec." % (t.interval))


    def identifyFeasibleBaseParameters(self):
        ''' use SDP optimization to solve OLS to find physically feasible base parameters (i.e. for
            which a consistent std solution exists), based on code from github.com/cdsousa/wam7_dyn_ident
        '''
        with helpers.Timer() as t:
            if self.opt['verbose']:
                print("Preparing SDP...")

            # build OLS matrix
            I = Identity
            def mrepl(m,repl):
                return m.applyfunc(lambda x: x.xreplace(repl))

            # base and standard parameter symbols
            delta = Matrix(self.model.param_syms)
            beta_symbs = self.model.base_syms

            # permutation of std to base columns projection
            # (simplify to reduce 1.0 to 1 etc., important for replacement)
            Pb = Matrix(self.model.Pb).applyfunc(lambda x: x.nsimplify())
            # permutation of std to non-identifiable columns (dependents)
            Pd = Matrix(self.model.Pd).applyfunc(lambda x: x.nsimplify())

            # projection matrix from independents to dependents
            #Kd = Matrix(self.model.linear_deps)
            #K = Matrix(self.model.K).applyfunc(lambda x: x.nsimplify()) #(Pb.T + Kd * Pd.T)

            # equations for base parameters expressed in independent std param symbols
            #beta = K * delta
            beta = Matrix(self.model.base_deps).applyfunc(lambda x: x.nsimplify())

            ## std vars that occur in base params (as many as base params, so only the single ones or chosen as independent ones)

            '''
            if self.opt['useBasisProjection']:
                # determined through base matrix, which included other variables too
                # (find first variable in eq, chosen as independent here)
                delta_b_syms = []
                for i in range(self.model.base_deps.shape[0]):
                    for s in self.model.base_deps[i].free_symbols:
                        if s not in delta_b_syms:
                            delta_b_syms.append(s)
                            break
                delta_b = Matrix(delta_b_syms)
            else:
                # determined through permutation matrix from QR (not correct if base matrix is orthogonalized afterwards)
            '''
            delta_b = Pb.T*delta

            ## std variables that are dependent, i.e. their value is a combination of independent columns
            '''
            if self.opt['useBasisProjection']:
                #determined from base eqns
                delta_not_d = self.model.base_deps[0].free_symbols
                for e in self.model.base_deps:
                    delta_not_d = delta_not_d.union(e.free_symbols)
                delta_d = []
                for s in delta:
                    if s not in delta_not_d:
                        delta_d.append(s)
                delta_d = Matrix(delta_d)
            else:
                # determined through permutation matrix from QR (not correct if base matrix is orthogonalized afterwards)
            '''
            delta_d = Pd.T*delta

            # rewrite LMIs for base params

            if self.opt['useBasisProjection']:
                # (Sousa code is assuming that delta_b for each eq has factor 1.0 in equations beta.
                # this is true if using Gautier dependency matrix, otherwise
                # correct is to properly transpose eqn base_n = a1*x1 + a2*x2 + ... +an*xn to
                # 1*xi = a1*x1/ai + a2*x2/ai + ... + an*xn/ai - base_n/ai )
                transposed_beta = Matrix([solve(beta[i], delta_b[i])[0] for i in range(len(beta))])
                self.varchange_dict = dict(zip(delta_b,  beta_symbs + transposed_beta))

                #add free vars to variables for optimization
                for eq in transposed_beta:
                    for s in eq.free_symbols:
                        if s not in delta_d:
                            delta_d = delta_d.col_join(Matrix([s]))
            else:
                self.varchange_dict = dict(zip(delta_b,  beta_symbs - (beta - delta_b)))
            DB_blocks = [mrepl(Di, self.varchange_dict) for Di in self.D_blocks]
            epsilon_safemargin = 1e-6
            self.DB_LMIs_marg = list([LMI_PSD(lm - epsilon_safemargin*eye(lm.shape[0])) for lm in DB_blocks])

            #import ipdb; ipdb.set_trace()
            #embed()

            Q, R = la.qr(self.model.YBase)
            Q1 = Q[:, 0:self.model.num_base_params]
            #Q2 = Q[:, self.model.num_base_params:]
            rho1 = Q1.T.dot(self.model.torques_stack)
            R1 = np.matrix(R[:self.model.num_base_params, :self.model.num_base_params])

            # OLS: minimize ||tau - Y*x_base||^2 (simplify)=> minimize ||rho1.T - R1*K*delta||^2
            # sub contact forces
            if self.opt['floatingBase']:
                contactForces = Q.T.dot(self.model.contactForcesSum)
            else:
                contactForces = zeros(self.model.num_base_params, 1)

            if is_old_sympy:
                e_rho1 = Matrix(rho1).T - (R1*beta_symbs - contactForces)
            else:
                e_rho1 = Matrix(rho1) - (R1*beta_symbs - contactForces)

            rho2_norm_sqr = la.norm(self.model.torques_stack - self.model.YBase.dot(self.model.xBase))**2
            u = Symbol('u')
            U_rho = BlockMatrix([[Matrix([u - rho2_norm_sqr]), e_rho1.T],
                                 [e_rho1, I(self.model.num_base_params)]])
            U_rho = U_rho.as_explicit()

            if self.opt['verbose']:
                print("Add constraint LMIs")

            lmis = [LMI_PSD(U_rho)] + self.DB_LMIs_marg
            variables = [u] + list(beta_symbs) + list(delta_d)

            # solve SDP
            objective_func = u

            if self.opt['verbose']:
                print("Solving constrained OLS as SDP")

            # start at CAD data, might increase convergence speed (atm only works with dsdp5,
            # otherwise returns primal as solution when failing)
            # TODO: get success or fail status and use it (e.g. use other method if failing)
            prime = np.concatenate((self.model.xBaseModel, np.array(Pd.T*self.model.xStdModel)[:,0]))
            solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime)

            #try again with wider bounds and dsdp5 cmd line
            if state is not 'optimal':
                print("Trying again with dsdp5 solver")
                optimization.solve_sdp = optimization.dsdp5
                solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime, wide_bounds=True)
                optimization.solve_sdp = optimization.cvxopt_conelp

            u_star = solution[0,0]
            if u_star:
                print("found base solution with {} error increase from OLS solution".format(u_star))
            beta_star = np.matrix(solution[1:1+self.model.num_base_params])

            self.model.xBase = np.squeeze(np.asarray(beta_star))

        if self.opt['showTiming']:
            print("Constrained SDP optimization took %.03f sec." % (t.interval))


    def findFeasibleStdFromFeasibleBase(self, xBase):
        ''' find a std feasible solution for feasible base solution (exists by definition) while
            minimizing param distance to a-priori parameters
        '''

        def mrepl(m, repl):
            return m.applyfunc(lambda x: x.xreplace(repl))
        I = Identity

        #symbols for std params
        delta = Matrix(self.model.param_syms)

        # equations for base parameters expressed in independent std param symbols
        #beta = K * delta
        beta = self.model.base_deps #.applyfunc(lambda x: x.nsimplify())

        #add explicit constraints for each base param equation and estimated value
        D_base_val_blocks = []
        for i in range(self.model.num_base_params):
            D_base_val_blocks.append( Matrix([beta[i] - xBase[i] - 0.0001]) )
            D_base_val_blocks.append( Matrix([xBase[i] + 0.0001 - beta[i]]) )
        self.D_blocks += D_base_val_blocks

        epsilon_safemargin = 1e-6
        self.LMIs_marg = list([LMI_PSD(lm - epsilon_safemargin*eye(lm.shape[0])) for lm in self.D_blocks])

        sol_cad_dist = Matrix(self.model.xStdModel - self.model.param_syms)
        u = Symbol('u')
        U_rho = BlockMatrix([[Matrix([u]), sol_cad_dist.T],
                             [sol_cad_dist, I(self.model.num_params)]])
        U_rho = U_rho.as_explicit()

        lmis = [LMI_PSD(U_rho)] + self.LMIs_marg
        variables = [u] + list(self.model.param_syms)
        objective_func = u   # 'find' problem

        solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=self.model.xStdModel)

        #try again with wider bounds and dsdp5 cmd line
        if state is not 'optimal':
            print("Trying again with dsdp5 solver")
            optimization.solve_sdp = optimization.dsdp5
            # start at CAD data to find solution faster
            solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=self.model.xStdModel, wide_bounds=True)
            optimization.solve_sdp = optimization.cvxopt_conelp

        u = solution[0, 0]
        print("found std solution with distance {} from CAD solution".format(u))
        self.model.xStd = np.squeeze(np.asarray(solution[1:]))


    def findFeasibleStdFromStd(self, xStd):
        ''' find closest feasible std solution for some std parameters (increases error) '''

        delta = Matrix(self.model.param_syms)
        I = Identity

        Pd = Matrix(self.model.Pd)
        delta_d = (Pd.T*delta)

        u = Symbol('u')
        U_delta = BlockMatrix([[Matrix([u]),       (xStd - delta).T],
                               [xStd - delta,    I(self.model.num_params)]])
        U_delta = U_delta.as_explicit()
        lmis = [LMI_PSD(U_delta)] + self.LMIs_marg
        variables = [u] + list(delta)
        objective_func = u

        prime = self.model.xStdModel
        solution, state = optimization.solve_sdp(objective_func, lmis, variables, primalstart=prime)

        u_star = solution[0,0]
        if u_star:
            print("found std solution with {} error increase from previous solution".format(u_star))
        delta_star = np.matrix(solution[1:])
        xStd = np.squeeze(np.asarray(delta_star))

        return xStd


    def estimateParameters(self):
        '''identify parameters using data and regressor (method depends on chosen options)'''

        if not self.data.num_used_samples > self.model.num_params*2 \
            and 'selectingBlocks' in self.opt and not self.opt['selectingBlocks']:
            print(Fore.RED+"not enough samples for identification!"+Fore.RESET)
            if self.opt['startOffset'] > 0:
                print("(startOffset is at {})".format(self.opt['startOffset']))
            sys.exit(1)

        if self.opt['verbose']:
            print("doing identification on {} samples".format(self.data.num_used_samples), end=' ')

        self.model.computeRegressors(self.data)

        if self.opt['useEssentialParams']:
            self.identifyBaseParameters()
            self.findBaseEssentialParameters()
            if self.opt['useAPriori']:
                self.getBaseParamsFromParamError()
            self.findStdFromBaseEssParameters()
            self.identifyStandardEssentialParameters()
        else:
            #need to identify OLS base params in any case
            self.identifyBaseParameters()

            if self.opt['useConsistencyConstraints']:
                #do SDP constrained OLS identification
                self.initSDP_LMIs()

                if self.opt['useAPriori']:
                    self.getBaseParamsFromParamError()

                if self.opt['identifyClosestToCAD']:
                    # first estimate feasible base params, then find corresponding feasible std
                    # params while minimizing distance to CAD
                    self.identifyFeasibleBaseParameters()
                    self.findFeasibleStdFromFeasibleBase(self.model.xBase)
                else:
                    # directly estimate constrained std params, distance to CAD not minimized
                    if self.opt['estimateWith'] is 'std_direct':
                        self.identifyStandardParametersDirect()
                        self.identifyFeasibleStandardParametersDirect()
                    else:
                        self.identifyFeasibleStandardParameters()

                # get feasible base params, then project back to std. distance to CAD not minimized and std not feasible
                #self.identifyFeasibleBaseParameters()
                #self.findStdFromBaseParameters()

                #get OLS standard parameters (with a priori), then correct to feasible
                #self.findStdFromBaseParameters()
                #if self.opt['useAPriori']:
                #    self.getBaseParamsFromParamError()

                if not self.paramHelpers.isPhysicalConsistent(self.model.xStd):
                    print("Correcting solution to feasible std (non-optimal)")
                    self.model.xStd = self.findFeasibleStdFromStd(self.model.xStd)
            else:
                #identify with OLS only

                #get standard params from estimated base param error
                if self.opt['estimateWith'] == 'std_direct':
                    self.identifyStandardParametersDirect()
                else:
                    self.findStdFromBaseParameters()
                    #only then go back to absolute base params
                    if self.opt['useAPriori']:
                        self.getBaseParamsFromParamError()



    def plot(self, text=None):
        """Create state and torque plots."""

        rel_time = self.model.T-self.model.T[0]
        if self.validation_file:
            rel_vtime = self.Tv-self.Tv[0]

        if self.opt['floatingBase'] and self.opt['plotBaseDynamics']:
            torque_labels = self.model.baseNames + self.model.jointNames
        else:
            torque_labels = self.model.jointNames

        if self.opt['floatingBase'] and not self.opt['plotBaseDynamics']:
            tauMeasured = self.model.tauMeasured[:, 6:]
            tauEstimated = self.tauEstimated[:, 6:]
            tauAPriori = self.tauAPriori[:, 6:]
            if self.validation_file:
                tauEstimatedValidation = self.tauEstimatedValidation[:, 6:]
                tauMeasuredValidation = self.tauMeasuredValidation[:, 6:]
        else:
            tauMeasured = self.model.tauMeasured
            tauEstimated = self.tauEstimated
            tauAPriori = self.tauAPriori
            if self.validation_file:
                tauEstimatedValidation = self.tauEstimatedValidation
                tauMeasuredValidation = self.tauMeasuredValidation

        if self.opt['plotPerJoint']:
            datasets = []
            # add plots for each joint
            for i in range(self.model.N_DOFS):
                datasets.append(
                    { 'unified_scaling': True, 'y_label': 'Torque (Nm)',
                      'labels': ['Measured', 'Estimated', 'CAD', 'Error M/E'], 'contains_base': False,
                      'dataset': [
                        {'data': [np.vstack((tauMeasured[:,i], tauEstimated[:,i], tauAPriori[:,i],
                            #tauMeasured[:,i]-tauEstimated[:,i]   #plot error
                            )).T],
                         'time': rel_time, 'title': torque_labels[i]}
                      ]
                    }
                )

            """
            i = 10
            datasets.append(
                { 'unified_scaling': False, 'y_label': 'rad', 'labels': ['Position', 'Torq Meas', 'Torq Est'], 'dataset':
                  [{'data': [np.vstack((self.data.samples['positions'][0:self.model.sample_end:self.opt['skipSamples']+1, i]*100, tauMeasured[:,i], tauEstimated[:,i])).T],
                    'time': rel_time, 'title': self.model.jointNames[i]},
                  ]
                }
            )
            """

            # positions per joint
            for i in range(self.model.N_DOFS):
                datasets.append(
                    {'unified_scaling': False, 'y_label': 'rad', 'labels': ['Position'], 'dataset':
                     [{'data': [self.data.samples['positions'][0:self.model.sample_end:self.opt['skipSamples']+1, i],
                               #self.data.samples['target_positions'][0:self.model.sample_end:self.opt['skipSamples']+1, i]
                               ],
                       'time': rel_time, 'title': self.model.jointNames[i]},
                      ]
                    }
                )

            # vel and acc combined
            datasets.append(
                {'unified_scaling': False, 'y_label': 'rad/s (/s2)', 'labels': self.model.jointNames, 'dataset':
                 [{'data': [self.data.samples['velocities'][0:self.model.sample_end:self.opt['skipSamples']+1]],
                   'time': rel_time, 'title': 'Velocities'},
                  {'data': [self.data.samples['accelerations'][0:self.model.sample_end:self.opt['skipSamples']+1]],
                   'time': rel_time, 'title': 'Accelerations'},
                 ]
                }
            )
        else:
            datasets = [
                {'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                 'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                 'dataset':
                 [{'data': [tauMeasured], 'time': rel_time, 'title': 'Measured Torques'},
                  {'data': [tauEstimated], 'time': rel_time, 'title': 'Estimated Torques'},
                  {'data': [tauAPriori], 'time': rel_time, 'title': 'CAD Torques'},
                 ]
                },
                {'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                 'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                 'dataset':
                 [{'data': [tauMeasured-tauEstimated], 'time': rel_time, 'title': 'Ident. Estimation Error'},
                  {'data': [tauMeasured-tauAPriori], 'time': rel_time, 'title': 'CAD Estimation Error'},
                 ]
                },
                {'unified_scaling': False, 'y_label': 'rad (/s, /s2)', 'labels': self.model.jointNames, 'dataset':
                 [{'data': [self.data.samples['positions'][0:self.model.sample_end:self.opt['skipSamples']+1]],
                   'time': rel_time, 'title': 'Positions'},
                  {'data': [self.data.samples['velocities'][0:self.model.sample_end:self.opt['skipSamples']+1]],
                   'time': rel_time, 'title': 'Velocities'},
                  {'data': [self.data.samples['accelerations'][0:self.model.sample_end:self.opt['skipSamples']+1]],
                   'time': rel_time, 'title': 'Accelerations'},
                 ]
                }
            ]

            if 'positions_raw' in self.data.samples:
                datasets[2]['dataset'][0]['data'].append(self.data.samples['positions_raw'][0:self.model.sample_end:self.opt['skipSamples']+1])
            if 'velocities_raw' in self.data.samples:
                datasets[2]['dataset'][1]['data'].append(self.data.samples['velocities_raw'][0:self.model.sample_end:self.opt['skipSamples']+1])

        if self.validation_file:
            datasets.append(
                { 'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                    'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                    'dataset':
                    [#{'data': [self.tauMeasuredValidation],
                     # 'time': rel_vtime, 'title': 'Measured Validation'},
                     {'data': [tauEstimatedValidation],
                      'time': rel_vtime, 'title': 'Estimated Validation'},
                     {'data': [tauEstimatedValidation-tauMeasuredValidation],
                      'time': rel_vtime, 'title': 'Validation Error'}
                    ]
                }
            )

        from identification.output import OutputMatplotlib
        if self.opt['outputModule'] == 'matplotlib':
            output = OutputMatplotlib(datasets, html=False)
            output.render(self)
        elif self.opt['outputModule'] == 'html':
            output = OutputMatplotlib(datasets, html=True, text=text)
            output.render(self)
            #output.runServer()
        else:
            print('No known output module given. Not creating plots!')


    def printMemUsage(self):
        import humanize
        total = 0
        print("Memory usage:")
        for v in self.__dict__.keys():
            if type(self.__dict__[v]).__module__ == np.__name__:
                size = self.__dict__[v].nbytes
                total += size
                print("{}: {} ".format( v, (humanize.naturalsize(size, binary=True)) ), end=' ')
            #TODO: extend for builtins
        print("- total: {}".format(humanize.naturalsize(total, binary=True)))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('--config', required=True, type=str, help="use options from given config file")
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
    parser.set_defaults(plot=False, regressor=None, model_real=None)
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    #capture stdout and print
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = ""

        def write(self, message):
            self.terminal.write(message)
            self.log += message

        def flush(self):
            self.terminal.flush()

    sys.stdout = Logger()
    logger = sys.stdout

    idf = Identification(config, args.model, args.model_real, args.measurements, args.regressor, args.validation)

    if idf.opt['selectBlocksFromMeasurements']:
        idf.opt['selectingBlocks'] = 1
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
        idf.opt['selectingBlocks'] = 0
        idf.opt['useEssentialParams'] = old_essential_option
        idf.opt['useConsistencyConstraints'] = old_feasible_option

    if idf.opt['removeNearZero']:
        idf.data.removeNearZeroSamples()

    if idf.opt['verbose']:
        print("estimating output parameters...")
    idf.estimateParameters()
    idf.estimateRegressorTorques()

    if args.model_output:
        if not idf.paramHelpers.isPhysicalConsistent(idf.model.xStd):
            print("can't create urdf file with estimated parameters since they are not physical consistent.")
        else:
            idf.urdfHelpers.replaceParamsInURDF(input_urdf=args.model, output_urdf=args.model_output, \
                                        new_params=idf.model.xStd, link_names=idf.model.linkNames)

    OutputConsole.render(idf)
    if args.validation: idf.estimateValidationTorques()

    if idf.opt['createPlots']: idf.plot(text=logger.log)
    if idf.opt['showMemUsage']: idf.printMemUsage()

if __name__ == '__main__':
   # import ipdb
   # import traceback
    #try:
    main()
    print("\n")

    '''
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            # open ipdb when an exception happens
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
    '''
