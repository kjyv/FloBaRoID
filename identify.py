#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import input
from builtins import zip
from builtins import range
from builtins import object
import sys
from typing import cast, Dict, List, Iterable

# math
import numpy as np
import numpy.linalg as la
import scipy
import scipy.linalg as sla
import scipy.stats as stats

# plotting
import matplotlib.pyplot as plt

# kinematics, dynamics and URDF reading
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

# submodules
from identification.model import Model
from identification.data import Data
from identification.output import OutputConsole
from identification import sdp
import identification.helpers as helpers

from colorama import Fore
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
        # type: (Dict, str, str, str, str, str) -> None
        self.opt = opt

        # some additional options (experiments)

        # in order ot get regressor and base equations, use basis projection matrix. Otherwise use
        # permutation from QR directly (Gautier/Sousa method)
        self.opt['useBasisProjection'] = 0

        # in case projection is used, orthogonalize the basis matrix (SDP estimation seem to work
        # more stable that way)
        self.opt['orthogonalizeBasis'] = 1

        # add regularization term to SDP identification that minimized CAD distance for non-identifiable params
        self.opt['useRegressorRegularization'] = 1
        self.opt['regularizationFactor'] = 1000.0   #proportion of distance term

        # if using fixed base dynamics, remove first link that is the fixed base which should completely
        # not be identifiable and not be part of equations (as it does not move)
        self.opt['deleteFixedBase'] = 1

        # end additional config flags


        # load model description and initialize
        self.model = Model(self.opt, urdf_file, regressor_file)

        # load measurements
        self.data = Data(self.opt)
        if measurements_files:
            self.data.init_from_files(measurements_files)

        self.paramHelpers = helpers.ParamHelpers(self.model, self.opt)
        self.urdfHelpers = helpers.URDFHelpers(self.paramHelpers, self.model, self.opt)
        self.sdp = sdp.SDP(self)
        if self.opt['constrainUsingNL']:
            from identification.nlopt import NLOPT
            self.nlopt = NLOPT(self)

        self.tauEstimated = None    # type: np._ArrayLike
        self.res_error = 100        # last residual error in percent

        self.urdf_file_real = urdf_file_real
        if self.urdf_file_real:
            dc = iDynTree.DynamicsRegressorGenerator()
            if not dc.loadRobotAndSensorsModelFromFile(urdf_file_real):
                sys.exit()
            tmp = iDynTree.VectorDynSize(self.model.num_model_params)
            #set regressor, otherwise getModelParameters segfaults
            dc.loadRegressorStructureFromString(self.model.regrXml)
            dc.getModelParameters(tmp)
            self.xStdReal = tmp.toNumPy()
            #add some zeros for friction
            self.xStdReal = np.concatenate((self.xStdReal, np.zeros(self.model.num_all_params-self.model.num_model_params)))
            if self.opt['identifyFriction']:
                self.paramHelpers.addFrictionFromURDF(self.model, self.urdf_file_real, self.xStdReal)

        self.validation_file = validation_file

        progress_inst = helpers.Progress(opt)
        self.progress = progress_inst.progress


    def estimateRegressorTorques(self, estimateWith=None, print_stats=False):
        # type: (str, bool) -> None
        """ get torque estimations using regressor, prepare for plotting """

        if not estimateWith:
            # use global parameter choice if none is given specifically
            estimateWith = self.opt['estimateWith']
        # estimate torques with idyntree regressor and different params
        if estimateWith == 'urdf':
            tauEst = np.dot(self.model.YStd, self.model.xStdModel[self.model.identified_params])
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
        else:
            fb = 0

        if self.opt['floatingBase']:
            # the base forces are expressed in the base frame for the regressor, so transform them
            # to world frame (inverse dynamics use world frame)

            '''
            pos = iDynTree.Position.Zero()
            tau_2dim = tauEst.reshape((self.data.num_used_samples, self.model.num_dofs+fb))
            for i in range(self.data.num_used_samples):
                idx = i*(self.opt['skipSamples'])+i
                rpy = self.data.samples['base_rpy'][idx]
                rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
                world_T_base = iDynTree.Transform(rot, pos).inverse()
                to_world = world_T_base.getRotation().toNumPy()
                tau_2dim[i, :3] = to_world.dot(tau_2dim[i, :3])
                tau_2dim[i, 3:6] = to_world.dot(tau_2dim[i, 3:6])
            tauEst = tau_2dim.flatten()
            '''

        if self.opt['addContacts']:
            tauEst += self.model.contactForcesSum

        self.tauEstimated = np.reshape(tauEst, (self.data.num_used_samples, self.model.num_dofs + fb))
        self.base_error = np.mean(sla.norm(self.model.tauMeasured - self.tauEstimated, axis=1))

        # give some data statistics
        if print_stats and (self.opt['verbose'] or self.opt['showErrorHistogram'] == 1):
            error_per_joint = np.mean(self.model.tauMeasured - self.tauEstimated, axis=1)

            #how gaussian is the error of the data vs estimation?
            #http://stats.stackexchange.com/questions/62291/can-one-measure-the-degree-of-empirical-data-being-gaussian
            if self.opt['verbose'] >= 2:
                '''
                W, p = stats.shapiro(error)
                if p > 0.05:
                    print("error is normal distributed")
                else:
                    print("error is not normal distributed (p={})".format(p))
                print("W: {} (> 0.999 isn't too far from normality)".format(W))
                '''

                k2, p = stats.mstats.normaltest(error_per_joint)
                if p > 0.05:
                    print("error is normal distributed")
                else:
                    print("error is not normal distributed (p={})".format(p))
                print("k2: {} (the closer it is to 0, the closer to normal distributed)".format(k2))

            if self.opt['showErrorHistogram'] == 1:
                plt.hist(error_per_joint, 50)
                plt.title("error histogram")
                plt.draw()
                plt.show()
                # don't show again if we come here later
                self.opt['showErrorHistogram'] = 2

        # reshape torques into one column per DOF for plotting (NUM_SAMPLES*num_dofsx1) -> (NUM_SAMPLESxnum_dofs)
        if estimateWith == 'urdf':
            self.tauAPriori = self.tauEstimated


    def estimateValidationTorques(self):
        """ calculate torques of trajectory from validation measurements and identified params """
        # TODO: don't duplicate simulation code
        # TODO: get identified params directly into idyntree (new KinDynComputations class does not
        # have inverse dynamics yet, so we have to go over a new urdf file for now)

        import os

        v_data = np.load(self.validation_file)
        dynComp = iDynTree.DynamicsComputations()

        if self.opt['estimateWith'] == 'urdf':
            params = self.model.xStdModel
        else:
            params = self.model.xStd

        outfile = self.model.urdf_file + '.tmp.urdf'

        self.urdfHelpers.replaceParamsInURDF(input_urdf=self.model.urdf_file,
                                             output_urdf=outfile,
                                             new_params=params)
        if self.opt['useRBDL']:
            import rbdl
            self.model.rbdlModel = rbdl.loadModel(outfile,
                                                  floating_base=self.opt['floatingBase'],
                                                  verbose=False)
            self.model.rbdlModel.gravity = np.array(self.model.gravity)
        else:
            dynComp.loadRobotModelFromFile(outfile)
        os.remove(outfile)

        old_skip = self.opt['skipSamples']
        self.opt['skipSamples'] = 8

        self.tauEstimatedValidation = None   # type: np._ArrayLike
        for m_idx in self.progress(range(0, v_data['positions'].shape[0], self.opt['skipSamples'] + 1)):
            if self.opt['useRBDL']:
                torques = self.model.simulateDynamicsRBDL(v_data, m_idx, None, params)
            else:
                torques = self.model.simulateDynamicsIDynTree(v_data, m_idx, dynComp, params)

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
            self.tauMeasuredValidation = \
                np.concatenate((self.tauEstimatedValidation[:, :6], self.tauMeasuredValidation), axis=1)

            #TODO: add contact forces to estimation, so far validation is only correct for fixed-base!
            print(Fore.RED+'No proper validation for floating base yet!'+Fore.RESET)

        self.opt['skipSamples'] = old_skip

        self.val_error = sla.norm(self.tauEstimatedValidation - self.tauMeasuredValidation) \
                                  * 100 / sla.norm(self.tauMeasuredValidation)
        print("Relative validation error: {}%".format(self.val_error))
        self.val_residual = np.mean(sla.norm(self.tauEstimatedValidation-self.tauMeasuredValidation, axis=1))
        print("Absolute validation error: {} Nm".format(self.val_residual))

        torque_limits = []
        for joint in self.model.jointNames:
            torque_limits.append(self.model.limits[joint]['torque'])
        self.val_nrms = helpers.getNRMSE(self.tauMeasuredValidation, self.tauEstimatedValidation, limits=torque_limits)
        print("NRMS validation error: {}%".format(self.val_nrms))


    def getBaseParamsFromParamError(self):
        # type: () -> None
        self.model.xBase += self.model.xBaseModel   # both param vecs link relative linearized

        if self.opt['useEssentialParams']:
            self.xBase_essential[self.baseEssentialIdx] += self.model.xBaseModel[self.baseEssentialIdx]


    def findStdFromBaseParameters(self):
        # type: () -> None
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
            self.model.xStd += self.model.xStdModel[self.model.identified_params]


    def getStdDevForParams(self):
        # type: () -> (np._ArrayLike[float])
        # this might not be working correctly
        if self.opt['useAPriori']:
            tauDiff = self.model.tauMeasured - self.tauEstimated
        else:
            tauDiff = self.tauEstimated

        if self.opt['floatingBase']: fb = 6
        else: fb = 0

        # get relative standard deviation of measurement and modeling error \sigma_{rho}^2
        r = self.data.num_used_samples * (self.model.num_dofs + fb)
        rho = np.square(sla.norm(tauDiff))
        sigma_rho = rho / (r - self.model.num_base_params)

        # get standard deviation \sigma_{x} (of the estimated parameter vector x)
        C_xx = sigma_rho * (sla.pinv(np.dot(self.model.YBase.T, self.model.YBase)))
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
            not_essential_idx = list()   # type: List[int]
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

            if self.opt['verbose']:
                W, p = stats.shapiro(error_start)
                #k2, p = stats.normaltest(error_start, axis=0)
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
            p_sigma_x = np.array([0])

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
                    oc = OutputConsole(self)
                    oc.render(self)
                    self.opt['showStandardParams'] = old_showStd
                    self.opt['showBaseParams'] = old_showBase

                    print(base_idx, np.argmax(p_sigma_x))
                    print(self.baseNonEssentialIdx)
                    input("Press return...")
                else:
                    has_run_once = 1

                # cancel the parameter with largest deviation
                param_idx = cast(int, np.argmax(p_sigma_x))
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
            dependents = []   # type: List[int]
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
        #if self.opt['dontIdentifyMasses']:
        #    ps = list(range(0, self.model.num_identified_params, 10))
        #    self.stdEssentialIdx = np.fromiter((x for x in self.stdEssentialIdx if x not in ps), int)

        self.stdNonEssentialIdx = [x for x in range(0, self.model.num_identified_params) if x not in self.stdEssentialIdx]

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
                    if idx % 10 in [1,2,3]:   # com value
                        v = cast(float, np.mean(self.model.xStdModel[p_start + 1:p_start + 4]) * 0.1)
                    elif idx % 10 in [4,5,6,7,8,9]:  # inertia value
                        inertia_range = np.array([4,5,6,7,8,9])+p_start
                        v = cast(float, np.mean(self.model.xStdModel[np.where(self.model.xStdModel[inertia_range] != 0)[0]+p_start+4]) * 0.1)
                    if v == 0:
                        v = 0.1
                    self.xStdEssential[idx] = v
                    #print idx, idx % 10, v
                idx += 1
                if idx > self.model.num_model_params:
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
        # type: (np._ArrayLike, np._ArrayLike, bool) -> None
        """use previously computed regressors and identify base parameter vector using ordinary or
           weighted least squares."""

        if YBase is None:
            YBase = self.model.YBase
        if tau is None:
            tau = self.model.tau

        if self.opt['useBasisProjection']:
            self.model.xBaseModel = self.model.xStdModel.dot(self.model.B)
        else:
            self.model.xBaseModel = self.model.K.dot(self.model.xStdModel[self.model.identified_params])

        if self.urdf_file_real:
            if self.opt['useBasisProjection']:
                self.xBaseReal = np.dot(self.model.Binv, self.xStdReal[self.model.identified_params])
            else:
                self.xBaseReal = self.model.K.dot(self.xStdReal[self.model.identified_params])

        # note: using pinv is only ok if low condition number, otherwise numerical issues can happen
        # should always try to avoid inversion of ill-conditioned matrices if possible

        # invert equation to get parameter vector from measurements and model + system state values
        self.model.YBaseInv = la.pinv(YBase)

        # identify using numpy least squares method (should be numerically more stable)
        self.model.xBase = la.lstsq(YBase, tau)[0]
        if self.opt['addContacts']:
            self.model.xBase -=  self.model.YBaseInv.dot(self.model.contactForcesSum)

        """
        # using pseudoinverse
        self.model.xBase = self.model.YBaseInv.dot(tau.T) - self.model.YBaseInv.dot(self.model.contactForcesSum)

        # damped least squares
        from scipy.sparse.linalg import lsqr
        self.model.xBase = lsqr(YBase, tau, damp=10)[0] - self.model.YBaseInv.dot(self.model.contactForcesSum)
        """

        # stop here if called recursively
        if id_only:
            return

        if self.opt['showBaseParams'] or self.opt['verbose'] or self.opt['useRegressorRegularization']:
            # get estimation once with previous ordinary LS solution parameters
            self.estimateRegressorTorques('base', print_stats=True)
            if 'selectingBlocks' not in self.opt or not self.opt['selectingBlocks']:
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
            r = self.data.num_used_samples*(self.model.num_dofs+fb)

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
            #        self.num_dofs*self.num_used_samples, self.num_dofs*self.num_used_samples)
            #G = scipy.sparse.spdiags(np.repeat(1/np.sqrt(self.sigma_rho), self.data.num_used_samples), 0, r, r)
            G = scipy.sparse.spdiags(np.repeat(np.array([1/self.p_sigma_x]), self.data.num_used_samples), 0, r, r)

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

            # get non-singular std regressor
            V_1 = VH.T[:, 0:nb]
            U_1 = U[:, 0:nb]
            s_1 = np.diag(s[0:nb])
            s_1_inv = la.inv(s_1)
            W_st_pinv = V_1.dot(s_1_inv).dot(U_1.T)
            W_st = la.pinv(W_st_pinv)
            self.YStd_nonsing = W_st

            #TODO: add contact forces
            x_est = W_st_pinv.dot(self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_est
            else:
                self.model.xStd = x_est

            """
            st = self.model.num_identified_params
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

            #TODO: add contact forces
            x_tmp = W_st_e_pinv.dot(self.model.tau)

            if self.opt['useAPriori']:
                self.model.xStd = self.model.xStdModel + x_tmp
            else:
                self.model.xStd = x_tmp

        if self.opt['showTiming']:
            print("Identifying %s std essential parameters took %.03f sec." % (len(self.stdEssentialIdx), t.interval))


    def estimateParameters(self):
        '''identify parameters using data and regressor (method depends on chosen options)'''

        if not self.data.num_used_samples > self.model.num_identified_params*2 \
                and 'selectingBlocks' in self.opt and not self.opt['selectingBlocks']:
            print(Fore.RED+"not enough samples for identification!"+Fore.RESET)
            if self.opt['startOffset'] > 0:
                print("(startOffset is at {})".format(self.opt['startOffset']))
            sys.exit(1)

        if self.opt['verbose']:
            print("computing standard regressor matrix for data samples")

        self.model.computeRegressors(self.data)

        if self.opt['verbose']:
            print("estimating parameters using regressor")

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

            if self.opt['constrainToConsistent']:
                if self.opt['useAPriori']:
                    self.getBaseParamsFromParamError()

                if self.opt['identifyClosestToCAD']:
                    # first estimate feasible base params, then find corresponding feasible std
                    # params while minimizing distance to CAD
                    self.sdp.initSDP_LMIs(self)
                    self.sdp.identifyFeasibleStandardParameters(self)

                    # get feasible base solution by projection
                    if self.opt['useBasisProjection']:
                        self.model.xBase = self.model.Binv.dot(self.model.xStd)
                    else:
                        self.model.xBase = self.model.K.dot(self.model.xStd)

                    print("Trying to find equal solution closer to a priori values")

                    if self.opt['constrainUsingNL']:
                        self.nlopt.identifyFeasibleStdFromFeasibleBase(self.model.xBase)
                    else:
                        self.sdp.findFeasibleStdFromFeasibleBase(self, self.model.xBase)
                else:
                    self.sdp.initSDP_LMIs(self)
                    # directly estimate constrained std params, distance to CAD not minimized
                    if self.opt['estimateWith'] == 'std_direct':
                        #self.identifyStandardParametersDirect()   #get std nonsingular regressor
                        self.sdp.identifyFeasibleStandardParametersDirect(self)  #use with sdp
                    else:
                        if self.opt['constrainUsingNL']:
                            self.model.xStd = self.model.xStdModel.copy()
                            self.nlopt.identifyFeasibleStandardParameters()
                        else:
                            self.sdp.identifyFeasibleStandardParameters(self)
                        #self.sdp.identifyFeasibleBaseParameters(self)
                        #self.model.xStd = self.model.xBase.dot(self.model.K)

                    if self.opt['useBasisProjection']:
                        self.model.xBase = self.model.Binv.dot(self.model.xStd)
                    else:
                        self.model.xBase = self.model.K.dot(self.model.xStd)

                # get OLS standard parameters (with a priori), then correct to feasible
                #self.findStdFromBaseParameters()
                #if self.opt['useAPriori']:
                #    self.getBaseParamsFromParamError()

                # correct std solution to feasible if necessary (e.g. infeasible solution from
                # unsuccessful optimization run)
                """
                if not self.paramHelpers.isPhysicalConsistent(self.model.xStd) and not self.opt['constrainUsingNL']:
                    #get full LMIs again
                    self.opt['deleteFixedBase'] = 0
                    self.sdp.initSDP_LMIs(self, remove_nonid=False)
                    print("Correcting solution to feasible std (non-optimal)")
                    self.model.xStd = self.sdp.findFeasibleStdFromStd(self, self.model.xStd)
                """
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
        # type: (str) -> None
        """Create state and torque plots."""

        if self.opt['verbose']:
            print('plotting')

        rel_time = self.model.T-self.model.T[0]
        if self.validation_file:
            rel_vtime = self.Tv-self.Tv[0]

        if self.opt['floatingBase']:
            fb = 6
        else:
            fb = 0

        if not self.opt['plotBaseDynamics'] or not self.opt['floatingBase']:
            # get only data for joints (skipping base data if present)
            tauMeasured = self.model.tauMeasured[:, fb:]
            tauEstimated = self.tauEstimated[:, fb:]
            tauAPriori = self.tauAPriori[:, fb:]
            if self.validation_file:
                tauEstimatedValidation = self.tauEstimatedValidation[:, fb:]
                tauMeasuredValidation = self.tauMeasuredValidation[:, fb:]
            torque_labels = self.model.jointNames
        else:
            # get all data for floating base
            tauMeasured = self.model.tauMeasured
            tauEstimated = self.tauEstimated
            tauAPriori = self.tauAPriori
            if self.validation_file:
                tauEstimatedValidation = self.tauEstimatedValidation
                tauMeasuredValidation = self.tauMeasuredValidation
            torque_labels = self.model.baseNames + self.model.jointNames

        if self.opt['plotPerJoint']:
            datasets = []
            # plot base dynamics
            if self.opt['floatingBase']:
                if self.opt['plotBaseDynamics']:
                    for i in range(6):
                        datasets.append({
                            'unified_scaling': False,
                            #'y_label': '$F {{ {} }}$ (Nm)'.format(i),
                            'y_label': 'Force (N)',
                            'labels': ['Measured', 'Identified'], 'contains_base': False,
                            'dataset': [{
                                'data': [np.vstack((tauMeasured[:,i], tauEstimated[:,i])).T],
                                'time': rel_time, 'title': torque_labels[i]}
                            ]}
                        )

            # add plots for each joint
            for i in range(fb, self.model.num_dofs):
                datasets.append({
                    'unified_scaling': False,
                    #'y_label': '$\\tau_{{ {} }}$ (Nm)'.format(i+1),
                    'y_label': 'Torque (Nm)',
                    'labels': ['Measured', 'Identified'], 'contains_base': False,
                    'dataset': [{
                        'data': [np.vstack((tauMeasured[:,i], tauEstimated[:,i])).T],
                        'time': rel_time, 'title': torque_labels[i]}
                    ]}
                )
                if self.opt['plotPrioriTorques']:
                    #plot a priori torques
                    apriori = tauAPriori[:,i]
                    datasets[-1]['dataset'][0]['data'][0] = np.vstack((datasets[-1]['dataset'][0]['data'][0].T, apriori)).T
                    datasets[-1]['labels'].append('CAD')

                if self.opt['plotErrors']:
                    #plot joint torque errors
                    e = tauMeasured[:,i] - tauEstimated[:,i]
                    datasets[-1]['dataset'][0]['data'][0] = np.vstack((datasets[-1]['dataset'][0]['data'][0].T, e)).T
                    datasets[-1]['labels'].append('Error M/E')

            # positions per joint
            for i in range(self.model.num_dofs):
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
        else:   #don't plot per joint
            datasets = [
                {'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                 'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                 'dataset':
                 [{'data': [tauMeasured], 'time': rel_time, 'title': 'Measured Torques'},
                  {'data': [tauEstimated], 'time': rel_time, 'title': 'Estimation with identified Params'},
                  {'data': [tauAPriori], 'time': rel_time, 'title': 'Estimation with A priori Params'},
                 ]
                },
                {'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                 'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                 'dataset':
                 [{'data': [tauMeasured-tauEstimated], 'time': rel_time, 'title': 'Identified Estimation Error'},
                  {'data': [tauMeasured-tauAPriori], 'time': rel_time, 'title': 'A priori Estimation Error'},
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
                    {'unified_scaling': True, 'y_label': 'Torque (Nm)', 'labels': torque_labels,
                   'contains_base': self.opt['floatingBase'] and self.opt['plotBaseDynamics'],
                   'dataset':
                   [#{'data': [self.tauMeasuredValidation],
                    # 'time': rel_vtime, 'title': 'Measured Validation'},
                    {'data': [tauEstimatedValidation],
                     'time': rel_vtime, 'title': 'Estimated Validation'},
                    {'data': [tauEstimatedValidation - tauMeasuredValidation],
                     'time': rel_vtime, 'title': 'Validation Error'}
                   ]
                }
            )

        if self.opt['outputModule'] == 'matplotlib':
            from identification.output import OutputMatplotlib
            output = OutputMatplotlib(datasets, text=text)
            output.render(self)
        else:
            print('No known output module given. Not creating plots!')


    def printMemUsage(self):
        import humanize
        total = 0
        print("Memory usage:")
        for v in self.__dict__:
            if type(self.__dict__[v]).__module__ == np.__name__:
                size = self.__dict__[v].nbytes
                total += size
                print("{}: {} ".format(v, (humanize.naturalsize(size, binary=True))), end=' ')
            #TODO: extend for builtins
        print("- total: {}".format(humanize.naturalsize(total, binary=True)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Load measurements and URDF model to get inertial parameters.')
    parser.add_argument('--config', required=True, type=str, help="use options from given config file")
    parser.add_argument('-m', '--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--model_real', required=False, type=str, help='the file to load the model params for\
                        comparison from')
    parser.add_argument('-o', '--model_output', '--output', required=False, type=str, help='the file to save the identified params to')

    parser.add_argument('--measurements', required=True, nargs='+', action='append', type=str,
                        help='the file(s) to load the measurements from')

    parser.add_argument('--validation', '--verification', '--verify', required=False, type=str,
                        help='the file to load the validation trajectory from')

    parser.add_argument('--regressor', required=False, type=str,
                        help='the file containing the regressor structure (for the iDynTree generator).\
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
            self.log = u""

        def write(self, message):
            self.terminal.write(message)
            self.log += message

        def flush(self):
            self.terminal.flush()

    sys.stdout = Logger()  # type: ignore
    logger = sys.stdout

    #for ipython, reset this with
    #import sys
    #sys.stdout = sys.__stdout__

    idf = Identification(config, args.model, args.model_real, args.measurements, args.regressor, args.validation)

    if idf.opt['selectBlocksFromMeasurements']:
        idf.opt['selectingBlocks'] = 1
        old_essential_option = idf.opt['useEssentialParams']
        idf.opt['useEssentialParams'] = 0

        old_feasible_option = idf.opt['constrainToConsistent']
        idf.opt['constrainToConsistent'] = 0

        # loop over input blocks and select good ones
        while 1:
            idf.estimateParameters()
            idf.data.getBlockStats(idf.model)
            idf.estimateRegressorTorques()
            oc = OutputConsole(idf)
            oc.render(summary_only=True)

            if idf.data.hasMoreSamples():
                idf.data.getNextSampleBlock()
            else:
                break

        idf.data.selectBlocks()
        idf.data.assembleSelectedBlocks()
        idf.opt['selectingBlocks'] = 0
        idf.opt['useEssentialParams'] = old_essential_option
        idf.opt['constrainToConsistent'] = old_feasible_option

    if idf.opt['removeNearZero']:
        idf.data.removeNearZeroSamples()

    idf.estimateParameters()
    idf.estimateRegressorTorques(print_stats=False)

    oc = OutputConsole(idf)
    oc.render()
    if args.validation: idf.estimateValidationTorques()

    if args.model_output:
        if not idf.paramHelpers.isPhysicalConsistent(idf.model.xStd):
            print("can't create urdf file with estimated parameters since they are not physical consistent.")
        else:
            idf.urdfHelpers.replaceParamsInURDF(input_urdf=args.model, output_urdf=args.model_output,
                                                new_params=idf.model.xStd)

    if idf.opt['createPlots']: idf.plot(text=logger.log)
    if idf.opt['showMemUsage']: idf.printMemUsage()


if __name__ == '__main__':
    #import ipdb
    #import traceback
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
