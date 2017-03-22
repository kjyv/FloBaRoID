from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

from typing import List, Dict, Any

import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy import signal
from scipy import misc
from identification.helpers import Timer
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import matplotlib.pyplot as plt
from IPython import embed

class Data(object):
    def __init__(self, opt):
        # type: (Dict[str, Any]) -> None
        self.opt = opt
        self.measurements = {}  # type: Dict[str, np._ArrayLike]   #loaded data
        self.num_loaded_samples = 0    # no of samples from file
        self.num_used_samples = 0      # no of samples after skipping
        self.samples = {}     # type: Dict[str, np._ArrayLike]   #selected data (when using block selection)

        self.usedBlocks = list()     # type: List[int]
        self.unusedBlocks = list()   # type: List[int]
        self.seenBlocks = list()     # type: List[int]

        # has some data been loaded?
        self.inited = False

    def init_from_data(self, data):
        # type: (Dict[str, np._ArrayLike]) -> None
        '''load data from numpy array'''

        self.samples = self.measurements = data.copy()
        self.num_loaded_samples = self.samples['positions'].shape[0]
        self.num_used_samples = self.num_loaded_samples//(self.opt['skipSamples']+1)
        if self.opt['verbose']:
            print('loaded {} data samples (using {})'.format(
                self.num_loaded_samples, self.num_used_samples))
        self.inited = True


    def init_from_files(self, measurements_files):
        '''load data from measurements_files, optionally skipping some values'''

        with Timer() as t:
            so = self.opt['startOffset']
            # load data from multiple files and concatenate, fix timing
            for fa in measurements_files:
                for fn in fa:
                    try:
                        #python3
                        m = np.load(fn, encoding='latin1', fix_imports=True)
                    except:
                        #python2.7
                        m = np.load(fn)
                    mv = {}
                    for k in m.keys():
                        mv[k] = m[k]
                        if k not in self.measurements:
                            # first file
                            if m[k].ndim == 0:
                                if isinstance(m[k].item(0), dict):
                                    #contacts
                                    contact_dict = {}
                                    for c in m[k].item(0).keys():
                                        if c != 'dummy_sim':   # could be removed but is here for compatibility
                                            contact_dict[c] = m[k].item(0)[c][so:, :]
                                    self.measurements[k] = np.array(contact_dict)
                                else:
                                    self.measurements[k] = m[k]
                            elif m[k].ndim == 1:
                                self.measurements[k] = m[k][so:]
                            else:
                                self.measurements[k] = m[k][so:, :]
                        else:
                            # following files, append data
                            if m[k].ndim == 0:
                                if isinstance(m[k].item(0), dict):
                                    #contacts
                                    contact_dict = {}
                                    for c in m[k].item(0).keys():
                                        if c != 'dummy_sim':
                                            contact_dict[c] = m[k].item(0)[c][so:, :]
                                    self.measurements[k] = np.array(contact_dict)
                                else:
                                    #TODO: get mean value of scalar values (needs to count how many values then)
                                    self.measurements[k] = m[k]
                            elif m[k].ndim == 1:
                                if k == 'times':
                                    # shift new values to start at 0 plus first time diff
                                    mv[k] = m[k] - m[k][so] + (m[k][so+1]-m[k][so])
                                    # add after last timestamp of previous data
                                    mv[k] = mv[k] + self.measurements[k][-1]
                                self.measurements[k] = np.concatenate( (self.measurements[k],
                                                                        mv[k][so:]),
                                                                        axis=0)
                            else:
                                self.measurements[k] = np.concatenate( (self.measurements[k],
                                                                        mv[k][so:, :]),
                                                                        axis=0)
                    m.close()

            self.num_loaded_samples = self.measurements['positions'].shape[0]
            self.num_used_samples = self.num_loaded_samples//(self.opt['skipSamples']+1)
            if self.opt['verbose']:
                print('loaded {} measurement samples (using {})'.format(
                    self.num_loaded_samples, self.num_used_samples))

            # create data that identification is working on (subset of all measurements)
            self.samples = {}
            self.block_pos = 0
            if self.opt['selectBlocksFromMeasurements']:
                # fill only with starting block
                for k in self.measurements.keys():
                    #TODO: add contacts into this logic as well
                    if self.measurements[k].ndim == 0:
                        self.samples[k] = self.measurements[k]
                    elif self.measurements[k].ndim == 1:
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.opt['blockSize']]
                    else:
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.opt['blockSize'], :]

                self.num_selected_samples = self.samples['positions'].shape[0]
                self.num_used_samples = self.num_selected_samples//(self.opt['skipSamples']+1)
            else:
                # simply use all data
                self.samples = self.measurements
            # rest of data selection is done from identification.py ATM, selected data will then be
            # in samples dict

        if self.opt['showTiming']:
            print("(loading samples from file took %.03f sec.)" % t.interval)

        self.inited = True

    def hasMoreSamples(self):
        """ tell if there are more samples to be added to the data used for identification """

        if not self.opt['selectBlocksFromMeasurements']:
            return False

        if self.block_pos + self.opt['blockSize'] >= self.num_loaded_samples:
            return False

        return True

    def updateNumSamples(self):
        self.num_selected_samples = self.samples['positions'].shape[0]
        self.num_used_samples = self.num_selected_samples//(self.opt['skipSamples']+1)

    def removeLastSampleBlock(self):
        if self.opt['verbose']:
            print("removing block starting at {}".format(self.block_pos))
        for k in self.measurements.keys():
            self.samples[k] = np.delete(self.samples[k], list(range(self.num_selected_samples - self.opt['blockSize'],
                                                               self.num_selected_samples)), axis=0)
        self.updateNumSamples()
        if self.opt['verbose']:
            print("we now have {} samples selected (using {})".format(self.num_selected_samples, self.num_used_samples))

    def getNextSampleBlock(self):
        """ fill samples with next measurements block """

        # advance to next block or end of data
        self.block_pos += self.opt['blockSize']

        if self.block_pos + self.opt['blockSize'] > self.num_loaded_samples:
            self.opt['blockSize'] = self.num_loaded_samples - self.block_pos

        if self.opt['verbose']:
            print("getting next block: {}/{}".format(self.block_pos, self.num_loaded_samples))

        #TODO: add contacts into this logic as well
        for k in self.measurements.keys():
            if self.measurements[k].ndim == 0:
                mv = self.measurements[k]
            elif self.measurements[k].ndim == 1:
                mv = self.measurements[k][self.block_pos:self.block_pos + self.opt['blockSize']]
            else:
                mv = self.measurements[k][self.block_pos:self.block_pos + self.opt['blockSize'],:]
            self.samples[k] = mv

        self.updateNumSamples()

    def getBlockStats(self, model):
        """ check if we want to keep a new data block with the already selected ones """
        self.model = model

        # possible criteria for minimization:
        # * condition number of (base) regressor (+variations)
        # * largest per link condition number gets smaller (some are really huge though and are not
        # getting smaller with most new data)
        # * estimation error gets smaller (same data or validation)
        # ratio of min/max rel std devs

        # use condition number of regressor
        #new_condition_number = la.cond(self.YBase.dot(np.diag(self.xBaseModel)))   #weighted with a priori
        new_condition_number = la.cond(model.YBase)

        # get condition number for each of the links
        linkConds = model.getSubregressorsConditionNumbers()

        """
        # use largest link condition number
        largest_idx = np.argmax(linkConds)  #2+np.argmax(linkConds[2:7])
        new_condition_number = linkConds[largest_idx]
        """

        # use validation error
        # new_condition_number = self.val_error

        # use std dev ratio
        ##self.estimateRegressorTorques()
        """
        # get standard deviation of measurement and modeling error \sigma_{rho}^2
        rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))
        sigma_rho = rho/(self.num_used_samples-self.model.num_base_params)

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

        self.seenBlocks.append((self.block_pos, self.opt['blockSize'], new_condition_number, linkConds))

    def selectBlocks(self):
        """of all blocks loaded, select only those that create minimal condition number (cf. Venture, 2010)"""

        # select blocks with some best % of condition numbers
        perc_cond = np.percentile([cond for (b,bs,cond,linkConds) in self.seenBlocks], self.opt['selectBestPerenctage'])

        cond_matrix = np.zeros((len(self.seenBlocks), self.model.num_links))
        c = 0
        for block in self.seenBlocks:
            (b,bs,cond,linkConds) = block
            if cond > perc_cond:
                if self.opt['verbose']:
                    print("not using block starting at {} (cond {})".format(b, cond))
                self.unusedBlocks.append(block)
            else:
                if self.opt['verbose']:
                    print("using block starting at {} (cond {})".format(b, cond))
                self.usedBlocks.append(block)

                # create variance matrix
                cond_matrix[c, :] = linkConds
                c+=1

        ## look at sub-regressor patterns and throw out some similar blocks
        if self.opt['verbose']:
            print("checking for similar sub-regressor patterns")

        #check for pairs that are less than e.g. 15% of each other away
        #if found, delete larger one of the original blocks from usedBlocks (move to unused)
        #TODO: check this with the same file twice as input, should not use any blocks from the second file
        variances = np.var(cond_matrix[0:c,:],axis=1)
        v_idx = np.array(list(range(0, c)))
        sort_idx = np.argsort(variances)

        to_delete = list()
        dist = 0.15
        i = 1
        while i < c:
            #keep two values of three close ones (only remove middle)
            #TODO: generalize to more values with this pattern
            if i<c-1 and np.abs(variances[sort_idx][i-1]-variances[sort_idx][i+1]) < np.abs(variances[sort_idx][i+1])*dist:
                to_delete.append(v_idx[sort_idx][i])
                i+=1
            #remove first if two are too close
            elif np.abs(variances[sort_idx][i-1]-variances[sort_idx][i]) < np.abs(variances[sort_idx][i])*dist:
                to_delete.append(v_idx[sort_idx][i-1])
            i+=1


        for d in np.sort(to_delete)[::-1]:
            if self.opt['verbose']:
                print("delete block {}".format(self.usedBlocks[d][0]))
            del self.usedBlocks[d]


    def assembleSelectedBlocks(self):
        self.model.getSubregressorsConditionNumbers()
        if self.opt['verbose']:
            print("assembling selected blocks...\n")
        for k in self.measurements.keys():
            if not len(self.usedBlocks):
                break

            #init with first block
            (b, bs, cond, linkConds) = self.usedBlocks[0]
            if self.measurements[k].ndim == 0:
                self.samples[k] = self.measurements[k]
            else:
                self.samples[k] = self.measurements[k][b:b+bs]

            #append
            for i in range(1, len(self.usedBlocks)):
                (b, bs, cond, linkConds) = self.usedBlocks[i]
                #TODO: add contacts into this logic as well
                if self.measurements[k].ndim == 0:
                    self.samples[k] = self.measurements[k]
                elif self.measurements[k].ndim == 1:
                    mv = self.measurements[k][b:b + bs]
                    #fix time offsets
                    mv = mv - mv[0] + (mv[1]-mv[0]) #let values start with first time diff
                    mv = mv + self.samples[k][-1]   #add after previous times
                    self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
                else:
                    mv = self.measurements[k][b:b + bs,:]
                    self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
        self.updateNumSamples()


    def removeNearZeroSamples(self):
        '''remove samples that have near zero velocity'''

        if self.opt['verbose']:
            print("removing near zero samples...", end=' ')
        to_delete = list()
        for t in range(self.num_loaded_samples):
            if np.max(np.abs(self.samples['velocities'][t])) < self.opt['minVel']:
                to_delete.append(t)

        for k in self.samples.keys():
            if self.samples[k].ndim == 0:
                #contacts
                if isinstance(self.samples[k].item(0), dict):
                    #contacts
                    for c in self.samples[k].item(0).keys():
                        self.samples[k].item(0)[c] = np.delete(self.samples[k].item(0)[c], to_delete, 0)
            else:
                self.samples[k] = np.delete(self.samples[k], to_delete, 0)
        self.updateNumSamples()
        if self.opt['verbose']:
            print ("remaining samples: {}".format(self.num_used_samples))


    def preprocess(self, Q, V, Vdot, Tau, T, Fs, Q_raw=None, V_raw=None, Tau_raw=None,
                   IMUlinVel=None, IMUrotVel=None, IMUlinAcc=None, IMUrotAcc=None,
                   IMUrpy=None, FT=None):

        ''' derivation and filtering of measurements. array values are set in place
            Q, Tau will be filtered
            V, Vdot, *_raw will be overwritten (initialized arrays need to be passed to have values written)

            IMUrotVel, IMUlinAcc, IMUrpy will be filtered
            IMUlinVel, IMUrotAcc will be overwritten
        '''

        def central_diff( array, times, n = 2 ):
            #varying time step
            div = times[1] - times[0]

            #central difference from Sousa code
            size = len( array )
            diff = np.zeros_like( array )
            if n == 1:
                diff[0] = ( array[1] - array[0]  ) / div
                for i in range(1, size-1):
                    div = times[i] - times[i-1]
                    diff[i] = ( array[i+1] - array[i-1]  ) / (2*div)
                diff[size-1] = ( array[size-1] - array[size-2]  ) / div
            elif n == 2:
                diff[0] = ( array[1] - array[0]  ) / div
                diff[1] = ( array[2] - array[0]  ) / (2*div)
                for i in range(2, size-2):
                    div = times[i] - times[i-1]
                    diff[i] = ( - array[i+2] + 8*array[i+1] - 8*array[i-1] + array[i-2] ) / (12*div)
                diff[size-2] = ( array[size-1] - array[size-3]  ) / (2*div)
                diff[size-1] = ( array[size-1] - array[size-2]  ) / div
            else:
                raise Exception('use n = 1 or 2')
            return diff

        def plot_filter(b,a):
            # Plot the frequency and phase response of the filter
            w, h = sp.signal.freqz(b, a, worN=8000)
            plt.subplot(2, 1, 1)
            plt.plot(0.5*Fs*w / np.pi, np.abs(h), 'b')
            plt.plot(fc, 0.5*np.sqrt(2), 'ko')
            plt.axvline(fc, color='k')
            plt.xlim(0, 0.5*Fs)
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
            plt.show()

        median_kernel_size = self.opt['filterMedianSize']

        ## Joint Positions

        # convert degs to rads
        # assuming angles don't wrap, otherwise use np.unwrap before
        if self.opt['useDeg']:
            posis_rad = np.deg2rad(Q)
            vels_rad = np.deg2rad(V)
            np.copyto(Q, posis_rad)   #(dst, src)
            np.copyto(V, vels_rad)

        #init low pass filter coefficients
        fc = self.opt['filterLowPass3'][0]   #3.0  #Cut-off frequency (Hz)
        order = self.opt['filterLowPass3'][1]  #4   #Filter order
        b_3, a_3 = sp.signal.butter(order, fc / (Fs/2), btype='low', analog=False)

        fc = self.opt['filterLowPass2'][0]  #6.0  #Cut-off frequency (Hz)
        order = self.opt['filterLowPass2'][1] #5   #Filter order
        b_6, a_6 = sp.signal.butter(order, fc / (Fs/2), btype='low', analog=False)

        fc = self.opt['filterLowPass1'][0]  #8.0  #Cut-off frequency (Hz)
        order = self.opt['filterLowPass1'][1] #5   #Filter order
        b_8, a_8 = sp.signal.butter(order, fc / (Fs/2), btype='low', analog=False)

        #plot_filter(b, a)

        # low-pass filter positions
        Q_orig = Q.copy()
        for j in range(0, self.opt['num_dofs']):
            Q[:, j] = sp.signal.filtfilt(b_8, a_8, Q_orig[:, j])
        if Q_raw is not None:
            np.copyto(Q_raw, Q_orig)

        ## Joint Velocities

        # calc velocity instead of taking measurements (uses filtered positions,
        # seems better than filtering noisy velocity measurements)
        V_self = np.empty_like(Q)

        diff = central_diff(Q, T, 2)
        np.copyto(V_self, diff)

        if V_raw is not None:
            np.copyto(V_raw, V_self)

        # median filter of velocities self to remove outliers
        vels_self_orig = V_self.copy()
        for j in range(0, self.opt['num_dofs']):
            V_self[:, j] = sp.signal.medfilt(vels_self_orig[:, j], median_kernel_size)

        # low-pass filter velocities self
        vels_self_orig = V_self.copy()
        for j in range(0, self.opt['num_dofs']):
            V_self[:, j] = sp.signal.filtfilt(b_6, a_6, vels_self_orig[:, j])

        np.copyto(V, V_self)

        ## Joint Accelerations

        # calc accelerations
        diff = central_diff(V_self, T, 2)
        np.copyto(Vdot, diff)

        # median filter of accelerations
        accls_orig = Vdot.copy()
        for j in range(0, self.opt['num_dofs']):
            Vdot[:, j] = sp.signal.medfilt(accls_orig[:, j], median_kernel_size)

        # low-pass filter of accelerations
        #accls_orig = Vdot.copy()
        #for j in range(0, self.opt['num_dofs']):
        #    Vdot[:, j] = sp.signal.filtfilt(b_3, a_3, accls_orig[:, j])

        ## Joint Torques

        if Tau_raw is not None:
            np.copyto(Tau_raw, Tau)

        # median filter of torques
        torques_orig = Tau.copy()
        for j in range(0, self.opt['num_dofs']):
            Tau[:, j] = sp.signal.medfilt(torques_orig[:, j], median_kernel_size)

        # low-pass of torques
        torques_orig = Tau.copy()
        for j in range(0, self.opt['num_dofs']):
            Tau[:, j] = sp.signal.filtfilt(b_8, a_8, torques_orig[:, j])

        ### IMU data
        if IMUlinAcc is not None and IMUrotVel is not None:
            # median filter
            IMUlinAcc_orig = IMUlinAcc.copy()
            IMUrotVel_orig = IMUrotVel.copy()
            for j in range(0, 3):
                IMUlinAcc[:, j] = sp.signal.medfilt(IMUlinAcc_orig[:, j], median_kernel_size)
                IMUrotVel[:, j] = sp.signal.medfilt(IMUrotVel_orig[:, j], median_kernel_size)

            #plot_filter(b_8, a_8)

            # low-pass filter
            IMUlinAcc_orig = IMUlinAcc.copy()
            IMUrotVel_orig = IMUrotVel.copy()
            IMUrpy_orig = IMUrpy.copy()
            for j in range(0, 3):
                IMUlinAcc[:, j] = sp.signal.filtfilt(b_8, a_8, IMUlinAcc_orig[:, j])
                IMUrotVel[:, j] = sp.signal.filtfilt(b_8, a_8, IMUrotVel_orig[:, j])
                IMUrpy[:, j] = sp.signal.filtfilt(b_3, a_3, IMUrpy_orig[:, j])

            if IMUlinVel is not None:
                #rotate data to (estimated) world frame (iDynTree floating base wants that)
                #TODO: use quaternions to avoid gimbal lock (orientation estimation needs to give quaternions already)
                IMUlinAccWorld = np.zeros_like(IMUlinAcc)
                IMUrotVelWorld = np.zeros_like(IMUrotVel)
                for i in range(0, IMUlinAcc.shape[0]):
                    rot = IMUrpy[i, :]
                    R = iDynTree.Rotation.RPY(rot[0], rot[1], rot[2]).toNumPy()
                    IMUlinAccWorld[i, :] = R.dot(IMUlinAcc[i, :])
                    IMUrotVelWorld[i, :] = R.dot(IMUrotVel[i, :])
                np.copyto(IMUrotVel, IMUrotVelWorld)

                grav_norm = np.mean(la.norm(IMUlinAccWorld, axis=1))
                if grav_norm < 9.81 or grav_norm > 9.82:
                    print('Warning: mean base acceleration is different than gravity ({})!'.format(grav_norm))
                    #scale up/down
                    #IMUlinAccWorld *= 9.81/grav_norm

                # subtract gravity vector
                IMUlinAccWorld -= np.array([0,0,-9.81])

                #try to skip initial values until close to 0 acceleration time is found
                if self.opt['waitForZeroAcc']:
                    means = np.mean(IMUlinAccWorld, axis=0)
                    IMUlinAccWorld -= means

                    start = 0
                    for j in range(0, 3):
                        #only start integrating when acceleration is small
                        for s in range(0, IMUlinAccWorld.shape[0]):
                            if la.norm(IMUlinAccWorld[s:s+10, j]) < self.opt['zeroAccThresh']:
                                start = np.max((s, start))
                                break

                    IMUlinAccWorld[:start, :] = 0
                    IMUlinAccWorld += means
                else:
                    if la.norm(IMUlinAccWorld[:, 0]) > 0.1:
                        print("Warning: proper base acceleration not zero at time 0 " \
                              "(assuming start at zero, integrated velocity will be wrong)!")

                # subtract means, includes wrong gravity offset and other static offsets
                IMUlinAccWorld -= np.mean(IMUlinAccWorld, axis=0)
                np.copyto(IMUlinAcc, IMUlinAccWorld)

                # integrate linear acceleration to get velocity
                for j in range(0, 3):
                    IMUlinVel[:, j] = sp.integrate.cumtrapz(IMUlinAcc[:, j], T, initial=0)
                    IMUlinVel[:, j] -= np.mean(IMUlinVel[:, j])   #indefinite integral, better constant correction?
                    #IMUlinVel[j, :] = R.T.dot(IMUlinVel[j, :])

            # get rotational acceleration as simple derivative of velocity
            if IMUrotAcc is not None:
                for j in range(0, 3):
                    IMUrotAcc[:, j] = np.gradient(IMUrotVel[:, j])


        #filter contact data
        if FT is not None:
            for ft in FT:
                # median filter
                ft_orig = ft.copy()
                for j in range(0, 3):
                    ft[:, j] = sp.signal.medfilt(ft_orig[:, j], median_kernel_size)

                # low-pass filter
                ft_orig = ft.copy()
                for j in range(0, 3):
                    ft[:, j] = sp.signal.filtfilt(b_3, a_3, ft_orig[:, j])
