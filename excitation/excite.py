#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from IPython import embed

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from robotCommunication import yarp_gym, ros_moveit

import argparse
parser = argparse.ArgumentParser(description='Generate an excitation and record measurements to <filename>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--filename', type=str, help='the filename to save the measurements to')
parser.add_argument('--periods', type=int, help='how many periods to run the trajectory')
parser.add_argument('--plot', help='plot measured data', action='store_true')
parser.add_argument('--dryrun', help="don't send the trajectory", action='store_true')
parser.add_argument('--simulate', help="simulate torques for measured values (e.g. for gazebo)", action='store_true')
parser.add_argument('--yarp', help="use yarp for robot communication", action='store_true')
parser.add_argument('--ros', help="use ros for robot communication", action='store_true')
parser.add_argument('--random-colors', dest='random_colors', help="use random colors for graphs", action='store_true')
parser.add_argument('--plot-targets', dest='plot_targets', help="plot targets instead of measurements", action='store_true')
parser.set_defaults(plot=False, dryrun=False, simulate=False, random_colors=False, filename='measurements.npz', periods=1)
args = parser.parse_args()

data = {}   #hold some global data vars in here
config = {}
config['args'] = args

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
config['jointNames'] = iDynTree.StringVector([])
iDynTree.dofsListFromURDF(args.model, config['jointNames'])
config['N_DOFS'] = len(config['jointNames'])

def postprocess(posis, posis_unfiltered, vels, vels_unfiltered, vels_self,
                accls, torques, torques_unfiltered, times, fs):
    # convert degs to rads
    # assuming angles don't wrap, otherwise use np.unwrap before
    if args.yarp:
        posis_rad = np.deg2rad(posis)
        vels_rad = np.deg2rad(vels)
        np.copyto(posis, posis_rad)
        np.copyto(vels, vels_rad)

    # low-pass filter positions
    posis_orig = posis.copy()
    fc = 3.0    #Cut-off frequency (Hz)
    order = 6   #Filter order
    b, a = sp.signal.butter(order, fc / (fs/2), btype='low', analog=False)
    for j in range(0, config['N_DOFS']):
        posis[:, j] = sp.signal.filtfilt(b, a, posis_orig[:, j])
    np.copyto(posis_unfiltered, posis_orig)

    # median filter of gazebo velocities
    vels_orig = vels.copy()
    for j in range(0, config['N_DOFS']):
        vels[:, j] = sp.signal.medfilt(vels_orig[:, j], 21)
    np.copyto(vels_unfiltered, vels_orig)

    # low-pass filter velocities
    vels_orig = vels.copy()
    fc = 2.0   #Cut-off frequency (Hz)
    order = 6  #Filter order
    b, a = sp.signal.butter(order, fc / (fs/2), btype='low', analog=False)
    for j in range(0, config['N_DOFS']):
        vels[:, j] = sp.signal.filtfilt(b, a, vels_orig[:, j])

    # Plot the frequency and phase response of the filter
    """
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

    # calc velocity instead of taking measurements (uses filtered positions,
    # seems better than filtering noisy velocity measurements)
    for i in range(1, posis.shape[0]):
        dT = times[i] - times[i-1]
        if dT != 0:
            vels_self[i] = (posis[i] - posis[i-1])/dT
        else:
            vels_self[i] = vels_self[i-1]

    # median filter of velocities self to remove outliers
    vels_self_orig = vels_self.copy()
    for j in range(0, config['N_DOFS']):
        vels_self[:, j] = sp.signal.medfilt(vels_self_orig[:, j], 9)

    # low-pass filter velocities self
    vels_self_orig = vels_self.copy()
    for j in range(0, config['N_DOFS']):
        vels_self[:, j] = sp.signal.filtfilt(b, a, vels_self_orig[:, j])

    # calc accelerations
    for i in range(1, vels_self.shape[0]):
        dT = times[i] - times[i-1]
        if dT != 0:
            accls[i] = (vels_self[i] - vels_self[i-1])/dT
        else:
            accls[i] = accls[i-1]

    # filtering accelerations not necessary?

    # median filter of accelerations
    accls_orig = accls.copy()
    for j in range(0, config['N_DOFS']):
        accls[:, j] = sp.signal.medfilt(accls_orig[:, j], 11)

    # low-pass filter of accelerations
    accls_orig = accls.copy()
    for j in range(0, config['N_DOFS']):
        accls[:, j] = sp.signal.filtfilt(b, a, accls_orig[:, j])

    # median filter of torques
    torques_orig = torques.copy()
    for j in range(0, config['N_DOFS']):
        torques[:, j] = sp.signal.medfilt(torques_orig[:, j], 11)

    # low-pass of torques
    torques_orig = torques.copy()
    for j in range(0, config['N_DOFS']):
        torques[:, j] = sp.signal.filtfilt(b, a, torques_orig[:, j])
    np.copyto(torques_unfiltered, torques_orig)

def plot(data):
    if args.random_colors:
        from random import sample
        from itertools import permutations

        # get a random color wheel
        Nlines = 200
        color_lvl = 8
        rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
        colors = sample(rgb,Nlines)
        print colors[0:config['N_DOFS']]
    else:
        # set some nice fixed colors
        # TODO: use palette with more than 7 values...
        colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.84705882,  0.        ,  0.1254902 ],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]

    # python measurements
    # reload measurements from this or last run (if run dry)
    measurements = np.load('measurements.npz')
    Q = measurements['positions']
    Qraw = measurements['positions_raw']
    Q_t = measurements['target_positions']
    V = measurements['velocities']
    Vraw = measurements['velocities_raw']
    V_t = measurements['target_velocities']
    dV = measurements['accelerations']
    dV_t = measurements['target_accelerations']
    Tau = measurements['torques']
    TauRaw = measurements['torques_raw']
    T = measurements['times']
    num_samples = measurements['positions'].shape[0]
    print 'loaded {} measurement samples'.format(num_samples)

    print "tracking error per joint:"
    for i in range(0, config['N_DOFS']):
        sse = np.sum((Q[:, i] - Q_t[:, i]) ** 2)
        print "joint {}: {}".format(i, sse)

    print "histogram of time diffs"
    dT = np.diff(T)
    H, B = np.histogram(dT)
    #plt.hist(H, B)
    print "bins: {}".format(B)
    print "sums: {}".format(H)
    late_msgs = (1 - float(np.sum(H)-np.sum(H[1:])) / float(np.sum(H))) * 100
    print "({}% messages too late)".format(late_msgs)
    print "\n"

    # what to plot (each tuple has a title and one or multiple data arrays)
    if args.plot_targets:    #plot target values
        datasets = [
            ([Q_t,], 'Target Positions'),
            ([V_t,], 'Target Velocities'),
            ([dV_t,], 'Target Accelerations')
            ]
    else:   #plot measurements and raw data (from measurements file)
        datasets = [
            ([Q, Qraw], 'Positions'),
            ([V, Vraw],'Velocities'),
            ([dV,], 'Accelerations'),
            ([Tau, TauRaw],'Measured Torques')
            ]

    d = 0
    cols = 2.0
    rows = round(len(datasets)/cols)
    for (data, title) in datasets:
        plt.subplot(rows, cols, d+1)
        plt.title(title)
        lines = list()
        labels = list()
        for d_i in range(0, len(data)):
            for i in range(0, config['N_DOFS']):
                l = config['jointNames'][i] if d_i == 0 else ''  #only put joint names in the legend once
                labels.append(l)
                line = plt.plot(T, data[d_i][:, i], color=colors[i], alpha=1-(d_i/3.0))
                lines.append(line[0])
        d+=1
    leg = plt.figlegend(lines, labels, 'upper right', fancybox=True, fontsize=10)
    leg.draggable()

    plt.show()

def main():
    if args.yarp:
        yarp_gym.main(config, data)
    elif args.ros:
        ros_moveit.main(config, data)

    # generate some empty arrays, will be calculated in preprocess()
    data['Vdot'] = np.zeros_like(data['V'])
    data['Vraw'] = np.zeros_like(data['V'])
    data['Vself'] = np.zeros_like(data['V'])
    data['Qraw'] = np.zeros_like(data['Q'])
    data['TauRaw'] = np.zeros_like(data['Tau'])

    # filter, differentiate, convert, etc.
    postprocess(data['Q'], data['Qraw'], data['V'], data['Vraw'], data['Vself'], data['Vdot'],
                data['Tau'], data['TauRaw'], data['T'], data['measured_frequency'])

    if args.simulate:
        dynComp = iDynTree.DynamicsComputations();
        dynComp.loadRobotModelFromFile(args.model);
        gravity = iDynTree.SpatialAcc();
        gravity.zero()
        gravity.setVal(2, -9.81);
        torques = iDynTree.VectorDynSize(config['N_DOFS'])

        for t in range(0, len(data['T'])):
            pos = data['Q'][t]
            vel = data['Vself'][t]
            acc = data['Vdot'][t]
            q = iDynTree.VectorDynSize.fromPyList(pos)
            dq = iDynTree.VectorDynSize.fromPyList(vel)
            ddq = iDynTree.VectorDynSize.fromPyList(acc)

            dynComp.setRobotState(q, dq, ddq, gravity)
            baseReactionForce = iDynTree.Wrench()   # assume zero for fixed base, otherwise use e.g. imu data

            # compute inverse dynamics with idyntree (simulate)
            dynComp.inverseDynamics(torques, baseReactionForce)
            data['Tau'][t] = torques.toNumPy()

            #add some noise to torques (simulated 'raw' data)
            data['TauRaw'][t] = data['Tau'][t] + np.random.randn(config['N_DOFS'])*np.max(data['Tau'])*0.05

    # write sample arrays to data file
    # TODO: if possible, save motor currents
    np.savez(args.filename,
             positions=data['Q'], positions_raw=data['Qraw'],
             velocities=data['Vself'], velocities_raw=data['Vraw'],
             accelerations=data['Vdot'],
             torques=data['Tau'], torques_raw=data['TauRaw'],
             target_positions=np.deg2rad(data['Qsent']), target_velocities=np.deg2rad(data['QdotSent']),
             target_accelerations=np.deg2rad(data['QddotSent']),
             times=data['T'], frequency=data['measured_frequency'])
    print "saved measurements to {}".format(args.filename)


if __name__ == '__main__':
    try:
        if not args.dryrun:
            main()
        if(args.plot):
            plot(data)
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            # open ipdb when an exception happens
            import sys, ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)

