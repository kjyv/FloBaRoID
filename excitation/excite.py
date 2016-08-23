#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy import sparse
from IPython import embed

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import argparse
parser = argparse.ArgumentParser(description='Generate an excitation and record measurements to <filename>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--filename', type=str, help='the filename to save the measurements to')
parser.add_argument('--config', required=True, type=str, help="use options from given config file")
parser.add_argument('--dryrun', help="don't send the trajectory", action='store_true')

parser.add_argument('--periods', type=int, help='how many periods to run the trajectory')
parser.add_argument('--plot', help='plot measured data', action='store_true')
parser.add_argument('--random-colors', dest='random_colors', help="use random colors for graphs", action='store_true')
parser.add_argument('--plot-targets', dest='plot_targets', help="plot targets instead of measurements", action='store_true')
parser.set_defaults(plot=False, dryrun=False, simulate=False, random_colors=False, filename='measurements.npz', periods=1)
args = parser.parse_args()

import yaml
with open(args.config, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config['args'] = args
config['model'] = args.model
config['jointNames'] = iDynTree.StringVector([])
iDynTree.dofsListFromURDF(args.model, config['jointNames'])
config['N_DOFS'] = len(config['jointNames'])
config['useAPriori'] = 0
config['skip_samples'] = 0

data = {}   #hold some global data vars in here

#append parent dir for relative import
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from identification.identification import Identification
from identification.model import Model
from identification.data import Data

from trajectoryGenerator import TrajectoryGenerator, TrajectoryOptimizer

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
    # TODO: Khalil, p.299 suggests a "central difference" method to avoid phase shift:
    # dq(k) = [q(k+1)-q(k-1)]/2T
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

def plot(data=None):
    fig = plt.figure(1)
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

    if not data:
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
    else:
        Q = data['positions']
        Qraw = data['positions']
        Q_t = data['target_positions']
        V = data['velocities']
        Vraw = data['velocities']
        V_t = data['target_velocities']
        dV = data['accelerations']
        dV_t = data['target_accelerations']
        Tau = data['torques']
        TauRaw = data['torques']
        T = data['times']
        num_samples = data['positions'].shape[0]

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
            if len(data[d_i].shape) > 1:
                for i in range(0, config['N_DOFS']):
                    l = config['jointNames'][i] if d_i == 0 else ''  #only put joint names in the legend once
                    labels.append(l)
                    line = plt.plot(T, data[d_i][:, i], color=colors[i], alpha=1-(d_i/2.0))
                    lines.append(line[0])
            else:
                #data vector
                plt.plot(T, data[d_i], label=title, color=colors[0], alpha=1-(d_i/2.0))
        d+=1
    leg = plt.figlegend(lines, labels, 'upper right', fancybox=True, fontsize=10)
    leg.draggable()

    plt.show()

def simulateTrajectory(config, trajectory, model=None):
    #trajectory = TrajectoryGenerator(config['N_DOFS'], use_deg=True)
    #trajectory.initWithRandomParams()

    # generate data arrays for simulation and regressor building
    old_sim = config['iDynSimulate']
    config['iDynSimulate'] = True

    if not model:
        model = Model(config, config['model'])

    data = Data(config)
    trajectory_data = {}
    trajectory_data['target_positions'] = []
    trajectory_data['target_velocities'] = []
    trajectory_data['target_accelerations'] = []
    trajectory_data['torques'] = []
    trajectory_data['times'] = []
    freq = 200.0
    for t in range(0, int(trajectory.getPeriodLength()*freq)):
        trajectory.setTime(t/freq)
        q = [trajectory.getAngle(d) for d in range(config['N_DOFS'])]
        q = np.array(q)
        if config['useDeg']:
            q = np.deg2rad(q)
        trajectory_data['target_positions'].append(q)

        qdot = [trajectory.getVelocity(d) for d in range(config['N_DOFS'])]
        qdot = np.array(qdot)
        if config['useDeg']:
            qdot = np.deg2rad(qdot)
        trajectory_data['target_velocities'].append(qdot)

        qddot = [trajectory.getAcceleration(d) for d in range(config['N_DOFS'])]
        qddot = np.array(qddot)
        if config['useDeg']:
            qddot = np.deg2rad(qddot)
        trajectory_data['target_accelerations'].append(qddot)

        trajectory_data['times'].append(t)
        trajectory_data['torques'].append(np.zeros(config['N_DOFS']))

    trajectory_data['target_positions'] = np.array(trajectory_data['target_positions'])
    trajectory_data['positions'] = trajectory_data['target_positions']
    trajectory_data['target_velocities'] = np.array(trajectory_data['target_velocities'])
    trajectory_data['velocities'] = trajectory_data['target_velocities']
    trajectory_data['target_accelerations'] = np.array(trajectory_data['target_accelerations'])
    trajectory_data['accelerations'] = trajectory_data['target_accelerations']
    trajectory_data['torques'] = np.array(trajectory_data['torques'])
    trajectory_data['times'] = np.array(trajectory_data['times'])
    trajectory_data['measured_frequency'] = freq

    data.init_from_data(trajectory_data)
    model.computeRegressors(data)

    config['iDynSimulate'] = old_sim

    return trajectory_data, model

def main():
    trajectoryOptimizer = TrajectoryOptimizer(config, simulation_func=simulateTrajectory)
    trajectory = trajectoryOptimizer.optimizeTrajectory()
    data, model = simulateTrajectory(config, trajectory)

    if config['exciteMethod'] == 'yarp':
        from robotCommunication import yarp_gym
        yarp_gym.main(config, trajectory, data)
    elif config['exciteMethod'] == 'ros':
        from robotCommunication import ros_moveit
        ros_moveit.main(config, trajectory, data, move_group="full_lwr")
    else:
        print("No excitation method given! Only doing simulation")
        saveMeasurements(args.filename, data)
        plot(data)
        return

    # generate some empty arrays, will be calculated in preprocess()
    if not data.has_key('Vdot'):
        data['Vdot'] = np.zeros_like(data['V'])
    data['Vraw'] = np.zeros_like(data['V'])
    data['Vself'] = np.zeros_like(data['V'])
    data['Qraw'] = np.zeros_like(data['Q'])
    data['TauRaw'] = np.zeros_like(data['Tau'])

    #simulate torque for measured data (since e.g. Gazebo produces unusable torque values)
    if config['excitationSimulate']:
        tau_len = data['Tau'].shape[0]   # get length of measured (zero) taus
        if tau_len < data['torques'].shape[0]:
            data['Tau'][:,:]  = data['torques'][0:tau_len,:]
            data['Vself'][:,:] = data['velocities'][0:tau_len,:]
            data['Vdot'][:,:] = data['accelerations'][0:tau_len,:]
        else:
            torques_len = data['torques'].shape[0]
            data['Tau'][0:torques_len,:]  = data['torques'][:,:]
            data['Vself'][0:torques_len,:] = data['velocities'][:,:]
            data['Vdot'][0:torques_len,:] = data['accelerations'][:,:]
    else:
        # filter, differentiate, convert, etc.
        postprocess(data['Q'], data['Qraw'], data['V'], data['Vraw'], data['Vself'], data['Vdot'],
                    data['Tau'], data['TauRaw'], data['T'], data['measured_frequency'])

    saveMeasurements(args.filename, data)

def saveMeasurements(filename, data):
    # write sample arrays to data file
    # TODO: if possible, save motor currents
    if config['exciteMethod']:
        np.savez(filename,
                 positions=data['Q'], positions_raw=data['Qraw'],
                 velocities=data['Vself'], velocities_raw=data['Vraw'],
                 accelerations=data['Vdot'],
                 torques=data['Tau'], torques_raw=data['TauRaw'],
                 target_positions=np.deg2rad(data['Qsent']), target_velocities=np.deg2rad(data['QdotSent']),
                 target_accelerations=np.deg2rad(data['QddotSent']),
                 times=data['T'], frequency=data['measured_frequency'])
    else:
        np.savez(filename,
                 positions=data['positions'], positions_raw=data['positions'],
                 velocities=data['velocities'], velocities_raw=data['velocities'],
                 accelerations=data['accelerations'],
                 torques=data['torques'], torques_raw=data['torques'],
                 target_positions=np.deg2rad(data['target_positions']), target_velocities=np.deg2rad(data['target_velocities']),
                 target_accelerations=np.deg2rad(data['target_accelerations']),
                 times=data['times'], frequency=data['measured_frequency'])

    print "saved measurements to {}".format(args.filename)

if __name__ == '__main__':
    #try:
    if not args.dryrun:
        main()
    if(args.plot):
        plot()
    """
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            # open ipdb when an exception happens
            import sys, ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
    """
