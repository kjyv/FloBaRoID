#!/usr/bin/env python2.7
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy import sparse

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import argparse
parser = argparse.ArgumentParser(description='Generate an excitation and record measurements to <filename>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--filename', type=str, help='the filename to save the measurements to')
parser.add_argument('--config', required=True, type=str, help="use options from given config file")
parser.add_argument('--dryrun', help="don't send the trajectory", action='store_true')

parser.add_argument('--periods', type=int, help='how many periods to run the trajectory')
parser.add_argument('--plot', help='plot measured data', action='store_true')
parser.add_argument('--plot-targets', dest='plot_targets', help="plot targets instead of measurements", action='store_true')
parser.set_defaults(plot=False, plot_targets=False, dryrun=False, simulate=False, filename='measurements.npz', periods=1)
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
if not iDynTree.dofsListFromURDF(args.model, config['jointNames']):
    sys.exit()
config['N_DOFS'] = len(config['jointNames'])
config['useAPriori'] = 0
config['skip_samples'] = 0

traj_data = {}   #hold some global data vars in here

#append parent dir for relative import
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from identification.model import Model
from identification.data import Data

from excitation.trajectoryGenerator import TrajectoryGenerator, TrajectoryOptimizer

def plot(data=None):
    fig = plt.figure(1)
    fig.clear()
    if False:
        from random import sample
        from itertools import permutations

        # get a random color wheel
        Nlines = 200
        color_lvl = 8
        rgb = np.array(list(permutations(list(range(0,256,color_lvl),3))))/255.0
        colors = sample(rgb,Nlines)
        print(colors[0:config['N_DOFS']])
    else:
        # set some fixed colors
        colors = []
        """colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.84705882,  0.        ,  0.1254902 ],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]
        """
        from palettable.tableau import Tableau_10, Tableau_20
        colors += Tableau_10.mpl_colors[0:6] + Tableau_20.mpl_colors

    if not data:
        # python measurements
        # reload measurements from this or last run (if run dry)
        measurements = np.load('measurements.npz')
        Q = measurements['positions']
        Qraw = measurements['positions_raw']
        V = measurements['velocities']
        Vraw = measurements['velocities_raw']
        dV = measurements['accelerations']
        Tau = measurements['torques']
        TauRaw = measurements['torques_raw']
        if args.plot_targets:
            Q_t = measurements['target_positions']
            V_t = measurements['target_velocities']
            dV_t = measurements['target_accelerations']
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

    print('loaded {} measurement samples'.format(num_samples))

    if args.plot_targets:
        print("tracking error per joint:")
        for i in range(0, config['N_DOFS']):
            sse = np.sum((Q[:, i] - Q_t[:, i]) ** 2)
            print("joint {}: {}".format(i, sse))

    print("histogram of time diffs")
    dT = np.diff(T)
    H, B = np.histogram(dT)
    #plt.hist(H, B)
    late_msgs = (1 - float(np.sum(H)-np.sum(H[1:])) / float(np.sum(H))) * 100
    print("bins: {}".format(B))
    print("sums: {}".format(H))
    print("({}% messages too late)\n".format(late_msgs))

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

def simulateTrajectory(config, trajectory, model=None, measurements=None):
    # generate data arrays for simulation and regressor building
    old_sim = config['simulateTorques']
    config['simulateTorques'] = True

    if not model:
        model = Model(config, config['model'])

    data = Data(config)
    trajectory_data = {}
    trajectory_data['target_positions'] = []
    trajectory_data['target_velocities'] = []
    trajectory_data['target_accelerations'] = []
    trajectory_data['torques'] = []
    trajectory_data['times'] = []
    freq=200.0
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

        trajectory_data['times'].append(t/freq)
        trajectory_data['torques'].append(np.zeros(config['N_DOFS']))

    num_samples = len(trajectory_data['times'])

    trajectory_data['target_positions'] = np.array(trajectory_data['target_positions'])
    trajectory_data['positions'] = trajectory_data['target_positions']
    trajectory_data['target_velocities'] = np.array(trajectory_data['target_velocities'])
    trajectory_data['velocities'] = trajectory_data['target_velocities']
    trajectory_data['target_accelerations'] = np.array(trajectory_data['target_accelerations'])
    trajectory_data['accelerations'] = trajectory_data['target_accelerations']
    trajectory_data['torques'] = np.array(trajectory_data['torques'])
    trajectory_data['times'] = np.array(trajectory_data['times'])
    trajectory_data['measured_frequency'] = freq
    trajectory_data['base_velocity'] = np.zeros( (num_samples, 6) )
    trajectory_data['base_acceleration'] = np.zeros( (num_samples, 6) )
    trajectory_data['base_rpy'] = np.zeros( (num_samples, 3) )
    trajectory_data['contacts'] = np.array({'dummy_sim': np.zeros( num_samples )})

    if measurements:
        trajectory_data['positions'] = measurements['Q']
        trajectory_data['velocities'] = measurements['V']
        trajectory_data['accelerations'] = measurements['Vdot']
        trajectory_data['measured_frequency'] = measurements['measured_frequency']


    old_skip = config['skip_samples']
    config['skip_samples'] = 0
    old_offset = config['start_offset']
    config['start_offset'] = 0
    data.init_from_data(trajectory_data)
    model.computeRegressors(data) #TODO: this needlessly also computes regressors in addition to simulation
    config['skip_samples'] = old_skip
    config['start_offset'] = old_offset
    config['simulateTorques'] = old_sim

    return trajectory_data, data, model

def main():
    traj_file = config['model'] + '.trajectory.npz'
    if config['optimizeTrajectory']:
        trajectoryOptimizer = TrajectoryOptimizer(config, simulation_func=simulateTrajectory)
        if config['showOptimizationTrajs']:
            trajectory = trajectoryOptimizer.optimizeTrajectory(plot_func=plot)
        else:
            trajectory = trajectoryOptimizer.optimizeTrajectory()
        np.savez(traj_file, use_deg=trajectory.use_deg, a=trajectory.a, b=trajectory.b,
                 q=trajectory.q, nf=trajectory.nf, wf=trajectory.w_f_global)
    else:
        try:
            tf = np.load(traj_file)
            trajectory = TrajectoryGenerator(config['N_DOFS'], use_deg=tf['use_deg'])
            trajectory.initWithParams(tf['a'], tf['b'], tf['q'], tf['nf'], tf['wf'])
        except IOError:
            trajectory = TrajectoryGenerator(config['N_DOFS']).initWithRandomParams()
            print("a {}".format([t_a.tolist() for t_a in trajectory.a]))
            print("b {}".format([t_b.tolist() for t_b in trajectory.b]))
            print("q {}".format(trajectory.q.tolist()))
            print("nf {}".format(trajectory.nf.tolist()))
            print("wf {}".format(trajectory.w_f_global))

    traj_data, data, model = simulateTrajectory(config, trajectory)

    if config['exciteMethod'] == 'yarp':
        from robotCommunication import yarp_gym
        yarp_gym.main(config, trajectory, traj_data)
    elif config['exciteMethod'] == 'ros':
        from robotCommunication import ros_moveit
        ros_moveit.main(config, trajectory, traj_data, move_group="full_lwr")
    else:
        print("No excitation method given! Only doing simulation")
        saveMeasurements(args.filename, traj_data)
        return

    # generate some empty arrays, will be calculated in preprocess()
    if 'Vdot' not in traj_data:
        traj_data['Vdot'] = np.zeros_like(traj_data['V'])
    traj_data['Vraw'] = np.zeros_like(traj_data['V'])
    traj_data['Qraw'] = np.zeros_like(traj_data['Q'])
    traj_data['TauRaw'] = np.zeros_like(traj_data['Tau'])

    #use simulated torques as measured data (since e.g. Gazebo produces unusable torque values)
    if config['excitationSimulate']:
        tau_len = traj_data['Tau'].shape[0]   # get length of measured (zero) taus
        if tau_len < traj_data['torques'].shape[0]:
            traj_data['Tau'][:,:] = traj_data['torques'][0:tau_len,:]
            if config['exciteMethod'] == None:
                traj_data['V'][:,:] = traj_data['velocities'][0:tau_len,:]
        else:
            torques_len = traj_data['torques'].shape[0]
            traj_data['Tau'][0:torques_len,:]  = traj_data['torques'][:,:]
            if config['exciteMethod'] == None:
                traj_data['V'][0:torques_len,:] = traj_data['velocities'][:,:]

    # filter, differentiate, convert, etc.
    data.preprocess(Q=traj_data['Q'], Q_raw=traj_data['Qraw'], V=traj_data['V'],
                    V_raw=traj_data['Vraw'], Vdot=traj_data['Vdot'], Tau=traj_data['Tau'],
                    Tau_raw = traj_data['TauRaw'], T=traj_data['T'], Fs=traj_data['measured_frequency'])

    #simulate again with measured/filtered data
    if config['excitationSimulate']:
        traj_data_sim, data, model = simulateTrajectory(config, trajectory, measurements=traj_data)
        traj_data['Tau'] = traj_data_sim['torques'][0:traj_data['Tau'].shape[0]]

    saveMeasurements(args.filename, traj_data)

def saveMeasurements(filename, data):
    # write sample arrays to data file
    # TODO: if possible, save motor currents
    if config['exciteMethod']:
        np.savez(filename,
                 positions=data['Q'], positions_raw=data['Qraw'],
                 velocities=data['V'], velocities_raw=data['Vraw'],
                 accelerations=data['Vdot'],
                 torques=data['Tau'], torques_raw=data['TauRaw'],
                 target_positions=np.deg2rad(data['Qsent']), target_velocities=np.deg2rad(data['QdotSent']),
                 target_accelerations=np.deg2rad(data['QddotSent']),
                 base_velocity=data['base_velocity'], base_acceleration=data['base_acceleration'],
                 base_rpy=data['base_rpy'], contacts=data['contacts'],
                 times=data['T'], frequency=data['measured_frequency'])
    else:
        np.savez(filename,
                 positions=data['positions'], positions_raw=data['positions'],
                 velocities=data['velocities'], velocities_raw=data['velocities'],
                 accelerations=data['accelerations'],
                 torques=data['torques'], torques_raw=data['torques'],
                 target_positions=np.deg2rad(data['target_positions']),
                 target_velocities=np.deg2rad(data['target_velocities']),
                 target_accelerations=np.deg2rad(data['target_accelerations']),
                 base_velocity=data['base_velocity'], base_acceleration=data['base_acceleration'],
                 base_rpy=data['base_rpy'], contacts=data['contacts'],
                 times=data['times'], frequency=data['measured_frequency'])

    print("saved measurements to {}".format(args.filename))

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
