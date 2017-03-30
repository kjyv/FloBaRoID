#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
from typing import Dict

import numpy as np
from colorama import Fore, Back, Style
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import argparse
parser = argparse.ArgumentParser(description='Send an excitation trajectory and record measurements to <filename>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--filename', type=str, help='the filename to save the measurements to')
parser.add_argument('--trajectory', type=str, help='the file to load the trajectory from')
parser.add_argument('--config', required=True, type=str, help="use options from given config file")
parser.add_argument('--dryrun', help="don't actually send the trajectory", action='store_true')

parser.add_argument('--plot', help='plot measured data', action='store_true')
parser.add_argument('--plot-targets', dest='plot_targets', help="plot targets instead of measurements", action='store_true')
parser.set_defaults(plot=False, plot_targets=False, dryrun=False, filename='measurements.npz')
args = parser.parse_args()

import yaml
with open(args.config, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config['args'] = args
config['urdf'] = args.model
config['plot_targets'] = args.plot_targets
config['jointNames'] = iDynTree.StringVector([])
if not iDynTree.dofsListFromURDF(config['urdf'], config['jointNames']):
    sys.exit()
config['num_dofs'] = len(config['jointNames'])

#append parent dir for relative import
#import os
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

from excitation.optimizer import plotter
from excitation.trajectoryGenerator import PulsedTrajectory, FixedPositionTrajectory, simulateTrajectory

traj_data = {}   # type: Dict[str, np._ArrayLike]   # hold some global data vars in here

def main():
    if args.trajectory:
        traj_file = args.trajectory
    else:
        traj_file = config['urdf'] + '.trajectory.npz'

    #load from file
    try:
        # replay optimized trajectory if found
        tf = np.load(traj_file, encoding='latin1')
        if 'static' in tf and tf['static']:
            # static posture file
            trajectory = FixedPositionTrajectory(config)
            trajectory.initWithAngles(tf['angles'])
            print("using static postures from file {}".format(traj_file))
        else:
            # proper trajectory
            trajectory = PulsedTrajectory(config['num_dofs'], use_deg=tf['use_deg'])
            trajectory.initWithParams(tf['a'], tf['b'], tf['q'], tf['nf'], tf['wf'])
            print("using trajectory from file {}".format(traj_file))
    except IOError:
        print("No trajectory file found, can't excite ({})!".format(traj_file))
        sys.exit(1)

    # generating simulation of trajectory in any case
    traj_data, data = simulateTrajectory(config, trajectory)
    if config['excitationSimulate'] and config['exciteMethod']:
        print(Fore.RED + 'Using simulated torques!' + Fore.RESET)

    if args.dryrun:
        return

    # excite real robot
    if config['exciteMethod'] == 'yarp':
        from excitation.robotCommunication import yarp_gym
        yarp_gym.main(config, trajectory, traj_data)
    elif config['exciteMethod'] == 'ros':
        from excitation.robotCommunication import ros_moveit
        ros_moveit.main(config, trajectory, traj_data)
    else:
        # or just use simulation data
        print("No excitation method given! Only doing simulation")
        saveMeasurements(args.filename, traj_data)
        return

    #adapt measured array sizes to input array sizes
    traj_data['Q'] = np.resize(traj_data['Q'], data.samples['positions'].shape)
    traj_data['V'] = np.resize(traj_data['V'], data.samples['velocities'].shape)
    traj_data['Tau'] = np.resize(traj_data['Tau'], data.samples['torques'].shape)
    traj_data['T'] = np.resize(traj_data['T'], data.samples['times'].shape)

    # generate some empty arrays, will be calculated in preprocess()
    if 'Vdot' not in traj_data:
        traj_data['Vdot'] = np.zeros_like(traj_data['V'])
    traj_data['Vraw'] = np.zeros_like(traj_data['V'])
    traj_data['Qraw'] = np.zeros_like(traj_data['Q'])
    traj_data['TauRaw'] = np.zeros_like(traj_data['Tau'])

    # if simulating torques, prepare some arrays with proper length (needs to be same as input for
    # simulation)
    if config['excitationSimulate']:
        tau_len = traj_data['Tau'].shape[0]   # get length of measured (zero) taus
        if tau_len < traj_data['torques'].shape[0]:
            #less measured samples than input samples
            traj_data['Tau'][:,:] = traj_data['torques'][0:tau_len,:]
            if config['exciteMethod'] == None:
                traj_data['V'][:,:] = traj_data['velocities'][0:tau_len,:]
        else:
            #less or equal input samples than measured samples
            torques_len = traj_data['torques'].shape[0]
            traj_data['Tau'][:torques_len, :]  = traj_data['torques'][:,:]
            if config['exciteMethod'] == None:
                traj_data['V'] = traj_data['velocities'][:,:]

    # filter, differentiate, convert, etc.
    data.preprocess(Q=traj_data['Q'], Q_raw=traj_data['Qraw'], V=traj_data['V'],
                    V_raw=traj_data['Vraw'], Vdot=traj_data['Vdot'], Tau=traj_data['Tau'],
                    Tau_raw = traj_data['TauRaw'], T=traj_data['T'], Fs=traj_data['measured_frequency'])

    # use simulated torques as measured data (since e.g. Gazebo produces unusable torque values)
    # (simulate again with measured/filtered data)
    if config['excitationSimulate']:
        traj_data_sim, data = simulateTrajectory(config, trajectory, measurements=traj_data)
        traj_data['Tau'] = data.samples['torques']

    saveMeasurements(args.filename, traj_data)

def saveMeasurements(filename, data):
    # write sample arrays to data file
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
    main()
    if(args.plot):
        plotter(config, filename=args.filename)
