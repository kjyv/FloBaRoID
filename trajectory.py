#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
import sys

import numpy as np
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from identify import Identification
from identification.model import Model
from excitation.trajectoryGenerator import PulsedTrajectory
from excitation.trajectoryOptimizer import TrajectoryOptimizer, simulateTrajectory
from excitation.postureOptimizer import PostureOptimizer

import argparse
parser = argparse.ArgumentParser(description='Generate excitation trajectories, save to <filename>.')
parser.add_argument('--filename', type=str, help='the filename to save the trajectory to, otherwise <model>.trajectory.npz')
parser.add_argument('--config', required=True, type=str, help="use options from given config file")
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--model_real', required=False, type=str, help='the file to load the "real" robot model from')
parser.add_argument('--world', required=False, type=str, help='the file to load world links from')
args = parser.parse_args()

import yaml
with open(args.config, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config['urdf'] = args.model
config['urdf_real'] = args.model_real
if config['useStaticTrajectories'] and not config['urdf_real']:
    print('When optimizing static postures, need model_real argument!')
    sys.exit()
config['jointNames'] = iDynTree.StringVector([])
if not iDynTree.dofsListFromURDF(config['urdf'], config['jointNames']):
    sys.exit()
config['num_dofs'] = len(config['jointNames'])
config['skipSamples'] = 0

def main():
    # save either optimized or random trajectory parameters to filename
    if args.filename:
        traj_file = args.filename
    else:
        traj_file = config['urdf'] + '.trajectory.npz'

    if config['optimizeTrajectory']:
        # find trajectory params by optimization
        old_sim = config['simulateTorques']
        config['simulateTorques'] = True
        model = Model(config, config['urdf'])
        if config['useStaticTrajectories']:
            old_gravity = config['identifyGravityParamsOnly']
            idf = Identification(config, config['urdf'], config['urdf_real'], measurements_files=None,
                                 regressor_file=None, validation_file=None)
            trajectoryOptimizer = PostureOptimizer(config, idf, model, simulation_func=simulateTrajectory, world=args.world)
            config['identifyGravityParamsOnly'] = old_gravity
        else:
            idf = Identification(config, config['urdf'], urdf_file_real=None, measurements_files=None,
                                 regressor_file=None, validation_file=None)
            trajectoryOptimizer = TrajectoryOptimizer(config, idf, model, simulation_func=simulateTrajectory, world=args.world)

        trajectory = trajectoryOptimizer.optimizeTrajectory()
        config['simulateTorques'] = old_sim
    else:
        # use some random params
        print("no optimized trajectory found, generating random one")
        trajectory = PulsedTrajectory(config['num_dofs'], use_deg=config['useDeg']).initWithRandomParams()
        print("a {}".format([t_a.tolist() for t_a in trajectory.a]))
        print("b {}".format([t_b.tolist() for t_b in trajectory.b]))
        print("q {}".format(trajectory.q.tolist()))
        print("nf {}".format(trajectory.nf.tolist()))
        print("wf {}".format(trajectory.w_f_global))

    print("Saving found trajectory to {}".format(traj_file))

    if config['useStaticTrajectories']:
        # always saved with rad angles
        np.savez(traj_file, static=True, angles=trajectory.angles)
    else:
        # TODO: remove degrees option
        np.savez(traj_file, use_deg=trajectory.use_deg, static=False, a=trajectory.a, b=trajectory.b,
                 q=trajectory.q, nf=trajectory.nf, wf=trajectory.w_f_global)

if __name__ == '__main__':
    main()
