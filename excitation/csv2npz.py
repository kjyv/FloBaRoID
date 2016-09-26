#!/usr/bin/env python3
#-*- coding: utf-8 -*-

#read data from Przemek's Walk-Man walking csv files
# (atm only one leg)

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
import os
import argparse
import numpy as np
import numpy.linalg as la

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) >= LooseVersion('1.5'):
    plt.style.use('seaborn-muted')

import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from identification.data import Data
from identification.model import Model

sampleTime = 0.005   #200 Hz

def readCSV(dir, config, plot):
    out = {}
    jointNames = ['R-HIP_R', 'R-HIP_Y', 'R-HIP_P', 'R-KNEE', 'R-ANK_P', 'R-ANK_R', 'L-HIP_R',
                  'L-HIP_Y', 'L-HIP_P', 'L-KNEE', 'L-ANK_P', 'L-ANK_R', 'WaistLat', 'WaistSag',
                  'WaistYaw', 'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1',
                  'LWrj2', 'NeckYawj', 'NeckPitchj', 'RShSag', 'RShLat', 'RShYaw', 'RElbj',
                  'RForearmPlate', 'RWrj1', 'RWrj2']

    file = os.path.join(dir, 'jointLog.csv')     #joint positions and torques
    f = np.loadtxt(file)
    out['positions'] = np.empty( (f.shape[0], config['N_DOFS']) )
    out['torques'] = np.empty( (f.shape[0], config['N_DOFS']) )
    out['times'] = np.empty( f.shape[0] )
    out['times'][:] = np.arange(0, f.shape[0])*sampleTime
    out['target_positions'] = np.empty( (f.shape[0], config['N_DOFS']) )

    # generate some empty arrays, will be calculated in preprocess()
    out['velocities'] = np.zeros_like(out['positions'])
    out['accelerations'] = np.zeros_like(out['positions'])

    fig = plt.figure()
    ax1 = fig.add_subplot(3,2,1) # three rows, two columns, first plot
    ax2 = fig.add_subplot(3,2,2)
    dofs_file = len(f[1])//6
    for dof in range(config['N_DOFS']):
        out['target_positions'][:, dof] = f[:, dof]   #position reference
        out['positions'][:, dof] = f[:, dof+dofs_file*2]   #motor encoders
        out['torques'][:, dof] = f[:, dof+dofs_file*4]   #torque sensors
        ax1.plot(out['times'], out['positions'][:, dof], label=jointNames[dof])
        ax2.plot(out['times'], out['torques'][:, dof], label=jointNames[dof])

    file = os.path.join(dir, 'feedbackData.csv')   #force torque and IMU
    f = np.loadtxt(file)
    out['FTright'] = np.empty( (f.shape[0], 6) )   #FT right foot, 3 force, 3 torque values
    out['IMUrpy'] = np.empty( (f.shape[0], 3) )    #IMU orientation, r,p,y
    out['IMUlinAcc'] = np.empty( (f.shape[0], 3) ) #IMU linear acceleration
    out['IMUlinAcc2'] = np.empty( (f.shape[0], 3) ) #IMU linear acceleration
    out['IMUrotVel'] = np.empty( (f.shape[0], 3) ) #IMU rotational velocity

    ax3 = fig.add_subplot(3,2,3)
    ax4 = fig.add_subplot(3,2,4)
    rpy_labels = ['r','p', 'y']
    acc_labels = ['x', 'y', 'z']
    for i in range(0,3):
        #use IMUrpy
        """
        #hw
        #out['IMUrpy'][:, i] = f[:, i]
        out['IMUlinAcc'][:, i] = f[:, 18+i]
        out['IMUrotVel'][:, i] = f[:, 21+i]
        #use VNrpy
        out['IMUrpy'][:, i] = f[:, 15+i]
        out['IMUlinAcc2'][:, i] = f[:, 24+i]
        #out['IMUrotVel'][:, i] = f[:, 27+i]
        """
        #sim
        out['IMUrpy'][:, i] = f[:, i]
        out['IMUlinAcc'][:, i] = f[:, 18+i]
        out['IMUrotVel'][:, i] = f[:, 21+i]

        #rotate VNrpy vals to robot frame (it's built in rotated like this)
        #hm, should really be pi, 0, 0 according to Przemek
        robotToIMU = iDynTree.Rotation.RPY(0, 0, np.pi).toNumPy()

        """
        #hw
        for j in range(0, out['IMUlinAcc2'].shape[0]):
            out['IMUlinAcc2'][j, :] = robotToIMU.dot(out['IMUlinAcc2'][j, :])
        """

        #for j in range(0, out['IMUrotVel'].shape[0]):
        #    out['IMUrotVel'][j, :] = robotToIMU.dot(out['IMUrotVel'][j, :])

        #for j in range(0, out['IMUrpy'].shape[0]):
        #    out['IMUrpy'][j, :] = robotToIMU.dot(out['IMUrpy'][j, :])

        if 0:
            #use other IMU and take average
            out['IMUlinAcc'][:, i] *= (9.81/1.2)   #scaling shouldn't be necessary...
            out['IMUlinAcc'] = np.mean([out['IMUlinAcc'], out['IMUlinAcc2']], axis=0)
        else:
            out['IMUlinAcc'] = out['IMUlinAcc2']

        ax3.plot(out['times'], out['IMUrpy'][:, i], label=rpy_labels[i])
        #ax3.plot(out['times'], out['IMUlinAcc2'][:, i], label=acc_labels[i])
        ax4.plot(out['times'], out['IMUlinAcc'][:, i], label=acc_labels[i])

    ax5 = fig.add_subplot(3,2,5)
    ft_labels = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z']
    for i in range(0,6):
        out['FTright'][:, i] = f[:, 3+6+i]
        ax5.plot(out['times'], out['FTright'][:, i], label=ft_labels[i])

    #set titles and enable legends for each subplot
    t = ['positions', 'torques', 'IMU rpy', 'IMU acc', 'FT right']
    for i in range(0,5):
        plt.subplot(321+i)
        eval('ax{}'.format(1+i)).legend(loc='best', fancybox=True, fontsize=10, title='')
        plt.title(t[i])

    fig.tight_layout()

    if plot:
        plt.show()

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load measurements from csv and write as npz.')
    parser.add_argument('--config', required=True, type=str, help="use options from given config file")
    parser.add_argument('--measurements', required=True, type=str, help='the directory to load the measurements from')
    parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--outfile', type=str, help='the filename to save the measurements to')
    parser.add_argument('--plot', help='whether to plot the data', action='store_true')
    parser.set_defaults(outfile='measurements.npz', plot=False)
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config['N_DOFS'] = 6   #walkman leg
    config['useDeg'] = False
    config['model'] = args.model

    out = readCSV(args.measurements, config, args.plot)
    out['frequency'] = 1.0/sampleTime
    data = Data(config)

    #init empty arrays for preprocess
    out['torques_raw'] = np.empty_like( out['torques'])
    out['IMUlinVel'] = np.empty( (out['times'].shape[0], 3) ) #IMU linear velocity
    out['IMUrotAcc'] = np.empty( (out['times'].shape[0], 3) ) #IMU rotational acceleration

    #filter, diff, integrate
    data.preprocess(Q=out['positions'], V=out['velocities'], Vdot=out['accelerations'],
                    Tau=out['torques'], Tau_raw=out['torques_raw'], T=out['times'],
                    Fs=out['frequency'], IMUlinVel=out['IMUlinVel'], IMUrotVel=out['IMUrotVel'],
                    IMUlinAcc=out['IMUlinAcc'], IMUrotAcc=out['IMUrotAcc'], IMUrpy=out['IMUrpy'],
                    FT=[out['FTright']])

    if args.plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(out['times'], out['IMUlinAcc'])
        plt.subplot(211)
        plt.title("linear accelerations (w/o gravity)")
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(out['times'], out['IMUlinVel'])
        plt.subplot(212)
        plt.title("linear velocities")
        fig.tight_layout()
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(out['times'], out['IMUrotVel'])
        plt.subplot(211)
        plt.title("rotational velocities")
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(out['times'], out['IMUrotAcc'])
        plt.subplot(212)
        plt.title("rotational accelerations")
        fig.tight_layout()
        plt.show()

    out['base_velocity'] = np.hstack( (out['IMUlinVel'], out['IMUrotVel']) )
    out['base_acceleration'] = np.hstack( (out['IMUlinAcc'], out['IMUrotAcc']) )
    out['contacts'] = np.array({'r_leg_ft': out['FTright']})
    out['base_rpy'] = out['IMUrpy']

    if config['excitationSimulate']:
        #use all data
        old_skip = config['skip_samples']
        config['skip_samples'] = 0
        old_offset = config['start_offset']
        config['start_offset'] = 0
        old_sim = config['simulateTorques']
        config['simulateTorques'] = 1
        data.init_from_data(out)
        model = Model(config, config['model'])
        model.computeRegressors(data)
        out['torques'] = out['torques_raw'] = model.data.samples['torques']
        config['skip_samples'] = old_skip
        config['start_offset'] = old_offset
        config['simulateTorques'] = old_sim

    np.savez(args.outfile, positions=out['positions'], positions_raw=out['positions'],
             velocities=out['velocities'], velocities_raw=out['velocities'],
             accelerations=out['accelerations'], torques=out['torques'],
             torques_raw=out['torques_raw'], base_velocity=out['base_velocity'],
             base_acceleration=out['base_acceleration'], base_rpy=out['base_rpy'],
             contacts=out['contacts'], times=out['times'], frequency=out['frequency'])
    print("Saved csv data as {}".format(args.outfile))
    print("Samples: {}, Time: {}s, Frequency: {} Hz".format(out['times'].shape[0], out['times'][-1], out['frequency']))
