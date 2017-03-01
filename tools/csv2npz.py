#!/usr/bin/env python3
#-*- coding: utf-8 -*-

#read data from different csv data files

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
import os
import argparse
import numpy as np

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) >= LooseVersion('1.5'):
    plt.style.use('seaborn-muted')

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from identification.data import Data
from identification.model import Model

#set to simulate torques with iDynTree and override wrong torques from gazebo
is_gazebo = 1

def readCentauroCSV(path, config, plot):
    #names in order of the supplied data
    jointNames = ['torso_yaw', 'j_arm2_1', 'j_arm2_2', 'j_arm2_3', 'j_arm2_4', 'j_arm2_5',
                  'j_arm2_6', 'j_arm2_7', 'j_arm1_1', 'j_arm1_2', 'j_arm1_3', 'j_arm1_4', 'j_arm1_5',
                  'j_arm1_6', 'j_arm1_7']
    urdf_jointOrder = [0, 8,9,10,11,12,13,14, 1,2,3,4,5,6,7]

    config['num_dofs'] = len(jointNames)

    out = {}

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1) # three rows, two columns, first plot
        ax2 = fig.add_subplot(2,1,2)

    #read one file per joint
    for dof in urdf_jointOrder:
        filepath = os.path.join(path, 'CentAcESC_{}_log.txt'.format(dof+1))
        f = np.loadtxt(filepath)

        if dof == 0:
            # generate some empty arrays, will be calculated in preprocess()
            out['positions'] = np.zeros( (f.shape[0], config['num_dofs']) )
            out['target_positions'] = np.zeros( (f.shape[0], config['num_dofs']) )
            out['torques'] = np.zeros( (f.shape[0], config['num_dofs']) )
#            out['currents'] = np.zeros( (f.shape[0], config['num_dofs']) )
            out['velocities'] = np.zeros_like(out['positions'])
            out['accelerations'] = np.zeros_like(out['positions'])
            out['times'] = np.zeros( f.shape[0] )
            #out['times'][:] = np.arange(0, f.shape[0])*sampleTime
            out['times'][:] = f[:, 0]/1e9

        #read data
        out['target_positions'][:, dof] = f[:, 17]       #position reference
        out['positions'][:, dof] = f[:, 8]  #link encoders
        out['torques'][:, dof] = f[:, 12]    #torque sensors

        if plot:
            ax1.plot(out['times'][::4], out['positions'][::4, dof], label=jointNames[dof])
            ax2.plot(out['times'][::4], out['torques'][::4, dof], label=jointNames[dof])

    if plot:
        #set titles and enable legends for each subplot
        t = ['positions', 'torques']
        for i in range(0,2):
            plt.subplot(211+i)
            eval('ax{}'.format(1+i)).legend(fancybox=True, fontsize=10, title='')
            plt.title(t[i])

        fig.tight_layout()
        plt.show()

    return out

def readWalkmanCSV(path, config, plot):
    out = {}
    #field order in csv file
    jointNames = ['R-HIP_R', 'R-HIP_Y', 'R-HIP_P', 'R-KNEE', 'R-ANK_P', 'R-ANK_R', 'L-HIP_R',
                  'L-HIP_Y', 'L-HIP_P', 'L-KNEE', 'L-ANK_P', 'L-ANK_R',
                  'WaistLat', 'WaistSag', 'WaistYaw',
                  'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1',
                  'LWrj2', 'NeckYawj', 'NeckPitchj', 'RShSag', 'RShLat', 'RShYaw', 'RElbj',
                  'RForearmPlate', 'RWrj1', 'RWrj2']
    #fields to leave out
    # len(jointNames) - len(ignoreJoints) needs to match DOFs that are defined in regressor xml or
    # are found automatically!
    #ignoreJoints = []

    # walkman with some fixed joints (not included in regressor)
    ignoreJoints = [jointNames.index('WaistLat'), jointNames.index('NeckYawj'), jointNames.index('NeckPitchj')]

    # idyntree urdf joint order (generator and dynComp class)
    # LHipLat, LHipYaw, LHipSag, LKneeSag, LAnkSag, LAnkLat,
    # RHipLat, RHipYaw, RHipSag, RKneeSag, RAnkSag, RAnkLat,
    # WaistLat, WaistSag, WaistYaw,
    # LShSag, LShLat, LShYaw, LElbj, LForearmPlate, LWrj1, LWrj2,
    # NeckYawj, NeckPitchj,
    # RShSag, RShLat, RShYaw, RElbj, RForearmPlate, RWrj1, RWrj2
    # mapping to urdf joints in indices:
    csv_T_urdf_indices = [6, 7, 8, 9, 10, 11,  0, 1, 2, 3, 4, 5,
                          12, 13, 14,  15, 16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    ## apply some data correction, should be temporary

    """
    joint_signs = np.array([1, -1, 1, 1, -1, 1,       #LHipLat -
                            -1, -1, -1, -1, 1, 1,      #RHipLat -
                            1, 1,                     #WaistSag -
                            1, -1, -1, 1, 1, -1, -1,  #LShSag -
                            1, -1, -1, -1, 1, 1, 1    #RShSag -
                           ])
    """
    joint_signs = np.array([-1, 1, -1, -1, 1, -1,     #LHipLat -
                            1, 1, 1, 1, -1, -1,       #RHipLat -
                            #1,                        #WaistLat
                            -1, -1,                   #WaistSag -
                            -1, 1, 1, -1, -1, 1, 1,   #LShSag -
                            -1, 1, 1, 1, -1, -1, -1   #RShSag -
                           ])
    joint_offsets = np.array([0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              #0,                      #WaistLat
                              0, 0,
                              0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0
                             ])

    # shift measured torques backwards by this many samples
    if not is_gazebo:
        time_offset = round(200*0.09)
    else:
        time_offset = 0

    # scale feet F/T sensors so zero value is force corresponding to weight
    # walkman whole weight is 139.14 kg (maybe was without protection at times? -4.7)
    # (equalized at peaks on one leg, scaled so sum of stance on both legs corrseponds to mass)
    # the F/T linear z axis should be negative
    # imu should have -9.81 in idle phases on the z component
    if not is_gazebo:
        scale = -1.02
        ft_left_scale = 0.9 * scale
        ft_right_scale = 1.15 * scale
        imu_scale = 1.035
    else:
        scale = -1.03
        ft_left_scale = scale
        ft_right_scale = scale
        imu_scale = 1.0

    print(np.array(jointNames)[csv_T_urdf_indices])

    config['num_dofs'] = len(jointNames) - len(ignoreJoints)

    filepath = os.path.join(path, 'jointLog.csv')     #joint positions and torques
    f = np.loadtxt(filepath)
    out['positions'] = np.zeros( (f.shape[0], config['num_dofs']) )
    out['torques'] = np.zeros( (f.shape[0], config['num_dofs']) )
    out['times'] = np.zeros( f.shape[0] )
    out['times'][:] = np.arange(0, f.shape[0])*sampleTime
    out['target_positions'] = np.zeros( (f.shape[0], config['num_dofs']) )

    # generate some empty arrays, will be calculated in preprocess()
    out['velocities'] = np.zeros_like(out['positions'])
    out['accelerations'] = np.zeros_like(out['positions'])

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(3,2,1) # three rows, two columns, first plot
        ax2 = fig.add_subplot(3,2,2)
    dofs_file = len(f[1])//7

    ign = 0    # when link is ignored, count offset to skip column in csv
    skip = 4   # skip some values when plotting
    for dof in range(0, config['num_dofs']):   # go though amount of links without ignored
        while dof+ign in ignoreJoints:
            ign+=1

        out['target_positions'][:, dof] = f[:, csv_T_urdf_indices[dof+ign]+dofs_file*0]   #position reference
        out['positions'][:, dof] = f[:, csv_T_urdf_indices[dof+ign]+dofs_file*2]   #link encoders
        f_len = f.shape[0]
        out['torques'][time_offset:, dof] = f[:f_len-time_offset, csv_T_urdf_indices[dof+ign]+dofs_file*4]   #torque sensors
        if plot:
            ax1.plot(out['times'][::skip], out['positions'][::skip, dof], label=jointNames[dof])
            ax2.plot(out['times'][::skip], out['torques'][::skip, dof], label=jointNames[dof])

    #correct signs and offsets (until measurements are fixed)
    if not is_gazebo:
        out['torques'] = out['torques'] * joint_signs + joint_offsets

    filepath = os.path.join(path, 'feedbackData.csv')    # force torque and IMU
    f = np.loadtxt(filepath)
    out['FTleft'] = np.zeros((f.shape[0], 6))       # FT left foot, 3 force, 3 torque values
    out['FTright'] = np.zeros((f.shape[0], 6))      # FT right foot, 3 force, 3 torque values
    out['IMUrpy'] = np.zeros((f.shape[0], 3))       # IMU orientation, r,p,y
    out['IMUlinAcc'] = np.zeros((f.shape[0], 3))    # IMU linear acceleration
    out['IMUlinAcc2'] = np.zeros((f.shape[0], 3))   # IMU linear acceleration 2nd IMU
    out['IMUrotVel'] = np.zeros((f.shape[0], 3))    # IMU rotational velocity

    if plot:
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        rpy_labels = ['r','p', 'y']
        acc_labels = ['x', 'y', 'z']
    for i in range(0,3):
        if not is_gazebo:
            # use data fields of LPMS IMU (LPMS-CU)
            #out['IMUrpy'][:, i] = np.deg2rad(f[:, i])    # data in deg
            out['IMUlinAcc'][:, i] = f[:, 18+i] * 9.81   # data in g -> (m/s2)/9.81
            out['IMUrotVel'][:, i] = np.deg2rad(f[:, 21+i])   # data in deg/s

            # use data fields of VN IMU (VectorNav VN-100)
            out['IMUrpy'][:, i] = f[:, 15+i]       # rad
            out['IMUlinAcc2'][:, i] = np.copy(f[:, 24+i])  #m/s2
            #out['IMUrotVel'][:, i] = f[:, 27+i]   # rad/s, but has some constant offsets
            #out['IMUrotVel'][:, i] -= np.mean(out['IMUrotVel'][:, i])  #remove those offsets
        else:
            # sim
            out['IMUrpy'][:, i] = f[:, i]
            out['IMUlinAcc'][:, i] = f[:, 18+i]
            out['IMUrotVel'][:, i] = f[:, 21+i]

    if not is_gazebo:
        # rotate VN-100 vals to robot frame (as it is physically rotated)
        #robotToIMU = iDynTree.Rotation.RPY(np.pi, 0, 0).toNumPy()
        #for j in range(0, out['IMUlinAcc2'].shape[0]):
        #    out['IMUlinAcc2'][j, :] = robotToIMU.dot(out['IMUlinAcc2'][j, :])
        # can also just flip y and z to get rotation
        out['IMUlinAcc2'][:, 1] *= -1
        out['IMUlinAcc2'][:, 2] *= -1

        # rotate rotational velocity (if using VN-100)
        #for j in range(0, out['IMUrotVel'].shape[0]):
        #    out['IMUrotVel'][j, :] = robotToIMU.dot(out['IMUrotVel'][j, :])

        # correct rotation estimation (should have roll +- 180 deg but doesn't?)
        #out['IMUrpy'][:, 0] -= np.pi

    '''
    # use both IMUs and take average
    grav_norm = np.mean(la.norm(out['IMUlinAcc'], axis=1))
    if grav_norm < 9.81 or grav_norm > 9.82:
        #print('Warning: mean base acceleration is different than gravity ({})! Scaling'.format(grav_norm))
        #scale up/down
        out['IMUlinAcc'] *= 9.81/grav_norm
    grav_norm = np.mean(la.norm(out['IMUlinAcc2'], axis=1))
    if grav_norm < 9.81 or grav_norm > 9.82:
        #print('Warning: mean base acceleration is different than gravity ({})! Scaling'.format(grav_norm))
        #scale up/down
        out['IMUlinAcc2'] *= 9.81/grav_norm

    out['IMUlinAcc'] = np.mean([-out['IMUlinAcc'], out['IMUlinAcc2']], axis=0)
    '''

    if not is_gazebo:
        #use second IMU
        out['IMUlinAcc'] = out['IMUlinAcc2']

    # gravity should point downward

    out['IMUlinAcc'] *= imu_scale

    if plot:
        for i in range(0,3):
            ax3.plot(out['times'][::skip], out['IMUrpy'][::skip, i], label=rpy_labels[i])
            ax4.plot(out['times'][::skip], out['IMUlinAcc'][::skip, i], label=acc_labels[i])

    if plot:
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)
        ft_labels = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z']

    #hardware and gazebo ft sensors seem to have different sign
    #hw sensors are unreliable in linear x and y axes, but small values so ignore them for now
    if not is_gazebo:
        out['FTleft'][:, 0] = f[:, 3]*0
        out['FTleft'][:, 1] = f[:, 4]*0
        out['FTleft'][:, 2] = f[:, 5]

        out['FTleft'][:, 3] = f[:, 6]
        out['FTleft'][:, 4] = f[:, 7]
        out['FTleft'][:, 5] = f[:, 8]

        out['FTright'][:, 0] = f[:, 9]*0
        out['FTright'][:, 1] = f[:, 10]*0
        out['FTright'][:, 2] = f[:, 11]

        out['FTright'][:, 3] = f[:, 12]
        out['FTright'][:, 4] = f[:, 13]
        out['FTright'][:, 5] = f[:, 14]

        #FTtoWorld = iDynTree.Rotation.RPY(0, 0, np.pi).toNumPy()
        #for j in range(0, out['FTright'].shape[0]):
        #    out['FTright'][j, 0:3] = FTtoWorld.dot(out['FTright'][j, 0:3])
        #    out['FTright'][j, 3:6] = FTtoWorld.dot(out['FTright'][j, 3:6])
    else:
        out['FTleft'][:, 0:6] = f[:, 3:9]
        out['FTright'][:, 0:6] = f[:, 9:15]

    # scale/invert all or just z?
    out['FTleft'] *= ft_left_scale
    out['FTright'] *= ft_right_scale

    if plot:
        for i in range(0,6):
            ax5.plot(out['times'][::skip], out['FTleft'][::skip, i], label=ft_labels[i])
            ax6.plot(out['times'][::skip], out['FTright'][::skip, i], label=ft_labels[i])

        #set titles and enable legends for each subplot
        t = ['positions', 'torques', 'IMU rpy', 'IMU acc', 'FT left', 'FT right']
        for i in range(0,5):
            plt.subplot(321+i)
            eval('ax{}'.format(1+i)).legend(fancybox=True, fontsize=10, title='')
            plt.title(t[i])

        fig.tight_layout()

        plt.show()

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load measurements from csv and write as npz.')
    parser.add_argument('--config', required=True, type=str, help="use options from given config file")
    parser.add_argument('--measurements', required=True, type=str, help='the directory to load the measurements from')
    parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--regressor', required=False, type=str, help='the file to load the regressor structure from (only for simulation)')
    parser.add_argument('--outfile', type=str, help='the filename to save the measurements to')
    parser.add_argument('--plot', help='whether to plot the data', action='store_true')
    parser.add_argument('--robot', required=True, help='which robot to import data for (walkman, centauro)', type=str)
    parser.set_defaults(outfile='measurements.npz', plot=False)
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config['useDeg'] = 0
    config['model'] = args.model

    if args.robot == 'centauro':
        out = readCentauroCSV(args.measurements, config, args.plot)
        sampleTime = np.mean(np.diff(out['times']))
    elif args.robot == 'walkman':
        sampleTime = 0.005   #200 Hz
        out = readWalkmanCSV(args.measurements, config, args.plot)
    out['frequency'] = 1.0/sampleTime
    data = Data(config)

    #init empty arrays for preprocess
    out['torques_raw'] = np.zeros_like( out['torques'])

    #filter, diff, integrate
    if args.robot == 'walkman':
        out['IMUlinVel'] = np.zeros( (out['times'].shape[0], 3) ) #IMU linear velocity
        out['IMUrotAcc'] = np.zeros( (out['times'].shape[0], 3) ) #IMU rotational acceleration
        data.preprocess(Q=out['positions'], V=out['velocities'], Vdot=out['accelerations'],
                        Tau=out['torques'], Tau_raw=out['torques_raw'], T=out['times'],
                        Fs=out['frequency'], IMUlinVel=out['IMUlinVel'], IMUrotVel=out['IMUrotVel'],
                        IMUlinAcc=out['IMUlinAcc'], IMUrotAcc=out['IMUrotAcc'], IMUrpy=out['IMUrpy'],
                        FT=[out['FTleft'], out['FTright']])

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
        out['contacts'] = np.array({'l_leg_ft': out['FTleft'], 'r_leg_ft': out['FTright']})
        out['base_rpy'] = out['IMUrpy']

    else:
        #fixed base
        data.preprocess(Q=out['positions'], V=out['velocities'], Vdot=out['accelerations'],
                        Tau=out['torques'], Tau_raw=out['torques_raw'], T=out['times'],
                        Fs=out['frequency'])

    #simulate with iDynTree if we're using gazebo data
    if is_gazebo:
        #use all data
        old_skip = config['skipSamples']
        config['skipSamples'] = 0
        old_offset = config['startOffset']
        config['startOffset'] = 0
        old_sim = config['simulateTorques']
        config['simulateTorques'] = 1
        data.init_from_data(out)
        model = Model(config, config['model'], args.regressor)
        if config['verbose']:
            print('simulating torques for motion data')
        model.computeRegressors(data)
        out['torques'] = out['torques_raw'] = model.data.samples['torques']
        config['skipSamples'] = old_skip
        config['startOffset'] = old_offset
        config['simulateTorques'] = old_sim

    if config['floatingBase']:
        np.savez(args.outfile, positions=out['positions'], positions_raw=out['positions'],
                 target_positions=out['target_positions'],
                 velocities=out['velocities'], velocities_raw=out['velocities'],
                 accelerations=out['accelerations'], torques=out['torques'],
                 torques_raw=out['torques_raw'], base_velocity=out['base_velocity'],
                 base_acceleration=out['base_acceleration'], base_rpy=out['base_rpy'],
                 contacts=out['contacts'], times=out['times'], frequency=out['frequency'])
    else:
        np.savez(args.outfile, positions=out['positions'], positions_raw=out['positions'],
                 velocities=out['velocities'], velocities_raw=out['velocities'],
                 accelerations=out['accelerations'], torques=out['torques'],
                 torques_raw=out['torques_raw'], contacts=out['contacts'], times=out['times'], frequency=out['frequency'])

    print("Saved csv data as {}".format(args.outfile))
    print("Samples: {}, Time: {}s, Frequency: {} Hz".format(out['times'].shape[0], out['times'][-1], out['frequency']))
