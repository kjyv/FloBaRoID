#!/usr/bin/env python2.7

import yarp
import numpy as np
#import PyKDL as kdl
import matplotlib.pyplot as plt

import time
from contextlib import contextmanager
@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))

import argparse
parser = argparse.ArgumentParser(description='Generate an excitation and record measurements to <filename>.')
parser.add_argument('--filename', default='measurements.npz', type=str,
        help='the filename to save the measurements to')
parser.add_argument('--plot', help='whether to plot sent trajectories', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

#TODO: use model information for this, generate trajectory in a generic model dependent way
N_DOFS = 7

#pulsating trajectory generator for one joint using fourier series from Sewers, Gansemann (1997)
class OscillationGenerator(object):
    def __init__(self, w_f, a, b, q0, nf):
        #- w_f is the global pulsation (frequency is w_f / 2pi)
        #- a and b are (arrays of) amplitudes of the sine/cosine
        #  functions for each joint
        #- q0 is the joint angle offset (center of pulsation)
        #- nf is the desired amount of coefficients for this fourier series
        self.w_f = float(w_f)
        self.a = a
        self.b = b
        self.q0 = float(q0)
        self.nf = nf

    def getAngle(self, t):
        #- t is the current time
        q = 0
        for l in range(1, self.nf+1):
            q = (self.a[l-1]/(self.w_f*l))*np.sin(self.w_f*l*t) \
              - (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l*t)
        return np.degrees(q) + self.nf*self.q0

    #TODO: add fourier series for velocity, use velocity ctrl

class TrajectoryGenerator(object):
    def __init__(self, dofs):
        self.dofs = dofs
        self.oscillators = list()

        self.w_f_global = 1.0
        a = [[0.4], [0.3], [0.75], [0.5], [1], [-0.7], [-0.8]]
        b = [[0.4], [0.3], [0.75], [0.8], [1], [1.3], [0.8]]
        q = [-10, 30, -80, -25, 0, 0, -15]
        nf = [1,1,1,1,1,1,1]

        for i in range(0, dofs):
            self.oscillators.append( OscillationGenerator(w_f=self.w_f_global, a=np.array(a[i]), b=np.array(b[i]), q0=q[i], nf= nf[i]) )

    def getAngle(self, dof):
        return self.oscillators[dof].getAngle(self.time)

    def getPeriodLength(self):
        #get periodicity in seconds
        return 2*np.pi/self.w_f_global

    def setTime(self, time):
        self.time = time

def gen_position_msg(msg_port, angles):
    bottle = msg_port.prepare()
    bottle.clear()
    print "send angles: {}".format(angles)
    bottle.fromString("(set_left_arm {}) 0".format(' '.join(map(str, angles)) ))
    return bottle

def gen_command(msg_port, command):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("({}) 0".format(command))
    return bottle

if __name__ == '__main__':
    #connect to yarp and open output port
    yarp.Network.init()
    while not yarp.Time.isValid():
        continue

    portName = '/excitation/command:'
    command_port = yarp.BufferedPortBottle()
    command_port.open(portName+'o')
    yarp.Network.connect(portName+'o', portName+'i')

    portName = '/excitation/dataOutput:'
    data_port = yarp.BufferedPortBottle()
    data_port.open(portName+"i")
    yarp.Network.connect(portName+'o', portName+'i')

    #create oscillators for each joint (synchronized frequencies)
    #parameters are a bit arbitrary so far...
    t_init = yarp.Time.now()
    t_elapsed = 0.0
    duration = 2*trajectories.getPeriodLength()   #init overall run duration to a periodic length

    measured_positions = list()
    measured_velocities = list()
    measured_torques = list()
    measured_time = list()

    first_pose = True
    sent_positions = list()
    sent_time = list()

    while t_elapsed < duration:
        trajectories.setTime(t_elapsed)
        angles = [trajectories.getAngle(i) for i in range(0, N_DOFS)]

        #set target angles
        gen_position_msg(command_port, angles)

        sent_positions.append(angles)
        sent_time.append(yarp.Time.now())

        #wait between commands but also together with other delays,
        #wait for 2*0.005s=100Hz
        yarp.Time.delay(0.005)

        gen_command(command_port, "get_left_arm_measurements")
        command_port.write()
        yarp.Time.delay(0.005)

        data = data_port.read(shouldWait=False)
        #it can happen that no data is received (yet), so measurement will be skipped
        if data:
            positions = np.zeros(N_DOFS)
            velocities = np.zeros(N_DOFS)
            torques = np.zeros(N_DOFS)
            for i in range(0, N_DOFS):
                 positions[i] = data.get(i).asDouble()
                 velocities[i] = data.get(i+7).asDouble()
                 torques[i] = data.get(i+7+7).asDouble()
            print "received positions: {}".format(positions)
            print "received velocities: {}".format(velocities)
            print "received torques: {}".format(torques)

            #collect measurement data
            measured_positions.append(positions)
            measured_velocities.append(velocities)
            measured_torques.append(torques)
            measured_time.append(yarp.Time.now())
        else:
            print "oops, skipped reading one frame"

        t_elapsed = yarp.Time.now() - t_init
        print "elapsed time: {}".format(t_elapsed)

    command_port.close()
    M1 = np.array(measured_positions)
    del measured_positions
    M2 = np.array(measured_velocities)
    del measured_velocities
    M3 = np.array(measured_torques)
    del measured_torques
    T = np.array(measured_time)
    del measured_time

    print "got {} samples.".format(M1.shape[0]),

    #write sample arrays to data file
    np.savez_compressed(args.filename, positions=M1, velocities=M2, torques=M3, times=T)
    print "saved to {}".format(args.filename)

#from IPython import embed; embed()

#plot values that were sent
if(args.plot):
    print "plotting {} values (about {} Hz)".format(len(sent_positions), len(sent_positions)/duration)
    M = np.array(sent_positions)
    t = np.array(sent_time)
    for i in range(0, N_DOFS):
        plt.plot(t, M[:, i], '+')
    plt.title('Positions')
    plt.show()
