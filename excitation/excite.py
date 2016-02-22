#!/usr/bin/env python2.7

import sys
import yarp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal

import argparse
parser = argparse.ArgumentParser(description='Generate an excitation and record measurements to <filename>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--filename', type=str, help='the filename to save the measurements to')
parser.add_argument('--periods', type=int, help='how many periods to run the trajectory')
parser.add_argument('--plot', help='plot sent/received data', action='store_true')
parser.add_argument('--dryrun', help="don't not send the trajectory", action='store_true')
parser.add_argument('--random-colors', dest='random_colors', help="use random colors for graphs", action='store_true')
parser.set_defaults(plot=False, dryrun=False, random_colors=False, filename='measurements.npz', periods=1)
args = parser.parse_args()

import iDynTree
jointNames = iDynTree.StringVector([])
iDynTree.dofsListFromURDF(args.model, jointNames)
N_DOFS = len(jointNames)

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
            q = (self.a[l-1]/(self.w_f*l))*np.sin(self.w_f*l*t) - \
                 (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l*t)
        return np.rad2deg(q) + self.nf*self.q0

    def getVelocity(self, t):
        dq = 0
        for l in range(1, self.nf+1):
            dq += self.a[l-1]*np.cos(self.w_f*l*t) + \
                  self.b[l-1]*np.sin(self.w_f*l*t)
        return np.rad2deg(dq)

class TrajectoryGenerator(object):
    def __init__(self, dofs):
        self.dofs = dofs
        self.oscillators = list()

        self.w_f_global = 1.0
        a = [[0.4], [0.3], [0.75], [0.5], [1], [-0.7], [-0.8]]
        b = [[0.4], [0.3], [0.75], [0.8], [1], [1.3], [0.8]]
        q = [10, 50, -80, -25, 50, 0, -15]
        nf = [1,1,1,1,1,1,1]

        for i in range(0, dofs):
            self.oscillators.append( OscillationGenerator(w_f=self.w_f_global,
                a=np.array(a[i]), b=np.array(b[i]), q0=q[i], nf= nf[i]) )

    def getAngle(self, dof):
        return self.oscillators[dof].getAngle(self.time)

    def getVelocity(self, dof):
        return self.oscillators[dof].getVelocity(self.time)

    def getPeriodLength(self):   #in seconds
        return 2*np.pi/self.w_f_global

    def setTime(self, time):     #in seconds
        self.time = time

def gen_position_msg(msg_port, angles, velocities):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("(set_left_arm {} {}) 0".format(' '.join(map(str, angles)), ' '.join(map(str, velocities)) ))
    return bottle

def gen_command(msg_port, command):
    bottle = msg_port.prepare()
    bottle.clear()
    bottle.fromString("({}) 0".format(command))
    return bottle

def main():
    #connect to yarp and open output port
    yarp.Network.init()
    yarp.Time.useNetworkClock("/clock")
    yarp.Time.now()  #use clock once to sync (?)
    while not yarp.Time.isValid():
        continue

    portName = '/excitation/command:'
    command_port = yarp.BufferedPortBottle()
    command_port.open(portName+'o')
    yarp.Network.connect(portName+'o', portName+'i')

    portName = '/excitation/state:'
    data_port = yarp.BufferedPortBottle()
    data_port.open(portName+"i")
    yarp.Network.connect(portName+'o', portName+'i')

    #init trajectory generator for all the joints
    trajectories = TrajectoryGenerator(N_DOFS)

    t_init = yarp.Time.now()
    t_elapsed = 0.0
    duration = args.periods*trajectories.getPeriodLength()   #init overall run duration to a periodic length

    measured_positions = list()
    measured_velocities = list()
    measured_accelerations = list()
    measured_torques = list()
    measured_time = list()

    first_pose = True
    sent_positions = list()
    sent_time = list()
    sent_velocities = list()

    #try high level p correction when using velocity ctrl
    e = [0] * N_DOFS
    velocity_correction = [0] * N_DOFS

    while t_elapsed < duration:
        #TODO: make sure we're starting at zero velocity (wait for it or move functions)
        trajectories.setTime(t_elapsed)
        angles = [trajectories.getAngle(i) for i in range(0, N_DOFS)]
        velocities = [trajectories.getVelocity(i) for i in range(0, N_DOFS)]
        for i in range(0, N_DOFS):
            velocities[i]+=velocity_correction[i]

        #set target angles
        gen_position_msg(command_port, angles, velocities)
        command_port.write()

        #set first angle vector and wait a bit to start at right position
        if first_pose:
            print "waiting a bit for initial position...",
            sys.stdout.flush()
            #TODO: actually read pose and compare
            yarp.Time.delay(2.0)
            t_init+=2.0
            print "ok."
        else:
            sent_positions.append(angles)
            sent_velocities.append(velocities)
            sent_time.append(yarp.Time.now())

        #loop delay: wait for 2*0.005s=100Hz (not used when synced to GYM timing)
        #yarp.Time.delay(0.010)

        #wait for next value, so sync to GYM loop
        data = data_port.read(shouldWait=True)
        if data:    #can only be not true if shouldWait=False, need delay in that case
            b_positions = data.get(0).asList()
            b_velocities = data.get(1).asList()
            b_torques = data.get(2).asList()
            d_time = data.get(3).asDouble()

            positions = np.zeros(N_DOFS)
            velocities = np.zeros(N_DOFS)
            accelerations = np.zeros(N_DOFS)
            torques = np.zeros(N_DOFS)

            if N_DOFS == b_positions.size():
                for i in range(0, N_DOFS):
                    positions[i] = b_positions.get(i).asDouble()
                    velocities[i] = b_velocities.get(i).asDouble()
                    torques[i] = b_torques.get(i).asDouble()
            else:
                print "warning, wrong amount of values received! ({} DOFS vs. {})".format(N_DOFS, b_positions.size())

            #test manual correction for position error
            p = 0
            for i in range(0,N_DOFS):
                e[i] = (angles[i] - positions[i])
                velocity_correction[i] = e[i]*p

            #collect measurement data
            if not first_pose:
                measured_positions.append(positions)
                measured_velocities.append(velocities)
                measured_torques.append(torques)
                measured_time.append(d_time)

            t_elapsed = d_time - t_init
        else:
            print "oops, skipped reading one frame"
            t_elapsed = yarp.Time.now() - t_init

        if first_pose:
            first_pose=False

    #clean up
    command_port.close()
    data_port.close()
    Q = np.array(measured_positions); del measured_positions
    Qsent = np.array(sent_positions);
    V = np.array(measured_velocities); del measured_velocities
    Vdot = np.zeros_like(V)   #will be calculated in preprocess()
    global Vraw
    Vraw = np.zeros_like(V)   #will be calculated in preprocess()
    Tau = np.array(measured_torques); del measured_torques
    global TauRaw
    TauRaw = np.zeros_like(Tau)   #will be calculated in preprocess()
    T = np.array(measured_time); del measured_time

    #filter, differentiate, etc.
    preprocess(Q, V, Vraw, Vdot, Tau, TauRaw, T)

    #write sample arrays to data file
    np.savez_compressed(args.filename, positions=Q, target_positions=Qsent,
            velocities=V, accelerations=Vdot, torques=Tau, times=T)
    print "saved measurements to {}".format(args.filename)

    ## some stats
    print "got {} samples in {}s.".format(Q.shape[0], duration),
    print "(about {} Hz)".format(len(sent_positions)/duration)

def preprocess(posis, vels, vels_unfiltered, accls, torques, torques_unfiltered, times):
    #filter velocity
    vels_orig = vels.copy()
    fc = 1.0 # Cut-off frequency (Hz)
    fs = 200.0 # Sampling rate (Hz)
    order = 6 # Filter order
    b, a = sp.signal.butter(order, fc / (fs/2), btype='low', analog=False)
    for j in range(0, N_DOFS):
        vels[:, j] = sp.signal.filtfilt(b, a, vels_orig[:, j])
    np.copyto(vels_unfiltered, vels_orig)


    """
    # Plot the frequency response.
    w, h = sp.signal.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(fc, 0.5*np.sqrt(2), 'ko')
    plt.axvline(fc, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    """

    #calc accelerations
    for i in range(1, vels.shape[0]):
        dT = times[i] - times[i-1]
        accls[i] = (vels[i] - vels[i-1])/dT

    #median filter of accelerations
    accls_orig = accls.copy()
    for j in range(0, N_DOFS):
        accls[:, j] = sp.signal.medfilt(accls_orig[:, j], 11)

    #median filter of torques
    torques_orig = torques.copy()
    for j in range(0, N_DOFS):
        torques[:, j] = sp.signal.medfilt(torques_orig[:, j], 11)

    #low-pass of torques
    torques_orig = torques.copy()
    for j in range(0, N_DOFS):
        torques[:, j] = sp.signal.filtfilt(b, a, torques_orig[:, j])
    np.copyto(torques_unfiltered, torques_orig)

def plot():
    if args.random_colors:
        from random import sample
        from itertools import permutations

        #get a random color wheel
        Nlines = 200
        color_lvl = 8
        rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
        colors = sample(rgb,Nlines)
        #print colors[0:7]
    else:
        #set some nice fixed colors
        colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.94117647,  0.03137255,  0.59607843],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]

    #python measurements
    #reload measurements from this or last run (if run dry)
    measurements = np.load('measurements.npz')
    M1 = measurements['positions']
    M1_t = measurements['target_positions']
    M2 = measurements['velocities']
    M2_dot = measurements['accelerations']
    M3 = measurements['torques']
    T = measurements['times']
    num_samples = measurements['positions'].shape[0]
    print 'loaded {} measurement samples'.format(num_samples)

    print "tracking error per joint:"
    for i in range(0,N_DOFS):
        sse = np.sum((M1[:, i] - M1_t[:, i]) ** 2)
        print "joint {}: {}".format(i, sse)

    print "histogram of yarp time diffs"
    dT = np.diff(T)
    H, B = np.histogram(dT)
    #plt.hist(H, B)
    print "bins: {}".format(B)
    print "sums: {}".format(H)
    late_msgs = (1 - float(np.sum(H)-np.sum(H[1:])) / float(np.sum(H))) * 100
    print "({}% messages too late)".format(late_msgs)
    print "\n"

    #what to plot (each tuple has a title and one or multiple data arrays)
    datasets = [
            ([M1, M1_t], 'Positions'),
            ([M2, Vraw],'Velocities'),
            ([M2_dot,], 'Accelerations'),
            ([M3, TauRaw],'Measured Torques')
            ]

    for (data, title) in datasets:
        plt.figure()
        plt.title(title)
        for i in range(0, N_DOFS):
            for d_i in range(0, len(data)):
                l = jointNames[i] if d_i == 0 else ''  #only put joint names in the legend once
                plt.plot(T, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
        plt.legend(loc='lower right')

    #yarp times over time indices
    plt.figure()
    plt.plot(range(0,len(T)), T)

    plt.show()

if __name__ == '__main__':
    #from IPython import embed; embed()

    try:
        if(not args.dryrun):
            main()

        if(args.plot):
            plot()
    except Exception as e:
        if type(e) is not KeyboardInterrupt:
            #open ipdb when an exception happens
            import sys, ipdb, traceback
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)

