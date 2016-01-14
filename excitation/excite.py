#!/usr/bin/env python2.7

import yarp
import numpy as np
#import PyKDL as kdl

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
              + (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l)
        return np.degrees(q) + self.nf*self.q0


def gen_position_msg(msg_port, angles):
    bottle = msg_port.prepare()
    bottle.clear()
    print angles
    bottle.fromString("(set_left_arm {}) 0".format(' '.join(map(str, angles)) ))
    return bottle

if __name__ == '__main__':
    #connect to yarp and open output port
    yarp.Network.init()
    while not yarp.Time.isValid():
        continue

    command_port = yarp.BufferedPortBottle()
    portName = '/excitation/command:'
    command_port.open(portName+'o')
    yarp.Network.connect(portName+'o', portName+'i')

    #create oscillators for each joint (synchronized frequencies)
    t_init = yarp.Time.now()
    t_elapsed = 0.0
    duration = 10
    w_f_global = 1.0
    trajectory0 = OscillationGenerator(w_f=w_f_global, a=np.array([0.4]), b=np.array([0.4]), q0=-10, nf=1)
    trajectory1 = OscillationGenerator(w_f=w_f_global, a=np.array([0.5]), b=np.array([0.3]), q0=10, nf=1)
    trajectory2 = OscillationGenerator(w_f=w_f_global, a=np.array([0.75]), b=np.array([0.75]), q0=-80, nf=1)
    trajectory3 = OscillationGenerator(w_f=w_f_global, a=np.array([0.6]), b=np.array([1.0]), q0=-20, nf=1)
    trajectory4 = OscillationGenerator(w_f=w_f_global, a=np.array([1]), b=np.array([1]), q0=0, nf=1)
    trajectory5 = OscillationGenerator(w_f=w_f_global, a=np.array([-0.7]), b=np.array([1.3]), q0=0, nf=1)
    trajectory6 = OscillationGenerator(w_f=w_f_global, a=np.array([-1]), b=np.array([1]), q0=0, nf=1)

    while t_elapsed < duration:
        angles = list()
        for i in range(0, 7):
            angles.append( eval("trajectory{}.getAngle(t_elapsed)".format(i)) )
        gen_position_msg(command_port, angles)

        t_elapsed = yarp.Time.now() - t_init
        command_port.write()
        yarp.Time.delay(0.01)

    command_port.close()
