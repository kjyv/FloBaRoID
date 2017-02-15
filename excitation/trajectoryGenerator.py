from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np

class TrajectoryGenerator(object):
    ''' pulsating trajectory generator for one joint using fourier series from
        Swevers, Gansemann (1997). Gives values for one time instant (at the current
        internal time value)
    '''
    def __init__(self, dofs, use_deg=False):
        self.dofs = dofs
        self.oscillators = list()
        self.use_deg = use_deg
        self.w_f_global = 1.0

    def initWithRandomParams(self):
        # init with random params
        # TODO: use specified bounds
        a = [0]*self.dofs
        b = [0]*self.dofs
        nf = np.random.randint(1,4, self.dofs)
        q = np.random.rand(self.dofs)*2-1
        for i in range(0, self.dofs):
            maximum = 2.0-np.abs(q[i])
            a[i] = np.random.rand(nf[i])*maximum-maximum/2
            b[i] = np.random.rand(nf[i])*maximum-maximum/2

        #random values are in rad, so convert
        if self.use_deg:
            q = np.rad2deg(q)
        #print a
        #print b
        #print q

        self.a = a
        self.b = b
        self.q = q
        self.nf = nf

        self.oscillators = list()
        for i in range(0, self.dofs):
            self.oscillators.append(OscillationGenerator(w_f = self.w_f_global, a = np.array(a[i]),
                                                         b = np.array(b[i]), q0 = q[i], nf = nf[i],
                                                         use_deg = self.use_deg
                                                        ))
        return self

    def initWithParams(self, a, b, q, nf, wf=None):
        ''' init with given params
            a - list of dof coefficients a
            b - list of dof coefficients b
            q - list of dof coefficients q_0
            nf - list of dof coefficients n_f
            (also see docstring of OscillationGenerator)
        '''

        if len(nf) != self.dofs or len(q) != self.dofs:
            raise Exception("Need DOFs many values for nf and q!")

        #for i in nf:
        #    if not ( len(a) == i and len(b) == i):
        #        raise Exception("Need nf many values in each parameter array value!")

        self.a = a
        self.b = b
        self.q = q
        self.nf = nf
        if wf:
            self.w_f_global = wf

        self.oscillators = list()
        for i in range(0, self.dofs):
            self.oscillators.append(OscillationGenerator(w_f = self.w_f_global, a = np.array(a[i]),
                                                         b = np.array(b[i]), q0 = q[i], nf = nf[i], use_deg = self.use_deg
                                                        ))
        return self

    def getAngle(self, dof):
        """ get angle at current time for joint dof """
        return self.oscillators[dof].getAngle(self.time)

    def getVelocity(self, dof):
        """ get velocity at current time for joint dof """
        return self.oscillators[dof].getVelocity(self.time)

    def getAcceleration(self, dof):
        """ get acceleration at current time for joint dof """
        return self.oscillators[dof].getAcceleration(self.time)

    def getPeriodLength(self):
        ''' get the period length of the oscillation in seconds '''
        return 2*np.pi/self.w_f_global

    def setTime(self, time):
        '''set current time in seconds'''
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        self.setTime(t_elapsed)
        if self.use_deg: thresh = 5
        else: thresh = np.deg2rad(5)
        return abs(self.getVelocity(0)) < thresh

# generate some static testing 'trajectories'
class FixedPositionTrajectory(object):
    def __init__(self):
        self.time = 0

    def getAngle(self, dof):
        """ get angle at current time for joint dof """
        # Walk-Man:
        # ['LHipLat', 'LHipYaw', 'LHipSag', 'LKneeSag', 'LAnkSag', 'LAnkLat',
        #  'RHipLat', 'RHipYaw', 'RHipSag', 'RKneeSag', 'RAnkSag', 'RAnkLat',
        #  'WaistSag', 'WaistYaw', #WaistLat is fixed atm
        #  'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2',
        #  'RShSag', 'RShLat', 'RShYaw', 'RElbj', 'RForearmPlate', 'RWrj1', 'RWrj2']
        '''
        # posture #0
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       #left leg
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         #right leg
                0.0, 0.0,                           #Waist
                0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                ][dof]
        # posture #1
        return [0.0, 0.0, -45.0, 0.0, -15.0, 0.0,       #left leg
                0.0, 0.0, -45.0, 0.0, -15.0, 0.0,         #right leg
                0.0, 0.0,                           #Waist
                90.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                90.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                ][dof]
        # posture #2
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       #left leg
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         #right leg
                0.0, 0.0,                           #Waist
                20.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                20.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                ][dof]
        # posture #3
        return [0.0, 0.0, 0.0, 0.0, -79.0, 0.0,       #left leg
                0.0, 0.0, 0.0, 0.0, -79.0, 0.0,         #right leg
                -6.0, 0.0,                           #Waist
                0.0, 90.0, 0.0, -90.0, 0.0, -45.0, 0.0,    #left arm
                0.0, -90.0, 0.0, -90.0, 0.0, -45.0, 0.0,   #right arm
                ][dof]
        '''
        # posture #4
        return [44.0, 0.0, 0.0, 60.0, 0.0, 0.0,       #left leg
                -44.0, 0.0, 0.0, 60.0, 0.0, 0.0,         #right leg
                0.0, 0.0,                           #Waist
                0.0, 10.0, 0.0, -90.0, 0.0, 0.0, 79.0,    #left arm
                0.0, -10.0, 0.0, -90.0, 0.0, 0.0, -79.0,   #right arm
                ][dof]

    def getVelocity(self, dof):
        """ get velocity at current time for joint dof """
        return 0.0

    def getAcceleration(self, dof):
        """ get acceleration at current time for joint dof """
        return 0.0

    def getPeriodLength(self):
        ''' get the period length of the oscillation in seconds '''
        return 1

    def setTime(self, time):
        '''set current time in seconds'''
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        return True

class OscillationGenerator(object):
    def __init__(self, w_f, a, b, q0, nf, use_deg):
        '''
        generate periodic oscillation from fourier series (Swevers, 1997)

        - w_f is the global pulsation (frequency is w_f / 2pi)
        - a and b are (arrays of) amplitudes of the sine/cosine
          functions for each joint
        - q0 is the joint angle offset (center of pulsation)
        - nf is the desired amount of coefficients for this fourier series
        '''
        self.w_f = float(w_f)
        self.a = a
        self.b = b
        self.use_deg = use_deg
        self.q0 = float(q0)
        if use_deg:
            self.q0 = np.deg2rad(self.q0)
        self.nf = nf

    def getAngle(self, t):
        #- t is the current time
        q = 0
        for l in range(1, self.nf+1):
            q += (self.a[l-1]/(self.w_f*l))*np.sin(self.w_f*l*t) - \
                 (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l*t)
        q += self.nf*self.q0
        if self.use_deg:
            q = np.rad2deg(q)
        return q

    def getVelocity(self, t):
        dq = 0
        for l in range(1, self.nf+1):
            dq += self.a[l-1]*np.cos(self.w_f*l*t) + \
                  self.b[l-1]*np.sin(self.w_f*l*t)
        if self.use_deg:
            dq = np.rad2deg(dq)
        return dq

    def getAcceleration(self, t):
        ddq = 0
        for l in range(1, self.nf+1):
            ddq += -self.a[l-1]*self.w_f*l*np.sin(self.w_f*l*t) + \
                    self.b[l-1]*self.w_f*l*np.cos(self.w_f*l*t)
        if self.use_deg:
            ddq = np.rad2deg(ddq)
        return ddq

