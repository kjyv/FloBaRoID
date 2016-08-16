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
        #walkman left arm
        #a = [[-0.2], [0.5], [-0.8], [0.5], [1], [-0.7], [-0.8], [-0.8]]
        #b = [[0.9], [0.9], [1.5], [0.8], [1], [1.3], [0.8], [0.8]]
        #q = [10, 50, -80, -25, 50, 0, -15, -15]
        #kuka lwr4+
        #nf = [1,1,1,1,1,1,1]
        #a = [[-0.7], [0.4], [-1.2], [-0.7], [0.8], [-1.3], [-0.9], [1.3]]
        #b = [[0.7], [0.4], [1.2], [0.7], [0.8], [1.3], [0.9], [0.3]]
        #q = [0, 0, 0, 0, 0, 0, 0, 0]
        #a = [[-0.7], [0.8], [-1.2], [-0.7], [0.8], [1.0], [-0.9], [1.3]]
        #b = [[0.7], [0.8], [1.2], [0.7], [0.8], [-1.0], [0.9], [0.3]]
        #q = [0, 0, 1, 1, 0, 0, 0, 0]
        #a = [[-0.0], [1.0], [-1.2], [-0.7], [0.8], [-1.3], [-1.0], [1.3]]
        #b = [[1.0], [0.0], [1.2], [0.7], [0.8], [1.3], [1.0], [1.3]]
        #q = [-1.0, -1.0, 0, 0, 0, 0, -1, 0]

        self.setRandomParams()

    def setRandomParams(self):
        #use random params
        a = [0]*self.dofs
        b = [0]*self.dofs
        nf = np.random.randint(1,4, self.dofs)
        q = np.random.rand(self.dofs)*2-1
        for i in range(0, self.dofs):
            max = 2.0-np.abs(q[i])
            a[i] = np.random.rand(nf[i])*max-max/2
            b[i] = np.random.rand(nf[i])*max-max/2
        #print a
        #print b
        #print q

        if not self.use_deg:
            q = np.deg2rad(q)

        for i in range(0, self.dofs):
            self.oscillators.append(OscillationGenerator(w_f = self.w_f_global, a = np.array(a[i]),
                                                         b = np.array(b[i]), q0 = q[i], nf = nf[i], use_deg = self.use_deg
                                                        ))

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
        self.q0 = float(q0)
        self.nf = nf
        self.use_deg = use_deg

    def getAngle(self, t):
        #- t is the current time
        q = 0
        for l in range(1, self.nf+1):
            q += (self.a[l-1]/(self.w_f*l))*np.sin(self.w_f*l*t) - \
                 (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l*t)
        if self.use_deg:
            q = np.rad2deg(q)
        q += self.nf*self.q0
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

