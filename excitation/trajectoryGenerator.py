from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from identification.model import Model
from identification.data import Data


def simulateTrajectory(config, trajectory, model=None, measurements=None):
    # type: (Dict, Trajectory, Model, np._ArrayLike) -> Tuple[Dict, Data]
    # generate data arrays for simulation and regressor building
    old_sim = config['simulateTorques']
    config['simulateTorques'] = True

    if config['floatingBase']: fb = 6
    else: fb = 0

    if not model:
        if 'urdf_real' in config and config['urdf_real']:
            print('Simulating using "real" model parameters.')
            urdf = config['urdf_real']
        else:
            urdf = config['urdf']

        model = Model(config, urdf)

    data = Data(config)
    trajectory_data = {}   # type: Dict[str, Union[List, np._ArrayLike]]
    trajectory_data['target_positions'] = []
    trajectory_data['target_velocities'] = []
    trajectory_data['target_accelerations'] = []
    trajectory_data['torques'] = []
    trajectory_data['times'] = []

    freq = config['excitationFrequency']
    for t in range(0, int(trajectory.getPeriodLength()*freq)):
        trajectory.setTime(t/freq)
        q = np.array([trajectory.getAngle(d) for d in range(config['num_dofs'])])
        if config['useDeg']:
            q = np.deg2rad(q)
        trajectory_data['target_positions'].append(q)

        qdot = np.array([trajectory.getVelocity(d) for d in range(config['num_dofs'])])
        if config['useDeg']:
            qdot = np.deg2rad(qdot)
        trajectory_data['target_velocities'].append(qdot)

        qddot = np.array([trajectory.getAcceleration(d) for d in range(config['num_dofs'])])
        if config['useDeg']:
            qddot = np.deg2rad(qddot)
        trajectory_data['target_accelerations'].append(qddot)

        trajectory_data['times'].append(t/freq)
        trajectory_data['torques'].append(np.zeros(config['num_dofs']+fb))

    num_samples = len(trajectory_data['times'])

    #convert lists to numpy arrays
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

    # add static contact force
    contacts = 1
    if config['floatingBase'] and contacts:
        contactFrame = 'contact_ft'
        # get base acceleration that results from acceleration at contact frame
        #contact_wrench =  np.zeros(6)
        #contact_wrench[2] = 9.81 * 3.0 # -g * mass
        contact_wrench = np.random.rand(6) * 10

        trajectory_data['base_rpy'] = np.random.rand((3*num_samples)).reshape((num_samples, 3))*0.5

        """
        contact_wrench[2] = 9.81 # * 139.122814
        len_contact = la.norm(contact_wrench[0:3])

        # get vector from contact frame to robot center of mass
        model_com = iDynTree.Position()
        model_com = model.dynComp.getWorldTransform(contactFrame).inverse()*model.dynComp.getCenterOfMass()
        model_com = model_com.toNumPy()

        # rotate contact wrench to be in line with COM (so that base link is not rotationally accelerated)
        contact_wrench[0:3] = (-model_com) / la.norm(model_com) * len_contact

        # rotate base accordingly (ore rather the whole robot) so gravity is parallel to contact force
        a_xz = np.array([0,9.81])      #xz of non-rotated contact force
        b_xz = contact_wrench[[0,2]]     #rotated contact force vec projected to xz
        pitch = np.arccos( (a_xz.dot(b_xz))/(la.norm(a_xz)*la.norm(b_xz)) )   #angle in xz layer
        a_yz = np.array([0,9.81])        #yz of non-rotated contact force
        b_yz = contact_wrench[[1,2]]     #rotated contact force vec projected to yz
        roll = np.arccos( (a_yz.dot(b_yz))/(la.norm(a_yz)*la.norm(b_yz)) )   #angle in yz layer
        yaw = 0

        trajectory_data['base_rpy'][:] += np.array([roll, pitch, yaw])
        """
        trajectory_data['contacts'] = np.array({contactFrame: np.tile(contact_wrench, (num_samples,1))})
    else:
        #TODO: add proper simulated contacts (from e.g. gazebo) for floating-base
        trajectory_data['contacts'] = np.array({})

    if measurements:
        trajectory_data['positions'] = measurements['Q']
        trajectory_data['velocities'] = measurements['V']
        trajectory_data['accelerations'] = measurements['Vdot']
        trajectory_data['measured_frequency'] = measurements['measured_frequency']

    old_skip = config['skipSamples']
    config['skipSamples'] = 0
    old_offset = config['startOffset']
    config['startOffset'] = 0
    data.init_from_data(trajectory_data)
    model.computeRegressors(data)
    trajectory_data['torques'][:,:] = data.samples['torques'][:,:]

    '''
    if config['floatingBase']:
        # add force of contact to keep robot fixed in space (always accelerate exactly against gravity)
        # floating base orientation has to be rotated so that accelerations resulting from hanging
        # are zero, i.e. the vector COM - contact point is parallel to gravity.
        if contacts:
            # get jacobian of contact frame at current posture
            dim = model.num_dofs+fb
            jacobian = iDynTree.MatrixDynSize(6, dim)
            model.dynComp.getFrameJacobian(contactFrame, jacobian)
            jacobian = jacobian.toNumPy()

            # get base link vel and acc and torques that result from contact force / acceleration
            contacts_torq = np.zeros(dim)
            contacts_torq = jacobian.T.dot(contact_wrench)
            trajectory_data['base_acceleration'] += contacts_torq[0:6]  # / 139.122
            data.samples['base_acceleration'][:,:] = trajectory_data['base_acceleration'][:,:]
            # simulate again with proper base acceleration
            model.computeRegressors(data, only_simulate=True)
            trajectory_data['torques'][:,:] = data.samples['torques'][:,:]
    '''

    config['skipSamples'] = old_skip
    config['startOffset'] = old_offset
    config['simulateTorques'] = old_sim

    return trajectory_data, data


class Trajectory(object):
    ''' base trajectory class '''
    def getAngle(self, dof):
        raise NotImplementedError()

    def getVelocity(self, dof):
        raise NotImplementedError()

    def getAcceleration(self, dof):
        raise NotImplementedError()

    def getPeriodLength(self):
        raise NotImplementedError()

    def setTime(self, time):
        raise NotImplementedError()

    def wait_for_zero_vel(self, t_elapsed):
        raise NotImplementedError()


class PulsedTrajectory(Trajectory):
    ''' pulsating trajectory generator for one joint using fourier series from
        Swevers, Gansemann (1997). Gives values for one time instant (at the current
        internal time value)
    '''
    def __init__(self, dofs, use_deg=False):
        # type: (List, bool) -> None
        self.dofs = dofs
        self.oscillators = list()  # type: List[OscillationGenerator]
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
        return 2 * np.pi / self.w_f_global

    def setTime(self, time):
        '''set current time in seconds'''
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        self.setTime(t_elapsed)
        if self.use_deg: thresh = 5.0
        else: thresh = np.deg2rad(5.0)
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
        self.use_deg = use_deg
        self.q0 = float(q0)
        if use_deg:
            self.q0 = np.deg2rad(self.q0)
        self.nf = nf

    def getAngle(self, t):
        #- t is the current time
        q = 0.0
        for l in range(1, self.nf+1):
            q += (self.a[l-1]/(self.w_f*l))*np.sin(self.w_f*l*t) - \
                 (self.b[l-1]/(self.w_f*l))*np.cos(self.w_f*l*t)
        q += self.nf*self.q0
        if self.use_deg:
            q = np.rad2deg(q)
        return q

    def getVelocity(self, t):
        dq = 0.0
        for l in range(1, self.nf+1):
            dq += self.a[l-1]*np.cos(self.w_f*l*t) + \
                  self.b[l-1]*np.sin(self.w_f*l*t)
        if self.use_deg:
            dq = np.rad2deg(dq)
        return dq

    def getAcceleration(self, t):
        ddq = 0.0
        for l in range(1, self.nf+1):
            ddq += -self.a[l-1]*self.w_f*l*np.sin(self.w_f*l*t) + \
                    self.b[l-1]*self.w_f*l*np.cos(self.w_f*l*t)
        if self.use_deg:
            ddq = np.rad2deg(ddq)
        return ddq


class FixedPositionTrajectory(Trajectory):
    """ generate static 'trajectories' """
    def __init__(self, config):
        # type: (Dict) -> None
        self.config = config
        self.time = 0.0
        self.use_deg = self.config['useDeg']
        self.angles = None  # type: List[Dict[str, any]]

    def initWithAngles(self, angles):
        # type: (List[Dict[str, Any]]) -> None
        ''' angles is a list containing for each posture a dict {
            start_time: float    # starting time in seconds of posture
            angles: List[float]  # angles for each joint
        }
        '''
        self.angles = angles
        self.posLength = angles[1]['start_time'] - angles[0]['start_time']

    def getAngle(self, dof):
        # type: (int) -> float
        """ get angle at current time for joint dof """

        if np.any(self.angles):
            for angle_set in self.angles:
                if angle_set['start_time'] >= self.time - self.posLength:
                    return angle_set['angles'][dof]

            # if no angle found (shouldn't happen)
            print('Warning: no angle found for time {}'.format(self.time))
            return 0.0
        else:
            # Walk-Man:
            # ['LHipLat', 'LHipYaw', 'LHipSag', 'LKneeSag', 'LAnkSag', 'LAnkLat',
            #  'RHipLat', 'RHipYaw', 'RHipSag', 'RKneeSag', 'RAnkSag', 'RAnkLat',
            #  'WaistSag', 'WaistYaw', #WaistLat is fixed atm
            #  'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2',
            #  'RShSag', 'RShLat', 'RShYaw', 'RElbj', 'RForearmPlate', 'RWrj1', 'RWrj2']
            '''
            # posture #0
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,      #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #1
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    20.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    20.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #2
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    85.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    85.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #3
            return [0.0, 0.0, -70.0, 90.0, -79.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -79.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 90.0, 0.0, -90.0, 0.0, -45.0, 0.0,    #left arm
                    0.0, -90.0, 0.0, -90.0, 0.0, -45.0, 0.0,   #right arm
                    ][dof]
            # posture #4
            return [44.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    -44.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 45.0, 0.0, -90.0, 0.0, 0.0, 79.0,    #left arm
                    0.0, -45.0, 0.0, -90.0, 0.0, 0.0, -79.0,   #right arm
                    ][dof]
            # posture #5
            return [44.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    -44.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    35.0, 0.0,                           #Waist
                    20.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    20.0, -45.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #6
            return [35.0, 0.0, 0.0, 130.0, -20.0, 0.0,       #left leg
                    -35.0, 0.0, 0.0, 130.0, -20.0, 0.0,         #right leg
                    -20.0, -45.0,                           #Waist
                    20.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    85.0, -45.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            '''
            # posture #7
            return [20.0, 0.0, -85.0, 0.0, 0.0, 0.0,       #left leg
                    -20.0, 0.0, -85.0, 0.0, 0.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]


    def getVelocity(self, dof):
        """ get velocity at current time for joint dof """
        return 0.0

    def getAcceleration(self, dof):
        """ get acceleration at current time for joint dof """
        return 0.0

    def getPeriodLength(self):
        ''' get the length of the trajectory in seconds '''
        return self.angles[1]['start_time']*len(self.angles)

    def setTime(self, time):
        '''set current time in seconds'''
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        return True

