from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import range
from builtins import object
import sys
import numpy as np
import numpy.linalg as la
from scipy import signal
import scipy.linalg as sla
import sympy
from sympy import symbols, Matrix

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from . import helpers

from IPython import embed
np.core.arrayprint._line_width = 160

class Model(object):
    def __init__(self, opt, urdf_file, regressor_file=None):
        self.urdf_file = urdf_file
        self.opt = opt

        if 'orthogonalizeBasis' not in self.opt:
            self.opt['orthogonalizeBasis'] = 1

        if 'useBasisProjection' not in self.opt:
            self.opt['useBasisProjection'] = 1

        if 'useRBDL' not in self.opt:
            self.opt['useRBDL'] = 0

        # create generator instance and load model
        self.generator = iDynTree.DynamicsRegressorGenerator()
        ret = self.generator.loadRobotAndSensorsModelFromFile(urdf_file)
        if not ret:
            sys.exit()

        # load also with new model class for some functions
        #self.idyn_model = iDynTree.Model()
        #iDynTree.modelFromURDF(urdf_file, self.idyn_model)

        #viz = iDynTree.Visualizer()
        #viz.addModel(self.idyn_model, 'model')
        #for i in range(0,30):
            #model_inst = viz.modelViz('model')
            #model_inst.setPositions(world_H_base, VectorDynSize jointPos)
        #    viz.draw()
        #viz.close()

        if self.opt['verbose']:
            print('loaded model {}'.format(urdf_file))

        # define what regressor type
        if regressor_file:
            with open(regressor_file, 'r') as filename:
                regrXml = filename.read()

            self.jointNames = []
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(regrXml)
            for l in tree.iter():
                if l.tag == 'joint':
                    self.jointNames.append(l.text)
            self.N_DOFS = len(self.jointNames)
        else:
            # (default for all joints)
            if self.opt['floatingBase']:
                regrXml = '''
                <regressor>
                  <baseLinkDynamics/>
                  <jointTorqueDynamics>
                    <allJoints/>
                  </jointTorqueDynamics>
                </regressor>'''
            else:
                regrXml = '''
                <regressor>
                  <jointTorqueDynamics>
                    <allJoints/>
                  </jointTorqueDynamics>
                </regressor>'''
        self.generator.loadRegressorStructureFromString(regrXml)
        self.regrXml = regrXml

        if not regressor_file:
            import re
            self.jointNames = re.sub(r"DOF Index: \d+ Name: ", "", self.generator.getDescriptionOfDegreesOfFreedom()).split()
            self.N_DOFS = self.generator.getNrOfDegreesOfFreedom()

        # TODO: reported dofs and links are not dependent on joints specified in regressor (but
        # uses all from model file)
        # dynComp simulates with all joints regardless of regressor, regressor rows should be as specified
        # (worked around ATM by reading from XML directly)
        if self.opt['verbose']:
            print('# DOFs: {}'.format(self.N_DOFS))
            print('Joints: {}'.format(self.jointNames))
            #print('Joints: {}'.format(self.generator.getDescriptionOfDegreesOfFreedom().replace(r"DOF Index:", "").replace("Name: ", "").replace("\n", " ")))
            #print('\nJoints: {}'.format([self.idyn_model.getJointName(i) for i in range(0, self.idyn_model.getNrOfDOFs())]))

        # Get the number of outputs of the regressor
        # (should eq #dofs + #base vals)
        self.N_OUT = self.generator.getNrOfOutputs()
        if self.opt['verbose']:
            if self.opt['floatingBase']:
                print('# outputs: {} (DOFs + 6 base)'.format(self.N_OUT))
            else:
                print('# outputs: {}'.format(self.N_OUT))

        self.linkNames = self.generator.getDescriptionOfLinks().split()
        if self.opt['verbose']:
            print('({})'.format(self.linkNames))

        self.N_LINKS = self.generator.getNrOfLinks()-self.generator.getNrOfFakeLinks()
        if self.opt['verbose']:
            print('# links: {} (+ {} fake)'.format(self.N_LINKS, self.generator.getNrOfFakeLinks()))

        # get initial inertia params (from urdf)
        self.num_params = self.generator.getNrOfParameters()
        # params counted without offset params
        self.num_inertial_params = self.num_params
        # add N offset params (offsets or constant friction) and 2*N velocity dependent friction params
        # (velocity +- for asymmetrical friction)
        if self.opt['identifyFriction']: self.num_params += 3*self.N_DOFS

        if self.opt['verbose']:
            print('# params: {}'.format(self.num_params))

        self.baseNames = ['base f_x', 'base f_y', 'base f_z', 'base m_x', 'base m_y', 'base m_z']

        self.gravity_twist = iDynTree.Twist.fromList([0,0,-9.81,0,0,0])

        if opt['simulateTorques'] or opt['useAPriori'] or opt['floatingBase']:
            if self.opt['useRBDL']:
                import rbdl
                self.rbdlModel = rbdl.loadModel(self.urdf_file, floating_base=self.opt['floatingBase'], verbose=False)
                self.rbdlModel.gravity = np.array([0, 0, -9.81])
            self.dynComp = iDynTree.DynamicsComputations()
            self.dynComp.loadRobotModelFromFile(self.urdf_file)

        # get model parameters
        xStdModel = iDynTree.VectorDynSize(self.num_inertial_params)
        self.generator.getModelParameters(xStdModel)
        self.xStdModel = xStdModel.toNumPy()
        if self.opt['identifyFriction']:
            self.xStdModel = np.concatenate((self.xStdModel, np.zeros(3*self.N_DOFS)))
        if opt['estimateWith'] == 'urdf':
            self.xStd = self.xStdModel

        # get model dependent projection matrix and linear column dependencies (i.e. base
        # groupings)
        # (put here so it's only done once for the loaded model)
        self.computeRegressorLinDepsQR()

    def simulateDynamicsRBDL(self, samples, sample_idx, dynComp=None):
        import rbdl

        # read sample data
        q = samples['positions'][sample_idx]
        qdot = samples['velocities'][sample_idx]
        qddot = samples['accelerations'][sample_idx]
        tau = samples['torques'][sample_idx].copy()

        '''
        if self.opt['floatingBase']:
            # The twist (linear/angular velocity) of the base, expressed in the world
            # orientation frame and with respect to the base origin
            base_velocity = samples['base_velocity'][sample_idx]
            # The 6d classical acceleration (linear/angular acceleration) of the base
            # expressed in the world orientation frame and with respect to the base
            # origin
            base_acceleration = samples['base_acceleration'][sample_idx]
            rpy = samples['base_rpy'][sample_idx]

            # get the homogeneous transformation that transforms vectors expressed
            # in the base reference frame to frames expressed in the world
            # reference frame, i.e. pos_world = world_T_base*pos_base
            # for identification purposes, the position does not matter but rotation is taken
            # from IMU estimation. The gravity, base velocity and acceleration all need to be
            # expressed in world frame then
            rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
            pos = iDynTree.Position.Zero()
            world_T_base = iDynTree.Transform(rot, pos)

            dynComp.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration,
                                  world_gravity)
            #TODO: how to set base vel,acc and rotation with rbdl?
        '''

        # compute inverse dynamics with rbdl
        rbdl.InverseDynamics(self.rbdlModel, q, qdot, qddot, tau)
        return tau


    def simulateDynamicsIDynTree(self, samples, sample_idx, dynComp=None):
        """ compute torques for one time step of measurements """

        if not dynComp:
            dynComp = self.dynComp
        world_gravity = iDynTree.SpatialAcc.fromList([0, 0, -9.81, 0, 0, 0])

        # read sample data
        pos = samples['positions'][sample_idx]
        vel = samples['velocities'][sample_idx]
        acc = samples['accelerations'][sample_idx]

        if self.opt['floatingBase']:
            # The twist (linear/angular velocity) of the base, expressed in the world
            # orientation frame and with respect to the base origin
            base_vel = samples['base_velocity'][sample_idx]
            base_velocity = iDynTree.Twist.fromList(base_vel)
            # The 6d classical acceleration (linear/angular acceleration) of the base
            # expressed in the world orientation frame and with respect to the base
            # origin
            base_acc = samples['base_acceleration'][sample_idx]
            base_acceleration = iDynTree.ClassicalAcc.fromList(base_acc)
            rpy = samples['base_rpy'][sample_idx]

        # system state for iDynTree
        q = iDynTree.VectorDynSize.fromList(pos)
        dq = iDynTree.VectorDynSize.fromList(vel)
        ddq = iDynTree.VectorDynSize.fromList(acc)

        # calc torques and forces with iDynTree dynamicsComputation class
        if self.opt['floatingBase']:
            # get the homogeneous transformation that transforms vectors expressed
            # in the base reference frame to frames expressed in the world
            # reference frame, i.e. pos_world = world_T_base*pos_base
            # for identification purposes, the position does not matter but rotation is taken
            # from IMU estimation. The gravity, base velocity and acceleration all need to be
            # expressed in world frame then
            dynComp.setFloatingBase(self.opt['baseLinkName'])
            rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
            pos = iDynTree.Position.Zero()
            world_T_base = iDynTree.Transform(rot, pos)

            dynComp.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration,
                                  world_gravity)
        else:
            dynComp.setRobotState(q, dq, ddq, world_gravity)

        # compute inverse dynamics
        torques = iDynTree.VectorDynSize(self.N_DOFS)
        baseReactionForce = iDynTree.Wrench()
        dynComp.inverseDynamics(torques, baseReactionForce)

        if self.opt['floatingBase']:
            return np.concatenate((baseReactionForce.toNumPy(), torques.toNumPy()))
        else:
            return torques.toNumPy()

    def computeRegressors(self, data):
        """ compute regressors from measurements for each time step of the measurement data
            and stack them vertically. also stack measured torques and get simulation data.
            for floating base, get estimated base forces (6D wrench) and add to torque measure stack
        """

        self.data = data

        num_time = 0
        simulate_time = 0

        #extra regressor rows for floating base
        if self.opt['floatingBase']: fb = 6
        else: fb = 0
        self.regressor_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples, self.num_params))
        self.torques_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))
        self.torquesAP_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))

        num_contacts = len(data.samples['contacts'].item().keys()) if 'contacts' in data.samples else 0
        self.contacts_stack = np.zeros(shape=(num_contacts, 6*data.num_used_samples))
        self.contactForcesSum = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))

        """loop over measurement data, optionally skip some values
            - get the regressor per time step
            - if necessary, calculate inverse dynamics to get simulated torques
            - if necessary, get torques from contact forces and add them to the torques
            - stack the torques, regressors and contacts into matrices
        """
        contacts = {}
        for sample_index in range(0, data.num_used_samples):
            m_idx = sample_index*(self.opt['skipSamples'])+sample_index
            with helpers.Timer() as t:
                # read samples
                pos = data.samples['positions'][m_idx]
                vel = data.samples['velocities'][m_idx]
                acc = data.samples['accelerations'][m_idx]
                torq = data.samples['torques'][m_idx]
                if 'contacts' in data.samples:
                    for frame in data.samples['contacts'].item(0).keys():
                        contacts[frame] = data.samples['contacts'].item(0)[frame][m_idx]

                # system state for iDynTree
                q = iDynTree.VectorDynSize.fromList(pos)
                dq = iDynTree.VectorDynSize.fromList(vel)
                ddq = iDynTree.VectorDynSize.fromList(acc)

                # in case that we simulate the torque measurements, need torque estimation for a priori parameters
                # or that we need to simulate the base reaction forces for floating base
                if self.opt['simulateTorques'] or self.opt['useAPriori'] or self.opt['floatingBase']:
                    if self.opt['useRBDL']:
                        torques = self.simulateDynamicsRBDL(data.samples, m_idx)
                    else:
                        torques = self.simulateDynamicsIDynTree(data.samples, m_idx)

                    if self.opt['useAPriori']:
                        # torques sometimes contain nans, just a very small C number that gets converted to nan?
                        torqAP = np.nan_to_num(torques)

                    if self.opt['simulateTorques']:
                        torq = np.nan_to_num(torques)
                    else:
                        if self.opt['floatingBase']:
                            #add estimated base forces to measured torq vector from file
                            torq = np.concatenate((np.nan_to_num(torques[0:6]), torq))

                #if self.opt['addNoise'] != 0:
                #    torq += np.random.randn(self.N_DOFS+fb)*self.opt['addNoise']

            simulate_time += t.interval

            #...still looping over measurement samples

            # get numerical regressor (std)
            row_index = (self.N_DOFS+fb)*sample_index   # index for current row in stacked regressor matrix
            with helpers.Timer() as t:
                if self.opt['floatingBase']:
                    vel = data.samples['base_velocity'][m_idx]
                    acc = data.samples['base_acceleration'][m_idx]
                    rpy = data.samples['base_rpy'][m_idx]
                    base_velocity = iDynTree.Twist.fromList(vel)
                    base_acceleration = iDynTree.Twist.fromList(acc)
                    rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
                    pos = iDynTree.Position.Zero()
                    world_T_base = iDynTree.Transform(rot, pos)
                    self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, self.gravity_twist)
                else:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)

                # get (standard) regressor
                regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_inertial_params)
                knownTerms = iDynTree.VectorDynSize(self.N_OUT)   # what are known terms useable for?
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print("Error during numeric computation of regressor")

                regressor = regressor.toNumPy()
                # the base forces are expressed in the base frame for the regressor, so transform them
                # (inverse dynamics use world frame)
                if self.opt['floatingBase']:
                    to_world = np.fromstring(world_T_base.getRotation().toString(), sep=' ').reshape((3,3))
                    regressor[0:3, :] = to_world.dot(regressor[0:3, :])
                    regressor[3:6, :] = to_world.dot(regressor[3:6, :])

                if self.opt['identifyFriction']:
                    # append unitary matrix to regressor for offsets/constant friction
                    static_diag = np.identity(self.N_DOFS)*np.sign(dq.toNumPy())
                    offset_regressor = np.vstack( (np.zeros((fb, self.N_DOFS)), static_diag))
                    regressor = np.concatenate((regressor, offset_regressor), axis=1)

                    # append positive/negative velocity matrix for velocity dependent asymmetrical friction
                    dq_p = dq.toNumPy().copy()
                    dq_p[dq_p < 0] = 0 #set to zero where v < 0
                    dq_m = dq.toNumPy().copy()
                    dq_m[dq_m > 0] = 0 #set to zero where v > 0
                    vel_diag = np.hstack((np.identity(self.N_DOFS)*dq_p, np.identity(self.N_DOFS)*dq_m))
                    friction_regressor = np.vstack( (np.zeros((fb, self.N_DOFS*2)), vel_diag))   # add base dynamics rows
                    regressor = np.concatenate((regressor, friction_regressor), axis=1)

                # stack on previous regressors
                np.copyto(self.regressor_stack[row_index:row_index+self.N_DOFS+fb], regressor)
            num_time += t.interval

            # stack results onto matrices of previous time steps
            np.copyto(self.torques_stack[row_index:row_index+self.N_DOFS+fb], torq)
            if self.opt['useAPriori']:
                np.copyto(self.torquesAP_stack[row_index:row_index+self.N_DOFS+fb], torqAP)

            contact_idx = (sample_index*6)
            for i in range(self.contacts_stack.shape[0]):
                frame = list(contacts.keys())[i]
                np.copyto(self.contacts_stack[i][contact_idx:contact_idx+6], contacts[frame])

        if len(contacts.keys()):
            # TODO: if robot does not have contact sensors, use HyQ null-space method (only for
            # static positions?)

            #convert contact forces into torque contribution
            for i in range(self.contacts_stack.shape[0]):
                frame = list(contacts.keys())[i]
                if frame == 'dummy_sim':  #ignore empty contacts from simulation
                    print("Empty contacts data!")
                    continue

                # get jacobian and contact force for each contact frame and measurement sample
                jacobian = iDynTree.MatrixDynSize(6, 6+self.N_DOFS)
                self.dynComp.getFrameJacobian(frame, jacobian)
                jacobian = jacobian.toNumPy()

                # mul each sample of measured contact forces with frame jacobian
                dim = self.N_DOFS+fb
                contacts_torq = np.empty(dim*self.data.num_used_samples)
                for s in range(self.data.num_used_samples):
                    contacts_torq[s*dim:(s+1)*dim] = jacobian.T.dot(self.contacts_stack[i][s*6:(s+1)*6])
                self.contactForcesSum += contacts_torq
            self.contactForcesSum_2dim = np.reshape(self.contactForcesSum, (data.num_used_samples, self.N_DOFS+6))

            #reshape torque stack
            torques_stack_2dim = np.reshape(self.torques_stack, (data.num_used_samples, self.N_DOFS+fb))

            #subtract measured contact forces from torque estimation from iDynTree
            if self.opt['simulateTorques']:
                self.torques_stack -= self.contactForcesSum
                #torques_stack_2dim[:, 6:] -= self.contactForcesSum_2dim[:, 6:]
                #self.torques_stack = torques_stack_2dim.flatten()
            else:
                # if not simulating, measurements of joint torques already contain contact contribution,
                # so only add it to the (simulated) base force estimation
                torques_stack_2dim[:, :6] -= self.contactForcesSum_2dim[:, :6]
                self.torques_stack = torques_stack_2dim.flatten()
            self.data.samples['torques'] = torques_stack_2dim[:, 6:]
        else:
            # also write back torques if simulating and fixed-base
            if self.opt['simulateTorques']:
                self.data.samples['torques'] = np.reshape(self.torques_stack, (data.num_used_samples, self.N_DOFS+fb))

        with helpers.Timer() as t:
            if self.opt['useAPriori']:
                # get torque delta to identify with
                self.tau = self.torques_stack - self.torquesAP_stack
            else:
                self.tau = self.torques_stack
        simulate_time+=t.interval

        self.YStd = self.regressor_stack
        if self.opt['useBasisProjection']:
            self.YBase = np.dot(self.YStd, self.B)   # project regressor to base regressor
        else:
            self.YBase = np.dot(self.YStd, self.Pb)  # regressor following Sousa, 2014

        if self.opt['verbose']:
            print("YStd: {}".format(self.YStd.shape), end=' ')
            print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))

        if self.opt['filterRegressor']:
            order = 5                            # Filter order
            fs = self.data.samples['frequency']  # Sampling freq
            fc = self.opt['filterRegCutoff']     # Cut-off frequency (Hz)
            b, a = signal.butter(order, fc / (fs / 2), btype='low', analog=False)
            for j in range(0, self.num_base_inertial_params):
                for i in range(0, self.N_DOFS):
                    self.YBase[i::self.N_DOFS, j] = signal.filtfilt(b, a, self.YBase[i::self.N_DOFS, j])

        self.sample_end = data.samples['positions'].shape[0]
        if self.opt['skipSamples'] > 0: self.sample_end -= (self.opt['skipSamples'])

        # keep absolute torques (self.tau can be relative)
        self.tauMeasured = np.reshape(self.torques_stack, (data.num_used_samples, self.N_DOFS+fb))

        self.T = data.samples['times'][0:self.sample_end:self.opt['skipSamples']+1]

        if self.opt['showTiming']:
            print('Simulation for regressors took %.03f sec.' % simulate_time)
            print('Getting regressors took %.03f sec.' % num_time)


    def getRandomRegressor(self, n_samples=None):
        """
        Utility function for generating a random regressor for numerical base parameter calculation
        Given n_samples, the Y (n_samples*getNrOfOutputs() X getNrOfParameters() ) regressor is
        obtained by stacking the n_samples generated regressors
        This function returns Y^T Y (getNrOfParameters() X getNrOfParameters() ) (that share the row space with Y)
        (partly ported from iDynTree)
        """

        regr_filename = self.urdf_file + '.regressor.npz'
        generate_new = False
        fb = self.opt['floatingBase']

        try:
            regr_file = np.load(regr_filename)
            R = regr_file['R']
            n = regr_file['n']   #number of samples that were used
            fb = regr_file['fb']  #floating base flag
            if self.opt['verbose']:
                print("loaded random regressor from {}".format(regr_filename))
            if n != n_samples or fb != self.opt['floatingBase'] or R.shape[0] != self.num_params:
                generate_new = True
            #TODO: save and check timestamp of urdf file, if newer regenerate
        except (IOError, KeyError):
            generate_new = True

        if generate_new:
            if self.opt['verbose']:
                print("generating random regressor")

            if not n_samples:
                n_samples = self.N_DOFS * 5000
            R = np.array((self.N_OUT, self.num_inertial_params))
            regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_inertial_params)
            knownTerms = iDynTree.VectorDynSize(self.N_OUT)
            limits = helpers.URDFHelpers.getJointLimits(self.urdf_file, use_deg=False)
            if len(limits) > 0:
                jn = self.jointNames
                q_lim_pos = [limits[jn[n]]['upper'] for n in range(self.N_DOFS)]
                q_lim_neg = [limits[jn[n]]['lower'] for n in range(self.N_DOFS)]
                dq_lim = [limits[jn[n]]['velocity'] for n in range(self.N_DOFS)]
                q_range = (np.array(q_lim_pos) - np.array(q_lim_neg)).tolist()
            for i in range(0, n_samples):
                # set random system state
                if len(limits) > 0:
                    rnd = np.random.rand(self.N_DOFS) #0..1
                    q = iDynTree.VectorDynSize.fromList((q_lim_neg+q_range*rnd).tolist())
                    dq = iDynTree.VectorDynSize.fromList(((np.random.rand(self.N_DOFS)-0.5)*2*dq_lim).tolist())
                    ddq = iDynTree.VectorDynSize.fromList(((np.random.rand(self.N_DOFS)-0.5)*2*np.pi).tolist())
                else:
                    q = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())
                    dq = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())
                    ddq = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.N_DOFS)*2-1)*np.pi).tolist())

                # TODO: make work with fixed dofs (set vel and acc to zero, look at iDynTree method)

                if self.opt['floatingBase']:
                    base_velocity = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
                    base_acceleration = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
                    rpy = np.random.ranf(3)*0.05
                    rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
                    pos = iDynTree.Position.Zero()
                    world_T_base = iDynTree.Transform(rot, pos)
                    self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, self.gravity_twist)
                else:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)

                # get regressor
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print("Error during numeric computation of regressor")

                A = regressor.toNumPy()
                #the base forces are expressed in the base frame for the regressor, so transform them
                if self.opt['floatingBase']:
                    to_world = np.fromstring(world_T_base.getRotation().toString(), sep=' ').reshape((3,3))
                    A[0:3, :] = to_world.dot(A[0:3, :])
                    A[3:6, :] = to_world.dot(A[3:6, :])

                if self.opt['identifyFriction']:
                    # append unitary matrix to regressor for offsets/constant friction
                    static_diag = np.identity(self.N_DOFS)*np.sign(dq.toNumPy())
                    offset_regressor = np.vstack( (np.zeros((fb*6, self.N_DOFS)), static_diag))
                    A = np.concatenate((A, offset_regressor), axis=1)

                    # append positive/negative velocity matrix for velocity dependent asymmetrical friction
                    dq_p = dq.toNumPy().copy()
                    dq_p[dq_p < 0] = 0 #set to zero where <0
                    dq_m = dq.toNumPy().copy()
                    dq_m[dq_m > 0] = 0 #set to zero where >0
                    vel_diag = np.hstack((np.identity(self.N_DOFS)*dq_p, np.identity(self.N_DOFS)*dq_m))
                    friction_regressor = np.vstack( (np.zeros((fb*6, self.N_DOFS*2)), vel_diag))
                    A = np.concatenate((A, friction_regressor), axis=1)

                # add to previous regressors, linear dependencies don't change
                # (if too many, saturation or accuracy problems?)
                if i==0:
                    R = A.T.dot(A)
                else:
                    R += A.T.dot(A)

            np.savez(regr_filename, R=R, n=n_samples, fb=self.opt['floatingBase'])

        if 'showRandomRegressor' in self.opt and self.opt['showRandomRegressor']:
            import matplotlib.pyplot as plt
            plt.imshow(R, interpolation='nearest')
            plt.show()

        return R


    def computeRegressorLinDepsQR(self):
        """get base regressor and identifiable basis matrix with QR decomposition

        gets independent columns (non-unique choice) each with its dependent ones, i.e.
        those std parameter indices that form each of the base parameters (including the linear factors)
        """
        #using random regressor gives us structural base params, not dependent on excitation
        #QR of transposed gives us basis of column space of original matrix
        Yrand = self.getRandomRegressor(n_samples=5000)

        #TODO: save all this following stuff into regressor file as well

        """
        # get basis directly from regressor matrix using QR
        Qt,Rt,Pt = sla.qr(Yrand.T, pivoting=True, mode='economic')

        #get rank
        r = np.where(np.abs(Rt.diagonal()) > self.opt['minTol'])[0].size
        self.num_base_params = r

        Qt[np.abs(Qt) < self.opt['minTol']] = 0

        #get basis projection matrix
        S = np.zeros_like(Rt)
        for i in range(Rt.shape[0]):
            if np.abs(Rt[i,i]) < self.opt['minTol']:
                continue
            if Rt[i,i] < 0:
                S[i,i] = -1
            if Rt[i,i] > 0:
                S[i,i] = 1
        self.B = Qt.dot(S)[:, :r]
        #self.B = Qt[:, 0:r]

        """

        #get basis use Gautier, 1990 way, also using QR decomposition

        # get column space dependencies
        Q,R,P = sla.qr(Yrand, pivoting=True, mode='economic')
        self.Q, self.R, self.P = Q,R,P

        #get rank
        r = np.where(np.abs(R.diagonal()) > self.opt['minTol'])[0].size
        self.num_base_params = r
        self.num_base_inertial_params = r - self.N_DOFS

        #create proper permutation matrix from vector
        self.Pp = np.zeros((P.size, P.size))
        for i in P:
            self.Pp[i, P[i]] = 1
        self.Pb = self.Pp.T[:, 0:self.num_base_params]
        self.Pd = self.Pp.T[:, self.num_base_params:]

        # get the choice of indices of "independent" columns of the regressor matrix
        # (representants chosen from each separate interdependent group of columns)
        self.independent_cols = P[0:r]

        # get column dependency matrix (with what factor are columns of "dependent" columns grouped)
        # i (independent column) = (value at i,j) * j (dependent column index among the others)
        R1 = R[0:r, 0:r]
        R2 = R[0:r, r:]
        self.linear_deps = sla.inv(R1).dot(R2)
        self.linear_deps[np.abs(self.linear_deps) < self.opt['minTol']] = 0

        self.Kd = self.linear_deps
        self.K = self.Pb.T + self.Kd.dot(self.Pd.T)

        # collect grouped columns for each independent column
        # and build base matrix
        self.B = np.zeros((self.num_params, self.num_base_params))
        for j in range(0, self.linear_deps.shape[0]):
            indep_idx = self.independent_cols[j]
            for i in range(0, self.linear_deps.shape[1]):
                for k in range(r, P.size):
                    #factor = round(self.linear_deps[j, k-r], 5)
                    factor = self.linear_deps[j, k-r]
                    if np.abs(factor)>self.opt['minTol']: self.B[P[k],j] = factor
            self.B[indep_idx,j] = 1

        if self.opt['orthogonalizeBasis']:
            #orthogonalize, so linear relationships can be inverted (if B is square, will orthonormalize)
            Q_B_qr, R_B_qr = la.qr(self.B)
            Q_B_qr[np.abs(Q_B_qr) < self.opt['minTol']] = 0
            S = np.zeros_like(R_B_qr)
            for i in range(R_B_qr.shape[0]):
                if np.abs(R_B_qr[i,i]) < self.opt['minTol']:
                    continue
                if R_B_qr[i,i] < 0:
                    S[i,i] = -1
                if R_B_qr[i,i] > 0:
                    S[i,i] = 1
            self.B = Q_B_qr.dot(S)
            #self.B = Q_B_qr
            self.Binv = self.B.T
        else:
            # in case B is not an orthogonal base (B.T != B^-1), we have to use pinv instead of T
            # (using QR on B yields orthonormal base if necessary)
            # in general, pinv is always working (but is numerically a bit different)
            self.Binv = la.pinv(self.B)

        # define sympy symbols for each std column
        self.base_syms = sympy.Matrix([sympy.Symbol('beta'+str(i),real=True) for i in range(self.num_base_params)])
        self.param_syms = list()
        self.mass_syms = list()
        for i in range(0, self.N_LINKS):
            #mass
            m = symbols('m_{}'.format(i))
            self.param_syms.append(m)
            self.mass_syms.append(m)

            #first moment of mass
            p = 'c_{}'.format(i)  #symbol prefix
            syms = [symbols(p+'x'), symbols(p+'y'), symbols(p+'z')]
            self.param_syms.extend(syms)
            #3x3 inertia tensor about link-frame (for link i)
            p = 'I_{}'.format(i)
            syms = [symbols(p+'xx'), symbols(p+'xy'), symbols(p+'xz'),
                    symbols(p+'xy'), symbols(p+'yy'), symbols(p+'yz'),
                    symbols(p+'xz'), symbols(p+'yz'), symbols(p+'zz')
                   ]
            self.param_syms.extend([syms[0], syms[1], syms[2], syms[4], syms[5], syms[8]])

        if self.opt['identifyFriction']:
            for i in range(0,self.N_DOFS):
                self.param_syms.extend([symbols('Fc_{}'.format(i))])
            for i in range(0,self.N_DOFS):
                self.param_syms.extend([symbols('Fv+_{}'.format(i))])
            for i in range(0,self.N_DOFS):
                self.param_syms.extend([symbols('Fv-_{}'.format(i))])
        self.param_syms = np.array(self.param_syms)

        ## get symbolic equations for base param dependencies
        # Each dependent parameter can be ignored (non-identifiable) or it can be
        # represented by grouping some base and/or dependent parameters.
        if self.opt['useBasisProjection']:
            if self.opt['orthogonalizeBasis']:
                #this is only correct if basis is orthogonal
                self.base_deps = np.dot(self.param_syms, self.B)
            else:
                #otherwise, we need to get relationships from the inverse
                B_qr_inv_z = la.pinv(self.B)
                B_qr_inv_z[np.abs(B_qr_inv_z) < self.opt['minTol']] = 0
                self.base_deps = np.dot(self.param_syms, B_qr_inv_z.T)
        else:
            # using projection matrix from Gautier/Sousa method for base eqns
            # (seems K is orthogonal)
            self.base_deps = Matrix(self.K) * Matrix(self.param_syms)

        # find std parameters that have no effect on estimation (not single or contributing to base
        # equations)
        base_deps_syms = []
        for i in range(self.base_deps.shape[0]):
            for s in self.base_deps[i].free_symbols:
                if s not in base_deps_syms:
                    base_deps_syms.append(s)
        self.non_id = [p for p in range(self.num_params) if self.param_syms[p] not in base_deps_syms]
        self.identifiable = [p for p in range(self.num_params) if p not in self.non_id]

    def computeRegressorLinDepsSVD(self):
        """get base regressor and identifiable basis matrix with iDynTree (SVD)"""

        with helpers.Timer() as t:
            # get subspace basis (for projection to base regressor/parameters)
            Yrand = self.getRandomRegressor(5000)
            #A = iDynTree.MatrixDynSize(self.num_params, self.num_params)
            #self.generator.generate_random_regressors(A, False, True, 2000)
            #Yrand = A.toNumPy()
            U, s, Vh = la.svd(Yrand, full_matrices=False)
            r = np.sum(s>self.opt['minTol'])
            self.B = -Vh.T[:, 0:r]
            self.num_base_params = r

            print("tau: {}".format(self.tau.shape), end=' ')

            print("YStd: {}".format(self.YStd.shape), end=' ')
            # project regressor to base regressor, Y_base = Y_std*B
            self.YBase = np.dot(self.YStd, self.B)
            if self.opt['verbose']:
                print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))

            self.num_base_params = self.YBase.shape[1]
        if self.showTiming:
            print("Getting the base regressor (iDynTree) took %.03f sec." % t.interval)


    def getSubregressorsConditionNumbers(self):
        # get condition number for each of the links
        linkConds = list()
        for i in range(0, self.N_LINKS):
            #get columns of base regressor that are dependent on std parameters of link i
            # TODO: try going further down to e.g. condition number of link mass, com, inertial
            # and ignore those groups of params

            ## get parts of base regressor with only independent columns (identifiable space)

            #get all independent std columns for link i
            #base_columns = [j for j in range(0, self.num_base_params) \
            #                      if self.independent_cols[j] in range(i*10, i*10+9+1)]

            # use base column dependencies to get combined params of base regressor with
            # coontribution on each each link (a bit inexact I guess)
            base_columns = list()
            for k in range(i*10, i*10+9+1):
                for j in range(0, self.num_base_params):
                    if self.param_syms[k] in self.base_deps[j].free_symbols:
                        if j not in base_columns:
                            base_columns.append(j)
                        continue

            if not len(base_columns):
                linkConds.append(1e16)
            else:
                linkConds.append(la.cond(self.YBase[:, base_columns]))

        print("Condition numbers of link sub-regressors: [{}]".format(dict(enumerate(linkConds))))

        return linkConds
