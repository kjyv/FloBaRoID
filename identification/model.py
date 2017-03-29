from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import range
from builtins import object
import sys
from typing import Any, Dict, List

import numpy as np
import numpy.linalg as la
from scipy import signal
import scipy.linalg as sla
import sympy
from sympy import symbols, Matrix

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
import identification.helpers as helpers
from identification.quaternion import Quaternion
from identification.data import Data

from IPython import embed

np.core.arrayprint._line_width = 160

class Model(object):
    def __init__(self, opt, urdf_file, regressor_file=None, regressor_init=True):
        # (Dict[str, Any, str, str]) -> None
        self.urdf_file = urdf_file
        self.opt = opt

        progress_inst = helpers.Progress(opt)
        self.progress = progress_inst.progress

        # set these options in case model was not instanciated from Identification
        if 'orthogonalizeBasis' not in self.opt:
            self.opt['orthogonalizeBasis'] = 1

        if 'useBasisProjection' not in self.opt:
            self.opt['useBasisProjection'] = 0

        # debug options
        self.opt['useRegressorForSimulation'] = 0
        self.opt['addContacts'] = 1

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

            self.jointNames = []   # type: List[str]
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(regrXml)
            for l in tree.iter():
                if l.tag == 'joint':
                    self.jointNames.append(l.text)
            self.num_dofs = len(self.jointNames)
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
            self.num_dofs = self.generator.getNrOfDegreesOfFreedom()

        # TODO: reported dofs and links are not dependent on joints specified in regressor (but
        # uses all from model file)
        # dynComp simulates with all joints regardless of regressor, regressor rows should be as specified
        # (worked around ATM by reading from XML directly)
        if self.opt['verbose']:
            print('# DOFs: {}'.format(self.num_dofs))
            print('Joints: {}'.format(self.jointNames))

        # Get the number of outputs of the regressor
        # (should eq #dofs + #base vals)
        self.N_OUT = self.generator.getNrOfOutputs()
        if self.opt['verbose']:
            if self.opt['floatingBase']:
                print('# regressor outputs: {} (DOFs + 6 base)'.format(self.N_OUT))
            else:
                print('# regressor outputs: {}'.format(self.N_OUT))


        self.num_links = self.generator.getNrOfLinks()-self.generator.getNrOfFakeLinks()
        if self.opt['verbose']:
            print('# links: {} (+ {} fake)'.format(self.num_links, self.generator.getNrOfFakeLinks()))

        self.inertia_params = list()  # type: List[int]
        self.mass_params = list()     # type: List[int]
        for i in range(self.num_links):
            self.mass_params.append(i*10)
            self.inertia_params.extend([i*10+4, i*10+5, i*10+6, i*10+7, i*10+8, i*10+9])

        #self.linkNames = self.generator.getDescriptionOfLinks().split()
        self.linkNames = []  # type: List[str]
        import re
        for d in self.generator.getDescriptionOfParameters().strip().split("\n"):
            link = re.findall(r"of link (.*)", d)[0]
            if link not in self.linkNames:
                self.linkNames.append(link)
        if self.opt['verbose']:
            print('{}'.format({i: self.linkNames[i] for i in range(self.num_links)}))

        self.limits = helpers.URDFHelpers.getJointLimits(self.urdf_file, use_deg=False)

        # get amount of initial inertia params (from urdf) (full params, no friction, no removed columns)
        self.num_model_params = self.num_links*10
        self.num_all_params = self.num_model_params

        # add N offset params (offsets or constant friction) and 2*N velocity dependent friction params
        # (velocity +- for asymmetrical friction)
        if self.opt['identifyFriction']:
            self.num_identified_params = self.num_model_params + self.num_dofs
            self.num_all_params += self.num_dofs

            if not self.opt['identifyGravityParamsOnly']:
                if self.opt['identifySymmetricVelFriction']:
                    self.num_identified_params += self.num_dofs
                    self.num_all_params += self.num_dofs
                else:
                    self.num_identified_params += 2*self.num_dofs
                    self.num_all_params += 2*self.num_dofs
        else:
            self.num_identified_params = self.num_model_params

        self.friction_params_start = self.num_model_params

        if self.opt['identifyGravityParamsOnly']:
            self.num_identified_params = self.num_identified_params - len(self.inertia_params)
            self.friction_params_start = self.num_model_params - len(self.inertia_params)

        if self.opt['verbose']:
            print('# params: {} ({} will be identified)'.format(self.num_model_params, self.num_identified_params))

        self.baseNames = ['base f_x', 'base f_y', 'base f_z', 'base m_x', 'base m_y', 'base m_z']

        self.gravity = [0,0,-9.81,0,0,0]
        self.gravity_twist = iDynTree.Twist.fromList(self.gravity)

        if self.opt['useRBDL']:
            import rbdl
            self.rbdlModel = rbdl.loadModel(self.urdf_file, floating_base=self.opt['floatingBase'], verbose=False)
            self.rbdlModel.gravity = np.array(self.gravity[0:3])
        self.dynComp = iDynTree.DynamicsComputations()
        self.dynComp.loadRobotModelFromFile(self.urdf_file)

        # get model parameters
        xStdModel = iDynTree.VectorDynSize(self.generator.getNrOfParameters())
        self.generator.getModelParameters(xStdModel)
        self.xStdModel = xStdModel.toNumPy()
        if self.opt['identifyFriction']:
            self.xStdModel = np.concatenate((self.xStdModel, np.zeros(self.num_dofs)))
            if not self.opt['identifyGravityParamsOnly']:
                if self.opt['identifySymmetricVelFriction']:
                    self.xStdModel = np.concatenate((self.xStdModel, np.zeros(self.num_dofs)))
                else:
                    self.xStdModel = np.concatenate((self.xStdModel, np.zeros(2*self.num_dofs)))
            helpers.ParamHelpers.addFrictionFromURDF(self, self.urdf_file, self.xStdModel)

        if opt['estimateWith'] == 'urdf':
            self.xStd = self.xStdModel

        if regressor_init:
            # get model dependent projection matrix and linear column dependencies (i.e. base
            # groupings)
            self.computeRegressorLinDepsQR()


    def simulateDynamicsRBDL(self, samples, sample_idx, dynComp=None, xStdModel=None):
        # type: (Dict[str, np._ArrayLike], int, iDynTree.DynamicsComputations, np._ArrayLike[float]) -> np._ArrayLike[float]
        import rbdl

        # read sample data
        q = samples['positions'][sample_idx]
        qdot = samples['velocities'][sample_idx]
        qddot = samples['accelerations'][sample_idx]
        fb = 0

        if xStdModel is None:
            xStdModel = self.xStdModel

        if self.opt['floatingBase']:
            fb = 6
            # The twist (linear/angular velocity) of the base, expressed in the world
            # orientation frame and with respect to the base origin
            # (samples are base frame)
            base_velocity = samples['base_velocity'][sample_idx]
            # The 6d classical acceleration (linear/angular acceleration) of the base
            # expressed in the world orientation frame and with respect to the base
            # origin
            # (samples are base frame)
            base_acc = samples['base_acceleration'][sample_idx]
            rpy = samples['base_rpy'][sample_idx]

            # the first three elements (0,1,2) of q are the position variables of the floating body
            # elements 3,4,5 of q are the x,y,z components of the quaternion of the floating body
            # the w component of the quaternion is appended at the end (?)
            rotq = Quaternion.fromRPY(rpy[0], rpy[1], rpy[2])
            q = np.concatenate((np.array([0,0,0]), rotq[0:3], q, np.array([rotq[3]])))

            #q = np.concatenate((np.array([0,0,0, 0,0,0]), q, np.array([0])))
            #how to get body id of base link (joint)?
            #self.rbdlModel.SetQuaternion(4, rotq, q)

            # the first three elements (0,1,2) of qdot is the linear velocity of the floating body
            # elements 3,4,5 of qdot is the angular velocity of the floating body
            # (world or base frame?)
            qdot = np.concatenate([base_velocity, qdot])

            # the first three elements (0,1,2) of qddot is the linear acceleration of the floating body
            # elements 3,4,5 of qddot is the angular acceleration of the floating body
            # (world or base frame?)
            qddot = np.concatenate([base_acc, qddot])

        # compute inverse dynamics with rbdl
        tau = np.zeros_like(qdot)
        rbdl.InverseDynamics(self.rbdlModel, q, qdot, qddot, tau)

        if self.opt['identifyFriction']:
            # add friction torques
            # constant
            sign = 1 #np.sign(vel)
            p_constant = range(self.friction_params_start, self.friction_params_start+self.num_dofs)
            tau[fb:] += sign*xStdModel[p_constant]

            # vel dependents
            if not self.opt['identifyGravityParamsOnly']:
                # (take only first half of params as they are not direction dependent in urdf anyway)
                p_vel = range(self.friction_params_start+self.num_dofs, self.friction_params_start+self.num_dofs*2)
                tau[fb:] += xStdModel[p_vel]*qdot[fb:]

        return tau


    def simulateDynamicsIDynTree(self, samples, sample_idx, dynComp=None, xStdModel=None):
        # type: (Dict[str, np._ArrayLike], int, iDynTree.DynamicsComputations, np._ArrayLike[float]) -> np._ArrayLike[float]
        """ compute torques for one time step of measurements """

        if not dynComp:
            dynComp = self.dynComp
        if xStdModel is None:
            xStdModel = self.xStdModel
        world_gravity = iDynTree.SpatialAcc.fromList(self.gravity)

        # read sample data
        pos = samples['positions'][sample_idx]
        vel = samples['velocities'][sample_idx]
        acc = samples['accelerations'][sample_idx]

        # system state for iDynTree
        q = iDynTree.VectorDynSize.fromList(pos)
        dq = iDynTree.VectorDynSize.fromList(vel)
        ddq = iDynTree.VectorDynSize.fromList(acc)

        # calc torques and forces with iDynTree dynamicsComputation class
        if self.opt['floatingBase']:
            base_vel = samples['base_velocity'][sample_idx]
            base_acc = samples['base_acceleration'][sample_idx]
            rpy = samples['base_rpy'][sample_idx]

            # get the homogeneous transformation that transforms vectors expressed
            # in the base reference frame to frames expressed in the world
            # reference frame, i.e. pos_world = world_T_base*pos_base
            # for identification purposes, the position does not matter but rotation is taken
            # from IMU estimation. The gravity, base velocity and acceleration all need to be
            # expressed in world frame
            rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
            pos = iDynTree.Position.Zero()
            world_T_base = iDynTree.Transform(rot, pos).inverse()

            '''
            # rotate base vel and acc to world frame
            to_world = world_T_base.getRotation().toNumPy()
            base_vel[0:3] = to_world.dot(base_vel[0:3])
            base_vel[3:] = to_world.dot(base_vel[3:])
            base_acc[0:3] = to_world.dot(base_acc[0:3])
            base_acc[3:] = to_world.dot(base_acc[3:])
            '''

            # The twist (linear, angular velocity) of the base, expressed in the world
            # orientation frame and with respect to the base origin
            base_velocity = iDynTree.Twist.fromList(base_vel)
            # The 6d classical acceleration (linear, angular acceleration) of the base
            # expressed in the world orientation frame and with respect to the base origin
            base_acceleration = iDynTree.ClassicalAcc.fromList(base_acc)

            dynComp.setRobotState(q, dq, ddq, world_T_base, base_velocity, base_acceleration,
                                  world_gravity)
        else:
            dynComp.setRobotState(q, dq, ddq, world_gravity)

        # compute inverse dynamics
        torques = iDynTree.VectorDynSize(self.num_dofs)
        baseReactionForce = iDynTree.Wrench()
        dynComp.inverseDynamics(torques, baseReactionForce)
        torques = torques.toNumPy()

        if self.opt['identifyFriction']:
            # add friction torques
            # constant
            sign = 1 #np.sign(vel)
            p_constant = range(self.friction_params_start, self.friction_params_start+self.num_dofs)
            torques += sign*xStdModel[p_constant]

            # vel dependents
            if not self.opt['identifyGravityParamsOnly']:
                # (take only first half of params as they are not direction dependent in urdf anyway)
                p_vel = range(self.friction_params_start+self.num_dofs, self.friction_params_start+self.num_dofs*2)
                torques += xStdModel[p_vel]*vel

        if self.opt['floatingBase']:
            return np.concatenate((baseReactionForce.toNumPy(), torques))
        else:
            return torques

    def computeRegressors(self, data, only_simulate=False):
        # type: (Model, Data, bool) -> (None)
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
        self.regressor_stack = np.zeros(shape=((self.num_dofs+fb)*data.num_used_samples, self.num_identified_params))
        self.torques_stack = np.zeros(shape=((self.num_dofs+fb)*data.num_used_samples))
        self.sim_torq_stack = np.zeros(shape=((self.num_dofs+fb)*data.num_used_samples))
        self.torquesAP_stack = np.zeros(shape=((self.num_dofs+fb)*data.num_used_samples))

        num_contacts = len(data.samples['contacts'].item(0).keys()) if 'contacts' in data.samples else 0
        self.contacts_stack = np.zeros(shape=(num_contacts, (self.num_dofs+fb)*data.num_used_samples))
        self.contactForcesSum = np.zeros(shape=((self.num_dofs+fb)*data.num_used_samples))

        """loop over measurement data, optionally skip some values
            - get the regressor for each time step
            - if necessary, calculate inverse dynamics to get simulated torques
            - if necessary, get torques from contact wrenches and add them to the torques
            - stack the torques, regressors and contacts into matrices
        """
        contacts = {}
        for sample_index in self.progress(range(data.num_used_samples)):
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

                if self.opt['identifyGravityParamsOnly']:
                    #set vel and acc to zero (should be almost zero already) to remove noise
                    vel[:] = 0.0
                    acc[:] = 0.0

                # system state for iDynTree
                q = iDynTree.VectorDynSize.fromList(pos)
                dq = iDynTree.VectorDynSize.fromList(vel)
                ddq = iDynTree.VectorDynSize.fromList(acc)

                # in case that we simulate the torque measurements, need torque estimation for a priori parameters
                # or that we need to simulate the base reaction forces for floating base
                if self.opt['simulateTorques'] or self.opt['useAPriori'] or self.opt['floatingBase']:
                    if self.opt['useRBDL']:
                        #TODO: make sure joint order of torques is the same as iDynTree!
                        sim_torques = self.simulateDynamicsRBDL(data.samples, m_idx)
                    else:
                        sim_torques = self.simulateDynamicsIDynTree(data.samples, m_idx)

                    if self.opt['useAPriori']:
                        # torques sometimes contain nans, just a very small C number that gets converted to nan?
                        torqAP = np.nan_to_num(sim_torques)

                    if not self.opt['useRegressorForSimulation']:
                        if self.opt['simulateTorques']:
                            torq = np.nan_to_num(sim_torques)
                        else:
                            # write estimated base forces to measured torq vector from file (usually
                            # can't be measured so they are simulated from the measured base motion,
                            # contacts are added further down)
                            if self.opt['floatingBase']:
                                if len(torq) < (self.num_dofs + fb):
                                    torq = np.concatenate((np.nan_to_num(sim_torques[0:6]), torq))
                                else:
                                    torq[0:6] = np.nan_to_num(sim_torques[0:6])

            simulate_time += t.interval

            #...still looping over measurement samples

            row_index = (self.num_dofs+fb)*sample_index   # index for current row in stacked matrices

            if not only_simulate:
                # get numerical regressor (std)
                with helpers.Timer() as t:
                    if self.opt['floatingBase']:
                        base_vel = data.samples['base_velocity'][m_idx]
                        base_acc = data.samples['base_acceleration'][m_idx]

                        # get transform from base to world
                        rpy = data.samples['base_rpy'][m_idx]
                        rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
                        pos = iDynTree.Position.Zero()
                        world_T_base = iDynTree.Transform(rot, pos).inverse()

                        '''
                        # rotate base vel and acc to world frame
                        to_world = world_T_base.getRotation().toNumPy()
                        base_vel[0:3] = to_world.dot(base_vel[0:3])
                        base_vel[3:] = to_world.dot(base_vel[3:])
                        base_acc[0:3] = to_world.dot(base_acc[0:3])
                        base_acc[3:] = to_world.dot(base_acc[3:])
                        '''

                        base_velocity = iDynTree.Twist.fromList(base_vel)
                        base_acceleration = iDynTree.Twist.fromList(base_acc)

                        self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration,
                                                     self.gravity_twist)
                    else:
                        self.generator.setRobotState(q,dq,ddq, self.gravity_twist)

                    # get (standard) regressor
                    regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_model_params)
                    knownTerms = iDynTree.VectorDynSize(self.N_OUT)   # what are known terms useable for?
                    if not self.generator.computeRegressor(regressor, knownTerms):
                        print("Error during numeric computation of regressor")

                    regressor = regressor.toNumPy()
                    if self.opt['floatingBase']:
                        # the base forces are expressed in the base frame for the regressor, so
                        # rotate them to world frame (inverse dynamics use world frame)
                        to_world = world_T_base.getRotation().toNumPy()
                        regressor[0:3, :] = to_world.dot(regressor[0:3, :])
                        regressor[3:6, :] = to_world.dot(regressor[3:6, :])

                    if self.opt['identifyGravityParamsOnly']:
                        #delete inertia param columns
                        regressor = np.delete(regressor, self.inertia_params, 1)

                    if self.opt['identifyFriction']:
                        # append unitary matrix to regressor for offsets/constant friction
                        sign = 1 #np.sign(dq.toNumPy())   #TODO: dependent on direction or always constant?
                        static_diag = np.identity(self.num_dofs)*sign
                        offset_regressor = np.vstack( (np.zeros((fb, self.num_dofs)), static_diag))
                        regressor = np.concatenate((regressor, offset_regressor), axis=1)

                        if not self.opt['identifyGravityParamsOnly']:
                            if self.opt['identifySymmetricVelFriction']:
                                # just use velocity directly
                                vel_diag = np.identity(self.num_dofs)*dq.toNumPy()
                                friction_regressor = np.vstack( (np.zeros((fb, self.num_dofs)), vel_diag))   # add base dynamics rows
                            else:
                                # append positive/negative velocity matrix for velocity dependent asymmetrical friction
                                dq_p = dq.toNumPy().copy()
                                dq_p[dq_p < 0] = 0 #set to zero where v < 0
                                dq_m = dq.toNumPy().copy()
                                dq_m[dq_m > 0] = 0 #set to zero where v > 0
                                vel_diag = np.hstack((np.identity(self.num_dofs)*dq_p, np.identity(self.num_dofs)*dq_m))
                                friction_regressor = np.vstack( (np.zeros((fb, self.num_dofs*2)), vel_diag))   # add base dynamics rows
                            regressor = np.concatenate((regressor, friction_regressor), axis=1)

                    # simulate with regressor
                    if self.opt['useRegressorForSimulation'] and (self.opt['simulateTorques'] or
                            self.opt['useAPriori'] or self.opt['floatingBase']):
                        torques = regressor.dot(self.xStdModel[self.identified_params])
                        if self.opt['simulateTorques']:
                            torq = torques
                        else:
                            # write estimated base forces to measured torq vector from file (usually
                            # can't be measured so they are simulated from the measured base motion,
                            # contacts are added further down)
                            if self.opt['floatingBase']:
                                if len(torq) < (self.num_dofs + fb):
                                    torq = np.concatenate((np.nan_to_num(torques[0:6]), torq))
                                else:
                                    torq[0:6] = np.nan_to_num(torques[0:6])
                        torques_simulated = np.nan_to_num(torques)

                    # stack on previous regressors
                    np.copyto(self.regressor_stack[row_index:row_index+self.num_dofs+fb], regressor)
                num_time += t.interval

            # stack results onto matrices of previous time steps
            np.copyto(self.torques_stack[row_index:row_index+self.num_dofs+fb], torq)

            if self.opt['useAPriori']:
                np.copyto(self.torquesAP_stack[row_index:row_index+self.num_dofs+fb], torqAP)

            if len(contacts.keys()):
                #convert contact wrenches into torque contribution
                for c in range(num_contacts):
                    frame = list(contacts.keys())[c]

                    dim = self.num_dofs+fb
                    # get jacobian and contact wrench for each contact frame and measurement sample
                    jacobian = iDynTree.MatrixDynSize(6, dim)
                    if not self.dynComp.getFrameJacobian(str(frame), jacobian):
                        continue
                    jacobian = jacobian.toNumPy()

                    # mul each sample of measured contact wrenches with frame jacobian
                    contacts_torq = np.empty(dim)
                    contacts_torq = jacobian.T.dot(contacts[frame])

                    contact_idx = (sample_index*dim)
                    np.copyto(self.contacts_stack[c][contact_idx:contact_idx+dim], contacts_torq[-dim:])

        # finished looping over samples

        # sum over (contact torques) for each contact frame
        self.contactForcesSum = np.sum(self.contacts_stack, axis=0)

        if self.opt['floatingBase']:
            if self.opt['simulateTorques']:
                # add measured contact wrench to torque estimation from iDynTree
                if self.opt['addContacts']:
                    self.torques_stack = self.torques_stack + self.contactForcesSum
            else:
                # if not simulating, measurements of joint torques already contain contact contribution,
                # so only add it to the (always simulated) base force estimation
                torques_stack_2dim = np.reshape(self.torques_stack, (data.num_used_samples, self.num_dofs+fb))
                self.contactForcesSum_2dim = np.reshape(self.contactForcesSum, (data.num_used_samples, self.num_dofs+fb))
                if self.opt['addContacts']:
                    torques_stack_2dim[:, :6] += self.contactForcesSum_2dim[:, :6]
                self.torques_stack = torques_stack_2dim.flatten()

        if self.opt['addContacts']:
            self.sim_torq_stack = self.sim_torq_stack + self.contactForcesSum

        if len(contacts.keys()) or self.opt['simulateTorques']:
            # write back torques to data object when simulating or contacts were added
            self.data.samples['torques'] = np.reshape(self.torques_stack, (data.num_used_samples, self.num_dofs+fb))

        with helpers.Timer() as t:
            if self.opt['useAPriori']:
                # get torque delta to identify with
                self.tau = self.torques_stack - self.torquesAP_stack
            else:
                self.tau = self.torques_stack
        simulate_time+=t.interval

        self.YStd = self.regressor_stack

        # if difference between random regressor (that was used for base projection) and regressor
        # from the data is too big, the base regressor can still have linear dependencies.
        # for these cases, it seems to be better to get the base columns directly from the data regressor matrix
        if not self.opt['useStructuralRegressor'] and not only_simulate:
            if self.opt['verbose']:
                print('Getting independent base columns again from data regressor')
            self.computeRegressorLinDepsQR(self.YStd)

        if self.opt['useBasisProjection']:
            self.YBase = np.dot(self.YStd, self.B)   # project regressor to base regressor
        else:
            self.YBase = np.dot(self.YStd, self.Pb)  # regressor following Sousa, 2014

        if self.opt['filterRegressor']:
            order = 5                            # Filter order
            fs = self.data.samples['frequency']  # Sampling freq
            fc = self.opt['filterRegCutoff']     # Cut-off frequency (Hz)
            b, a = signal.butter(order, fc / (fs / 2), btype='low', analog=False)
            for j in range(0, self.num_base_inertial_params):
                for i in range(0, self.num_dofs):
                    self.YBase[i::self.num_dofs, j] = signal.filtfilt(b, a, self.YBase[i::self.num_dofs, j])

        self.sample_end = data.samples['positions'].shape[0]
        if self.opt['skipSamples'] > 0: self.sample_end -= (self.opt['skipSamples'])

        # keep absolute torques (self.tau can be relative)
        self.tauMeasured = np.reshape(self.torques_stack, (data.num_used_samples, self.num_dofs+fb))

        self.T = data.samples['times'][0:self.sample_end:self.opt['skipSamples']+1]

        if self.opt['showTiming']:
            print('(simulation for regressors took %.03f sec.)' % simulate_time)
            print('(getting regressors took %.03f sec.)' % num_time)

        if self.opt['verbose'] == 2:
            print("YStd: {}".format(self.YStd.shape), end=' ')
            print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))


    def getRandomRegressor(self, n_samples=None):
        """
        Utility function for generating a random regressor for numerical base parameter calculation
        Given n_samples, the Y (n_samples*getNrOfOutputs() X getNrOfParameters() ) regressor is
        obtained by stacking the n_samples generated regressors
        This function returns Y^T Y (getNrOfParameters() X getNrOfParameters() ) (that share the row space with Y)
        (partly ported from iDynTree)
        """

        if self.opt['identifyGravityParamsOnly']:
            regr_filename = self.urdf_file + '.gravity_regressor.npz'
        else:
            regr_filename = self.urdf_file + '.regressor.npz'
        generate_new = False
        fb = self.opt['floatingBase']

        try:
            regr_file = np.load(regr_filename)
            R = regr_file['R']  #regressor matrix
            Q = regr_file['Q']  #QR decomposition
            RQ = regr_file['RQ']
            PQ = regr_file['PQ']
            n = regr_file['n']   #number of samples that were used
            fbase = regr_file['fb']  #floating base flag
            grav = regr_file['grav_only']
            fric = regr_file['fric']
            fric_sym = regr_file['fric_sym']
            if self.opt['verbose']:
                print("loaded random structural regressor from {}".format(regr_filename))
            if n != n_samples or fbase != fb or R.shape[0] != self.num_identified_params or \
                    self.opt['identifyGravityParamsOnly'] != grav or \
                    fric != self.opt['identifyFriction'] or fric_sym != self.opt['identifySymmetricVelFriction']:
                generate_new = True
            #TODO: save and check timestamp of urdf file, if newer regenerate
        except (IOError, KeyError):
            generate_new = True

        if generate_new:
            if not n_samples:
                n_samples = self.num_dofs * 1000

            if self.opt['verbose']:
                print("(re-)generating structural regressor ({} random positions)".format(n_samples))

            R = np.array((self.N_OUT, self.num_model_params))
            regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_model_params)
            knownTerms = iDynTree.VectorDynSize(self.N_OUT)
            if len(self.limits) > 0:
                jn = self.jointNames
                q_lim_pos = [self.limits[jn[n]]['upper'] for n in range(self.num_dofs)]
                q_lim_neg = [self.limits[jn[n]]['lower'] for n in range(self.num_dofs)]
                dq_lim = [self.limits[jn[n]]['velocity'] for n in range(self.num_dofs)]
                q_range = (np.array(q_lim_pos) - np.array(q_lim_neg)).tolist()
            for i in self.progress(range(0, n_samples)):
                # set random system state
                if len(self.limits) > 0:
                    rnd = np.random.rand(self.num_dofs) #0..1
                    q = iDynTree.VectorDynSize.fromList((q_lim_neg+q_range*rnd))
                    if self.opt['identifyGravityParamsOnly']:
                        #set vel and acc to zero for static case
                        vel = np.zeros(self.num_dofs)
                        acc = np.zeros(self.num_dofs)
                    else:
                        vel = ((np.random.rand(self.num_dofs)-0.5)*2*dq_lim)
                        acc = ((np.random.rand(self.num_dofs)-0.5)*2*np.pi)
                    dq = iDynTree.VectorDynSize.fromList(vel.tolist())
                    ddq = iDynTree.VectorDynSize.fromList(acc.tolist())
                else:
                    q = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.num_dofs)*2-1)*np.pi).tolist())
                    dq = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.num_dofs)*2-1)*np.pi).tolist())
                    ddq = iDynTree.VectorDynSize.fromList(((np.random.ranf(self.num_dofs)*2-1)*np.pi).tolist())

                # TODO: make work with fixed dofs (set vel and acc to zero, look at iDynTree method)

                if self.opt['floatingBase']:
                    base_vel = np.pi*np.random.rand(6)
                    base_acc = np.pi*np.random.rand(6)
                    if self.opt['identifyGravityParamsOnly']:
                        #set vel and acc to zero for static case (reduces resulting amount of base dependencies)
                        base_vel[:] = 0.0
                        base_acc[:] = 0.0
                    rpy = np.random.ranf(3)*0.1
                    rot = iDynTree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
                    pos = iDynTree.Position.Zero()
                    world_T_base = iDynTree.Transform(rot, pos).inverse()

                    '''
                    # rotate base vel and acc to world frame
                    to_world = world_T_base.getRotation().toNumPy()
                    base_vel[0:3] = to_world.dot(base_vel[0:3])
                    base_vel[3:] = to_world.dot(base_vel[3:])
                    base_acc[0:3] = to_world.dot(base_acc[0:3])
                    base_acc[3:] = to_world.dot(base_acc[3:])
                    '''

                    base_velocity = iDynTree.Twist.fromList(base_vel)
                    base_acceleration = iDynTree.Twist.fromList(base_acc)

                    self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, self.gravity_twist)
                else:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)

                # get regressor
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print("Error during numeric computation of regressor")

                A = regressor.toNumPy()

                #the base forces are expressed in the base frame for the regressor, so rotate them
                if self.opt['floatingBase']:
                    to_world = np.fromstring(world_T_base.getRotation().toString(), sep=' ').reshape((3,3))
                    A[0:3, :] = to_world.dot(A[0:3, :])
                    A[3:6, :] = to_world.dot(A[3:6, :])

                if self.opt['identifyGravityParamsOnly']:
                    #delete inertia param columns
                    A = np.delete(A, self.inertia_params, 1)

                if self.opt['identifyFriction']:
                    # append unitary matrix to regressor for offsets/constant friction
                    sign = 1 #np.sign(dq.toNumPy())
                    static_diag = np.identity(self.num_dofs)*sign
                    offset_regressor = np.vstack( (np.zeros((fb*6, self.num_dofs)), static_diag))
                    A = np.concatenate((A, offset_regressor), axis=1)

                    if not self.opt['identifyGravityParamsOnly']:
                        if self.opt['identifySymmetricVelFriction']:
                            # just use velocity directly
                            vel_diag = np.identity(self.num_dofs)*dq.toNumPy()
                            friction_regressor = np.vstack( (np.zeros((fb*6, self.num_dofs)), vel_diag))   # add base dynamics rows
                        else:
                            # append positive/negative velocity matrix for velocity dependent asymmetrical friction
                            dq_p = dq.toNumPy().copy()
                            dq_p[dq_p < 0] = 0 #set to zero where v < 0
                            dq_m = dq.toNumPy().copy()
                            dq_m[dq_m > 0] = 0 #set to zero where v > 0
                            vel_diag = np.hstack((np.identity(self.num_dofs)*dq_p, np.identity(self.num_dofs)*dq_m))
                            friction_regressor = np.vstack( (np.zeros((fb*6, self.num_dofs*2)), vel_diag))   # add base dynamics rows
                        A = np.concatenate((A, friction_regressor), axis=1)

                # add to previous regressors, linear dependencies don't change
                # (if too many, saturation or accuracy problems?)
                if i==0:
                    R = A.T.dot(A)
                else:
                    R += A.T.dot(A)

            # get column space dependencies
            Q,RQ,PQ = sla.qr(R, pivoting=True, mode='economic')

            np.savez(regr_filename, R=R, Q=Q, RQ=RQ, PQ=PQ, n=n_samples, fb=self.opt['floatingBase'], grav_only=self.opt['identifyGravityParamsOnly'], fric=self.opt['identifyFriction'], fric_sym=self.opt['identifySymmetricVelFriction'])

        if 'showRandomRegressor' in self.opt and self.opt['showRandomRegressor']:
            import matplotlib.pyplot as plt
            plt.imshow(R, interpolation='nearest')
            plt.show()

        return R, Q,RQ,PQ


    def computeRegressorLinDepsQR(self, regressor=None):
        """get base regressor and identifiable basis matrix with QR decomposition

        gets independent columns (non-unique choice) each with its dependent ones, i.e.
        those std parameter indices that form each of the base parameters (including the linear factors)
        """
        if regressor is not None:
            # if supplied, get dependencies from specific regressor
            Y = regressor
            self.Q, self.R, self.P = sla.qr(Y, pivoting=True, mode='economic')
        else:
            #using random regressor gives us structural base params, not dependent on excitation
            #QR of transposed gives us basis of column space of original matrix (Gautier, 1990)
            Y, self.Q, self.R, self.P = self.getRandomRegressor(n_samples=self.opt['randomSamples'])

        """
        # get basis directly from regressor matrix using QR
        Qt,Rt,Pt = sla.qr(Y.T, pivoting=True, mode='economic')

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
        #get rank
        r = np.where(np.abs(self.R.diagonal()) > self.opt['minTol'])[0].size
        self.num_base_params = r
        self.num_base_inertial_params = r - self.num_dofs

        #create proper permutation matrix from vector
        self.Pp = np.zeros((self.P.size, self.P.size))
        for i in self.P:
            self.Pp[i, self.P[i]] = 1
        self.Pb = self.Pp.T[:, 0:self.num_base_params]
        self.Pd = self.Pp.T[:, self.num_base_params:]

        # get the choice of indices of "independent" columns of the regressor matrix
        # (representants chosen from each separate interdependent group of columns)
        self.independent_cols = self.P[0:r]

        # get column dependency matrix (with what factor are columns of "dependent" columns grouped)
        # i (independent column) = (value at i,j) * j (dependent column index among the others)
        R1 = self.R[0:r, 0:r]
        R2 = self.R[0:r, r:]
        self.linear_deps = sla.inv(R1).dot(R2)
        self.linear_deps[np.abs(self.linear_deps) < self.opt['minTol']] = 0

        self.Kd = self.linear_deps
        self.K = self.Pb.T + self.Kd.dot(self.Pd.T)

        # collect grouped columns for each independent column
        # and build base matrix
        # (slow too, save to file)
        if self.opt['useBasisProjection']:
            self.B = np.zeros((self.num_identified_params, self.num_base_params))
            for j in range(0, self.linear_deps.shape[0]):
                indep_idx = self.independent_cols[j]
                for i in range(0, self.linear_deps.shape[1]):
                    for k in range(r, self.P.size):
                        factor = self.linear_deps[j, k-r]
                        if np.abs(factor)>self.opt['minTol']: self.B[self.P[k],j] = factor
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
        self.param_syms = list()     # type: List[sympy.Symbol]
        self.mass_syms = list()      # type: List[sympy.Symbol]
        self.friction_syms = list()  # type: List[sympy.Symbol]
        #indices of params within full param vector that are going to be identified
        self.identified_params = list()  # type: List[int]
        for i in range(0, self.num_links):
            #mass
            m = symbols('m_{}'.format(i))
            self.param_syms.append(m)
            self.identified_params.append(i*10)
            self.mass_syms.append(m)

            #first moment of mass
            p = 'c_{}'.format(i)  #symbol prefix
            syms = [symbols(p+'x'), symbols(p+'y'), symbols(p+'z')]
            self.param_syms.extend(syms)
            self.identified_params.extend([i*10+1, i*10+2, i*10+3])

            #3x3 inertia tensor about link-frame (for link i)
            p = 'I_{}'.format(i)
            syms = [symbols(p+'xx'), symbols(p+'xy'), symbols(p+'xz'),
                    symbols(p+'xy'), symbols(p+'yy'), symbols(p+'yz'),
                    symbols(p+'xz'), symbols(p+'yz'), symbols(p+'zz')
                   ]
            self.param_syms.extend([syms[0], syms[1], syms[2], syms[4], syms[5], syms[8]])

            if not self.opt['identifyGravityParamsOnly']:
                self.identified_params.extend([i*10+4, i*10+5, i*10+6, i*10+7, i*10+8, i*10+9])

        if self.opt['identifyFriction']:
            mp = self.num_model_params
            for i in range(0,self.num_dofs):
                s = [symbols('Fc_{}'.format(i))]
                self.param_syms.extend(s)
                self.friction_syms.extend(s)
                self.identified_params.append(mp+i)
            if not self.opt['identifyGravityParamsOnly']:
                if self.opt['identifySymmetricVelFriction']:
                    for i in range(0,self.num_dofs):
                        s = [symbols('Fv_{}'.format(i))]
                        self.param_syms.extend(s)
                        self.friction_syms.extend(s)
                        self.identified_params.append(mp+self.num_dofs+i)
                else:
                    for i in range(0,self.num_dofs):
                        s = [symbols('Fv+_{}'.format(i))]
                        self.param_syms.extend(s)
                        self.friction_syms.extend(s)
                        self.identified_params.append(mp+self.num_dofs+i)
                    for i in range(0,self.num_dofs):
                        s = [symbols('Fv-_{}'.format(i))]
                        self.param_syms.extend(s)
                        self.friction_syms.extend(s)
                        self.identified_params.append(mp+2*self.num_dofs+i)
        self.param_syms = np.array(self.param_syms)

        ## get symbolic equations for base param dependencies
        # Each dependent parameter can be ignored (non-identifiable) or it can be
        # represented by grouping some base and/or dependent parameters.
        # TODO: put this in regressor cache file (it's slow)
        if self.opt['useBasisProjection']:
            if self.opt['orthogonalizeBasis']:
                #this is only correct if basis is orthogonal
                self.base_deps = np.dot(self.param_syms[self.identified_params], self.B)
            else:
                #otherwise, we need to get relationships from the inverse
                B_qr_inv_z = la.pinv(self.B)
                B_qr_inv_z[np.abs(B_qr_inv_z) < self.opt['minTol']] = 0
                self.base_deps = np.dot(self.param_syms[self.identified_params], B_qr_inv_z.T)
        else:
            # using projection matrix from Gautier/Sousa method for base eqns
            # (K is orthogonal)
            self.base_deps = Matrix(self.K) * Matrix(self.param_syms[self.identified_params])

        # find std parameters that have no effect on estimation (not single or contributing to base
        # equations)
        # TODO: also put this in regressor cache file
        base_deps_syms = []   # type: List[sympy.Symbol]
        for i in range(self.base_deps.shape[0]):
            for s in self.base_deps[i].free_symbols:
                if s not in base_deps_syms:
                    base_deps_syms.append(s)
        self.non_id = [p for p in range(self.num_all_params) if self.param_syms[p] not in base_deps_syms]
        self.identifiable = [p for p in range(self.num_all_params) if p not in self.non_id]


    def getSubregressorsConditionNumbers(self):
        # get condition number for each of the links
        linkConds = list()
        for i in range(0, self.num_links):
            #get columns of base regressor that are dependent on std parameters of link i
            # TODO: try going further down to e.g. condition number of link mass, com, inertial
            # and ignore those groups of params

            ## get parts of base regressor with only independent columns (identifiable space)

            #get all independent std columns for link i
            #base_columns = [j for j in range(0, self.num_base_params) \
            #                      if self.independent_cols[j] in range(i*10, i*10+9+1)]

            # use base column dependencies to get combined params of base regressor with
            # coontribution on each each link (a bit inexact I guess)
            base_columns = list()  # type: List[int]
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

        if self.opt['verbose']:
            print("Condition numbers of link sub-regressors: [{}]".format(dict(enumerate(linkConds))))

        return linkConds
