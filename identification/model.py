import sys
import numpy as np
import numpy.linalg as la
from scipy import signal
import scipy.linalg as sla
from sympy import symbols
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from IPython import embed

import helpers

class Model(object):
    def __init__(self, opt, urdf_file, regressor_file=None):
        self.urdf_file = urdf_file
        self.opt = opt

        # create generator instance and load model
        self.generator = iDynTree.DynamicsRegressorGenerator()
        ret = self.generator.loadRobotAndSensorsModelFromFile(urdf_file)
        if not ret:
            sys.exit()

        # load also with new model class for some functions
        self.idyn_model = iDynTree.Model()
        iDynTree.modelFromURDF(urdf_file, self.idyn_model)
        if self.opt['verbose']:
            print 'loaded model {}'.format(urdf_file)

        # define what regressor type
        if regressor_file:
            with open(regressor_file, 'r') as file:
               regrXml = file.read()
        else:
            # (default for all joints)
            if self.opt['floating_base']:
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

        # TODO: this and the following are not dependent on joints specified in regressor (but
        # uses all from model file)!
        self.N_DOFS = self.generator.getNrOfDegreesOfFreedom()
        if self.opt['verbose']:
            print '# DOFs: {}'.format(self.N_DOFS)

        # Get the number of outputs of the regressor
        # (should be #links - #fakeLinks)
        self.N_OUT = self.generator.getNrOfOutputs()
        if self.opt['verbose']:
            print '# outputs: {}'.format(self.N_OUT)

        # get initial inertia params (from urdf)
        self.num_params = self.generator.getNrOfParameters()
        if self.opt['verbose']:
            print '# params: {}'.format(self.num_params)

        self.N_LINKS = self.generator.getNrOfLinks()-self.generator.getNrOfFakeLinks()
        if self.opt['verbose']:
            print '# links: {} ({} fake)'.format(self.N_LINKS+self.generator.getNrOfFakeLinks(),
                                             self.generator.getNrOfFakeLinks())

        self.link_names = []
        for i in range(0, self.N_LINKS):
            self.link_names.append(self.idyn_model.getLinkName(i))
        if self.opt['verbose']:
            print '({})'.format(self.link_names)

        self.baseNames = ['base f_x', 'base f_y', 'base f_z', 'base m_x', 'base m_y', 'base m_z']
        self.jointNames = [self.generator.getDescriptionOfDegreeOfFreedom(dof) for dof in range(0, self.N_DOFS)]

        self.gravity_twist = iDynTree.Twist.fromList([0,0,-9.81,0,0,0])

        if opt['iDynSimulate'] or opt['useAPriori'] or opt['floating_base']:
            self.dynComp = iDynTree.DynamicsComputations();
            self.dynComp.loadRobotModelFromFile(self.urdf_file);

            self.world_gravity = iDynTree.SpatialAcc.fromList([0, 0, -9.81, 0, 0, 0])

        # get model parameters
        xStdModel = iDynTree.VectorDynSize(self.num_params)
        self.generator.getModelParameters(xStdModel)
        self.xStdModel = xStdModel.toNumPy()
        if opt['estimateWith'] == 'urdf':
            self.xStd = self.xStdModel

        # get model dependent projection matrix and linear column dependencies (i.e. base
        # groupings)
        # (put here so it's only done once for the loaded model)
        self.computeRegressorLinDepsQR()

    def computeRegressors(self, data):
        """ compute regressors for each time step of the measurement data and stack them vertically,
            also compute torques for these measurements with the a priori parameters (from URDF)
        """

        self.data = data

        num_time = 0
        simulate_time = 0

        #extra regressor rows for floating base
        if self.opt['floating_base']: fb = 6
        else: fb = 0
        self.regressor_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples, self.num_params))
        self.torques_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))
        self.torquesAP_stack = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))
        if self.opt['floating_base']:
            self.contacts_stack = np.zeros(shape=(len(data.samples['contacts'].item().keys()), 6*data.num_used_samples))
            self.contactForcesSum = np.zeros(shape=((self.N_DOFS+fb)*data.num_used_samples))

        """loop over measurements (optionally skip some values) and get one regressor per time step"""
        for sample_index in range(0, data.num_used_samples):
            m_idx = sample_index*(self.opt['skip_samples'])+sample_index
            with helpers.Timer() as t:
                # read samples
                pos = data.samples['positions'][m_idx]
                vel = data.samples['velocities'][m_idx]
                acc = data.samples['accelerations'][m_idx]
                torq = data.samples['torques'][m_idx]
                if self.opt['floating_base']:
                    contacts = {}
                    for frame in data.samples['contacts'].item().keys():
                        #TODO: define proper sign for input data
                        contacts[frame] = -data.samples['contacts'].item()[frame][m_idx]
                    # The twist (linear/angular velocity) of the base, expressed in the world
                    # orientation frame and with respect to the base origin
                    base_vel = data.samples['base_velocity'][m_idx]
                    base_velocity = iDynTree.Twist.fromList(base_vel)
                    # The 6d classical acceleration (linear/angular acceleration) of the base
                    # expressed in the world orientation frame and with respect to the base
                    # origin (not spatial acc)
                    base_acc = data.samples['base_acceleration'][m_idx]
                    base_acceleration = iDynTree.ClassicalAcc.fromList(base_acc)

                # system state for iDynTree
                q = iDynTree.VectorDynSize.fromList(pos)
                dq = iDynTree.VectorDynSize.fromList(vel)
                ddq = iDynTree.VectorDynSize.fromList(acc)

                # in case that we simulate the torque measurements, need torque estimation for a priori parameters
                # or that we need to simulate the base reaction forces for floating base
                if self.opt['iDynSimulate'] or self.opt['useAPriori'] or self.opt['floating_base']:
                    # calc torques and forces with iDynTree dynamicsComputation class
                    if self.opt['floating_base']:
                        # the homogeneous transformation that transforms position vectors expressed
                        # in the base reference frame in position frames expressed in the world
                        # reference frame, i.e. pos_world = world_T_base*pos_base
                        self.dynComp.setFloatingBase(self.opt['base_link_name'])
                        world_T_base = self.dynComp.getWorldBaseTransform()
                        self.dynComp.setRobotState(q, dq, ddq, world_T_base,
                                                   base_velocity, base_acceleration,
                                                   self.world_gravity)
                    else:
                        self.dynComp.setRobotState(q, dq, ddq, self.world_gravity)

                    # compute inverse dynamics with idyntree (simulate)
                    torques = iDynTree.VectorDynSize(self.N_DOFS)  #being calculated by inverseDynamics
                    baseReactionForce = iDynTree.Wrench()   #being calculated by inverseDynamics
                    self.dynComp.inverseDynamics(torques, baseReactionForce)

                    if self.opt['useAPriori']:
                        if self.opt['floating_base']:
                            torqAP = np.concatenate((baseReactionForce.toNumPy(), torques.toNumPy()))
                        else:
                            torqAP = torques.toNumPy()
                        # torques sometimes contain nans, just a very small number in C that gets converted to nan?
                        np.nan_to_num(torqAP)

                    if self.opt['iDynSimulate']:
                        if self.opt['floating_base']:
                            torq = np.concatenate((baseReactionForce.toNumPy(), torques.toNumPy()))
                        else:
                            torq = torques.toNumPy()
                        torq = np.nan_to_num(torq)
                        data.samples['torques'][m_idx] = torques.toNumPy()
                    else:
                        if self.opt['floating_base']:
                            #add estimated base forces to measured torq vector
                            torq = np.concatenate((baseReactionForce.toNumPy(), torq))

                if self.opt['addNoise'] != 0:
                    torq += np.random.randn(self.N_DOFS+fb)*self.opt['addNoise']

            simulate_time += t.interval

            #...still looping over measurement samples

            # get numerical regressor (std)
            row_index = (self.N_DOFS+fb)*sample_index   # index for current row in stacked regressor matrix
            with helpers.Timer() as t:
                if self.opt['floating_base']:
                    vel = data.samples['base_velocity'][m_idx]
                    acc = data.samples['base_acceleration'][m_idx]
                    base_velocity = iDynTree.Twist.fromList(vel)
                    base_acceleration = iDynTree.Twist.fromList(acc)
                    world_T_base = self.dynComp.getWorldTransform(self.opt['base_link_name'])
                    self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, self.gravity_twist)
                else:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)
                    #self.generator.setTorqueSensorMeasurement(iDynTree.VectorDynSize.fromList(torq))   #why this?

                # get (standard) regressor
                regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_params)
                knownTerms = iDynTree.VectorDynSize(self.N_OUT)   # what are known terms useable for?
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print "Error during numeric computation of regressor"

                # stack on previous regressors
                np.copyto(self.regressor_stack[row_index:row_index+self.N_DOFS+fb], regressor.toNumPy())
            num_time += t.interval

            # stack results onto matrices of previous timesteps
            np.copyto(self.torques_stack[row_index:row_index+self.N_DOFS+fb], torq)
            if self.opt['useAPriori']:
                np.copyto(self.torquesAP_stack[row_index:row_index+self.N_DOFS+fb], torqAP)

            contact_idx = (sample_index*6)
            if self.opt['floating_base']:
                for i in range(self.contacts_stack.shape[0]):
                    frame = contacts.keys()[i]
                    np.copyto(self.contacts_stack[i][contact_idx:contact_idx+6], contacts[frame])

        if self.opt['floating_base']:
            #TODO: if robot does not have contact sensors, use HyQ null-space trick

            #convert contact forces into torque contribution
            for i in range(self.contacts_stack.shape[0]):
                frame = contacts.keys()[i]

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

            #subtract measured contact forces from torque estimation from iDynTree
            if self.opt['iDynSimulate']:
                self.torques_stack -= self.contactForcesSum
            if self.opt['useAPriori']:
                self.torquesAP_stack -= self.contactForcesSum

        with helpers.Timer() as t:
            if self.opt['useAPriori']:
                # get torque delta to identify with
                self.tau = self.torques_stack - self.torquesAP_stack
            else:
                self.tau = self.torques_stack
        simulate_time+=t.interval

        self.YStd = self.regressor_stack
        self.YBase = np.dot(self.YStd, self.B)   # project regressor to base regressor

        if self.opt['verbose']:
            print("YStd: {}".format(self.YStd.shape)),
        print("YBase: {}, cond: {}".format(self.YBase.shape, la.cond(self.YBase)))

        if self.opt['filterRegressor']:
            order = 6                       #Filter order
            try:
                fs = self.data.samples['frequency']  #Sampling freq
            except:
                fs = 200.0
            fc = 5                          #Cut-off frequency (Hz)
            b, a = signal.butter(order, fc / (fs/2), btype='low', analog=False)
            for j in range(0, self.num_base_params):
                for i in range(0, self.N_DOFS):
                    self.YBase[i::self.N_DOFS, j] = signal.filtfilt(b, a, self.YBase[i::self.N_DOFS, j])

        self.sample_end = data.samples['positions'].shape[0]
        if self.opt['skip_samples'] > 0: self.sample_end -= (self.opt['skip_samples'])

        '''
        if self.opt['iDynSimulate']:
            if self.opt['useAPriori']:
                tau = self.torques_stack    # use original measurements, not delta
            else:
                tau = self.tau
            self.tauMeasured = np.reshape(tau, (data.num_used_samples, self.N_DOFS+fb))
        else:
            self.tauMeasured = data.samples['torques'][0:self.sample_end:self.opt['skip_samples']+1, :]
        '''
        tau = self.torques_stack    # use original measurements, not delta
        self.tauMeasured = np.reshape(tau, (data.num_used_samples, self.N_DOFS+fb))

        self.T = data.samples['times'][0:self.sample_end:self.opt['skip_samples']+1]

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
        try:
            regr_file = np.load(regr_filename)
            R = regr_file['R']
            n = regr_file['n']
            if self.opt['verbose']:
                print("loaded random regressor from {}".format(regr_filename))
            if n != n_samples:
                generate_new = True
            #TODO: save and check timestamp of urdf file, if newer regenerate
        except IOError, KeyError:
            generate_new = True

        if generate_new:
            if self.opt['verbose']:
                print("generating random regressor")
            import random

            if not n_samples:
                n_samples = self.N_DOFS * 5000
            R = np.array((self.N_OUT, self.num_params))
            regressor = iDynTree.MatrixDynSize(self.N_OUT, self.num_params)
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

                if self.opt['floating_base']:
                    base_velocity = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
                    base_acceleration = iDynTree.Twist.fromList(np.pi*np.random.rand(6))
                    world_T_base = self.dynComp.getWorldTransform(self.opt['base_link_name'])
                    self.generator.setRobotState(q,dq,ddq, world_T_base, base_velocity, base_acceleration, self.gravity_twist)
                else:
                    self.generator.setRobotState(q,dq,ddq, self.gravity_twist)

                # get regressor
                if not self.generator.computeRegressor(regressor, knownTerms):
                    print "Error during numeric computation of regressor"

                A = regressor.toNumPy()

                # add to previous regressors, linear dependency doesn't change
                # (if too many, saturation or accuracy problems?)
                if i==0:
                    R = A.T.dot(A)
                else:
                    R += A.T.dot(A)

            np.savez(regr_filename, R=R, n=n_samples)

        if self.opt.has_key('showRandomRegressor') and self.opt['showRandomRegressor']:
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
        Qt,Rt,Pt = sla.qr(Yrand.T, pivoting=True, mode='economic')

        #get rank
        r = np.where(np.abs(Rt.diagonal()) > self.opt['min_tol'])[0].size
        self.num_base_params = r

        #get basis projection matrix
        self.B = Qt[:, 0:r]
        """

        # get column space dependencies
        Q,R,P = sla.qr(Yrand, pivoting=True, mode='economic')
        self.Q, self.R, self.P = Q,R,P

        #get rank
        r = np.where(np.abs(R.diagonal()) > self.opt['min_tol'])[0].size
        self.num_base_params = r

        #create proper permutation matrix from vector
        self.Pp = np.zeros((P.size, P.size))
        for i in P:
            self.Pp[i, P[i]] = 1

        # get the choice of indices of "independent" columns of the regressor matrix
        # (representants chosen from each separate interdependent group of columns)
        self.independent_cols = P[0:r]

        # get column dependency matrix (with what factor are columns of "dependent" columns grouped)
        # i (independent column) = (value at i,j) * j (dependent column index among the others)
        R1 = R[0:r, 0:r]
        R2 = R[0:r, r:]
        self.linear_deps = sla.inv(R1).dot(R2)

        # collect grouped columns for each independent column
        # build base matrix
        self.B = np.zeros((self.num_params, self.num_base_params))
        for j in range(0, self.linear_deps.shape[0]):
            indep_idx = self.independent_cols[j]
            for i in range(0, self.linear_deps.shape[1]):
                for k in range(r, P.size):
                    fact = round(self.linear_deps[j, k-r], 5)
                    if np.abs(fact)>self.opt['min_tol']: self.B[P[k],j] = fact
            self.B[indep_idx,j] = 1

        if self.opt.has_key('orthogonalizeBasis') and self.opt['orthogonalizeBasis']:
            #orthogonalize, so linear relationships can be inverted
            Q_B_qr, R_B_qr = la.qr(self.B)
            Q_B_qr[np.abs(Q_B_qr) < self.opt['min_tol']] = 0
            self.B = Q_B_qr
            self.Binv = self.B.T
        else:
            self.Binv = la.pinv(self.B)   #if basis B is not orthogonal

        # define sympy symbols for each std column
        self.param_syms = list()
        self.mass_syms = list()
        for i in range(0,self.N_LINKS):
            #mass
            m = symbols('m_{}'.format(i))
            self.param_syms.append(m)
            self.mass_syms.append(m)

            #first moment of mass
            p = 'l_{}'.format(i)  #symbol prefix
            syms = [symbols(p+'x'), symbols(p+'y'), symbols(p+'z')]
            self.param_syms.extend(syms)
            #3x3 inertia tensor about link-frame (for link i)
            p = 'L_{}'.format(i)
            syms = [symbols(p+'xx'), symbols(p+'xy'), symbols(p+'xz'),
                    symbols(p+'xy'), symbols(p+'yy'), symbols(p+'yz'),
                    symbols(p+'xz'), symbols(p+'yz'), symbols(p+'zz')
                   ]
            self.param_syms.extend([syms[0], syms[1], syms[2], syms[4], syms[5], syms[8]])

        #create symbolic equations for base param dependencies
        self.base_deps = np.dot(self.param_syms, self.B)

        '''
        #use reduced row echelon form to get basis for identifiable subspace
        #(rrf does not get minimal reduced space though)
        from sympy import Matrix
        Ew = Matrix(Yrand).rref()
        Ew_np = np.array(Ew[0].tolist(), dtype=float)
        # B in Paper:
        self.R = Ew_np[~np.all(Ew_np==0, axis=1)]    #remove rows that are all zero
        self.Rpinv = la.pinv(self.R)   # == P in Paper
        '''


    def computeRegressorLinDepsSVD(self):
        """get base regressor and identifiable basis matrix with iDynTree (SVD)"""

        with helpers.Timer() as t:
            # get subspace basis (for projection to base regressor/parameters)
            Yrand = self.getRandomRegressor(5000)
            #A = iDynTree.MatrixDynSize(self.num_params, self.num_params)
            #self.generator.generate_random_regressors(A, False, True, 2000)
            #Yrand = A.toNumPy()
            U, s, Vh = la.svd(Yrand, full_matrices=False)
            r = np.sum(s>self.opt['min_tol'])
            self.B = -Vh.T[:, 0:r]
            self.num_base_params = r

            print("tau: {}".format(self.tau.shape)),

            print("YStd: {}".format(self.YStd.shape)),
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
            base_columns = [j for j in range(0, self.num_base_params) \
                                  if self.independent_cols[j] in range(i*10, i*10+9)]

            #get base params that have corresponding std params for the current link
            #TODO: check if this is a valid approach, also use proper new dependents
            for j in range(0, self.num_base_params):
                for dep in np.where(np.abs(self.linear_deps[j, :])>0.1)[0]:
                    if dep in range(i*10, i*10+9):
                        base_columns.append(j)
            if not len(base_columns):
                linkConds.append(99999)
            else:
                linkConds.append(la.cond(self.YBase[:, base_columns]))
            #linkConds.append(la.cond(self.YStd[:, i*10:i*10+9]))
        print("Condition numbers of link sub-regressors: [{}]".format(linkConds))

        return linkConds
