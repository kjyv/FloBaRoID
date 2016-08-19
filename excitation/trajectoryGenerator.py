import numpy as np
from identification.helpers import URDFHelpers

import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')


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


    def initWithRandomParams(self):
        # init with random params
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

        self.oscillators = list()
        for i in range(0, self.dofs):
            self.oscillators.append(OscillationGenerator(w_f = self.w_f_global, a = np.array(a[i]),
                                                         b = np.array(b[i]), q0 = q[i], nf = nf[i], use_deg = self.use_deg
                                                        ))

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

        if wf:
            self.w_f_global = wf

        #for i in nf:
        #    if not ( len(a) == i and len(b) == i):
        #        raise Exception("Need nf many values in each parameter array value!")

        self.oscillators = list()
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

class TrajectoryOptimizer(object):
    def __init__(self, config, simulation_func):
        self.config = config
        self.sim_func = simulation_func
        # init some classes
        self.limits = URDFHelpers.getJointLimits(config['model'], use_deg = config['useDeg'])
        self.trajectory = TrajectoryGenerator(config['N_DOFS'], use_deg = config['useDeg'])

        self.dofs = self.config['N_DOFS']
        # number of fourier partial sums for each joint (same for all joints atm)
        # (needs to be larger with more dofs? means a lot more variables)
        self.nf = [2]*config['N_DOFS']

    def initGraph(self):
        # init graphing of optimization
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)
        plt.ion()
        self.xar = []
        self.yar = []
        #self.ax1.plot(self.xar,self.yar)

        self.updateGraphEveryVals = 5

        # 'globals' for objfunc
        self.iter_cnt = 0   #iteration counter
        self.last_g = None
        self.constr = []

    def updateGraph(self):
        if self.iter_cnt % self.updateGraphEveryVals == 0:
            self.ax1.plot(self.xar, self.yar, color='g')
            if self.iter_cnt == 1: plt.show(block=False)
            plt.pause(0.01)

    def vecToParams(self, x):
        wf = x[0]
        q = x[1:self.dofs+1]
        ab_len = self.dofs*self.nf[0]
        a = np.array(np.split(x[self.dofs+1:self.dofs+1+ab_len], self.dofs))
        b = np.array(np.split(x[self.dofs+1+ab_len:self.dofs+1+ab_len*2], self.dofs))
        return wf, q, a, b

    def objfunc(self, x):
        self.iter_cnt += 1
        wf, q, a, b = self.vecToParams(x)

        print 'wf {}'.format(wf)
        print 'a {}'.format(np.round(a,5).tolist())
        print 'b {}'.format(np.round(b,5).tolist())
        print 'q {}'.format(np.round(q,5).tolist())

        #wf out of bounds, skip call
        if not self.testBounds(x):
            fail = 1
            return 10e10, [1.0]*self.dofs*5, fail

        self.trajectory.initWithParams(a,b,q, self.nf, wf)

        if 'model' in locals():
            trajectory_data, model = self.sim_func(self.config, self.trajectory, model)
        else:
            trajectory_data, model = self.sim_func(self.config, self.trajectory)
        last_trajectory_data = trajectory_data

        f = np.linalg.cond(model.YBase)
        #f = np.log(np.linalg.det(model.YBase.T.dot(model.YBase)))   #fisher information matrix

        #xBaseModel = np.dot(model.Binv, model.xStdModel)
        #f = np.linalg.cond(model.YBase.dot(np.diag(xBaseModel)))    #weighted with CAD params

        print "\niter #{}: objective function value: {}".format(self.iter_cnt, f)
        if self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.updateGraph()

        # add constraints  (later tested for all: g(n) <= 0)
        g = [0.0]*self.dofs*5
        # check for joint limits
        jn = self.config['jointNames']
        for n in range(self.dofs):
            # joint pos lower
            g[n] = self.limits[jn[n]]['lower'] - np.min(trajectory_data['positions'][:, n])
            # joint pos upper
            g[self.dofs+n] = np.min(trajectory_data['positions'][:, n]) - self.limits[jn[n]]['upper']
            # max joint vel
            g[2*self.dofs+n] = np.max(np.abs(trajectory_data['velocities'][:, n])) - self.limits[jn[n]]['velocity']
            # max torques
            g[3*self.dofs+n] = np.max(np.abs(trajectory_data['torques'][:, n])) - self.limits[jn[n]]['torque']
            # max joint vel of trajectory should at least be 30% of joint limit
            g[4*self.dofs+n] = self.limits[jn[n]]['velocity']*0.1 - np.max(np.abs(trajectory_data['velocities'][:, n]))
        self.last_g = g
        #constr = [{'type':'ineq', 'fun': last_g[i]} for i in range(config['N_DOFS'])]

        self.testConstraints(g)

        #TODO: add minimum max torque
        #TODO: limit accelerations (probably not necessary since torques are limited)
        #TODO: allow some manual constraints for angles (from config)
        #TODO: add cartesian/collision constraints, e.g. using fcl

        fail = 0
        return f, g, fail

    def testBounds(self, x):
        #test variable bounds
        wf, q, a, b = self.vecToParams(x)
        wf_t = wf >= self.wf_min and wf <= self.wf_max
        q_t = np.all(q <= self.qmax) and np.all(q >= self.qmin)
        a_t = np.all(a <= self.amax) and np.all(a >= self.amin)
        b_t = np.all(b <= self.bmax) and np.all(b >= self.bmin)
        res = wf_t and q_t

        if not res:
            print "bounds violated"

        return res

    def testConstraints(self, g):
        res = np.all(np.array(g) <= 0)
        if not res:
            print "constraints violated:"
            if True in np.in1d(range(1,2*self.dofs), np.where(np.array(g) > 0)):
                print "angle limits"
            if True in np.in1d(range(2*self.dofs,3*self.dofs), np.where(np.array(g) > 0)):
                print "max velocity limits"
            if True in np.in1d(range(3*self.dofs,4*self.dofs), np.where(np.array(g) > 0)):
                print "max torque limits"
            if True in np.in1d(range(4*self.dofs,5*self.dofs), np.where(np.array(g) > 0)):
                print "min velocity limits"
        return res

    def testParams(**kwargs):
        x = kwargs['x_new']
        return testBounds(x) and testConstraints(last_g)

    def optimizeTrajectory(self):
        # use non-linear optimization to find parameters for minimal
        # condition number trajectory

        if self.config['showOptimizationGraph']:
            self.initGraph()

        ## pyOpt

        from pyOpt import Optimization
        from pyOpt import SLSQP, CONMIN, ALHSO, ALPSO

        # Instanciate Optimization Problem
        opt_prob = Optimization('Trajectory optimization', self.objfunc)
        opt_prob.addObj('f')

        # add variables, define bounds
        # w_f - pulsation
        self.wf_min = 0.5
        self.wf_max = 1.5
        opt_prob.addVar('wf', 'c', value=1, lower=self.wf_min, upper=self.wf_max)

        # q - offsets
        self.qmin = -45.0
        self.qmax = 45.0
        if not self.config['useDeg']:
            self.qmin = np.deg2rad(self.qmin)
            self.qmax = np.deg2rad(self.qmax)
        for i in range(self.dofs):
            opt_prob.addVar('q_%d'%i,'c', value=0.0, lower=self.qmin, upper=self.qmax)

        # a, b - sin/cos params
        self.amin = self.bmin = -1.0
        self.amax = self.bmax = 1.0
        for i in range(self.dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('a{}_{}'.format(i,j), 'c', value=-0.1/(j+1), lower=self.amin, upper=self.amax)
        for i in range(self.dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('b{}_{}'.format(i,j), 'c', value=0.1/(j+1), lower=self.bmin, upper=self.bmax)

        # add constraint vars (constraint functions are in obfunc)
        opt_prob.addConGroup('g', self.dofs*5, 'i')
        #print opt_prob

        ### optimize using pyOpt (global)

        initial = [v.value for v in opt_prob._variables.values()]

        if self.config['useGlobalOptimization']:
            opt = ALPSO()  #particle swarm
            opt.setOption('stopCriteria', 0)
            opt.setOption('maxInnerIter', 5)  #limit amount of iters to small number
            opt.setOption('maxOuterIter', 10)  #limit amount of iters to small number
            opt.setOption('printInnerIters', 0)
            opt.setOption('printOuterIters', 1)
            opt.setOption('SwarmSize', 20)

    #        opt = ALHSO()   #harmony search
    #        opt.setOption('maxoutiter', 10)
    #        opt.setOption('maxinniter', 5)
    #        opt.setOption('stopcriteria', 1)
    #        opt.setOption('stopiters', 3)
    #        opt.setOption('atol', 0.01)
    #        opt.setOption('prtinniter', 0)
    #        opt.setOption('prtoutiter', 1)

            # run fist (global) optimization
            try:
                #reuse history
                opt(opt_prob, store_hst=True, hot_start=True, xstart=initial)
            except NameError:
                opt(opt_prob, store_hst=True, xstart=initial)
            print opt_prob.solution(0)

            ### using scipy (global-local optimization)
    #        def printMinima(x, f, accept):
    #            print("found local minimum with cond {}. accepted: ".format(f, accept))
    #        bounds = [(v.lower, v.upper) for v in opt_prob._variables.values()]
    #        np.random.seed(1)
    #        global_sol = sp.optimize.basinhopping(objfunc, initial, disp=True,
    #                                              accept_test=testParams,
    #                                              callback=printMinima,
    #                                              niter=50, #niter_success=10,
    #                                              T=1000,
    #                                              minimizer_kwargs={
    #                                                  'bounds': bounds,
    #                                                  'constraints': constr,
    #                                                  #'method':'SLSQP',  #only for smooth functions
    #                                                  #'options':{'maxiter': 10, 'disp':True}  #doesn't stop after maxiter
    #
    #                                                  'method':'COBYLA',
    #                                                  'options':{'maxiter': 20, 'disp':True}
    #                                              }
    #                                             )
    #        print("basin-hopping solution found:")
    #        print global_sol.message
    #        print global_sol.x

        ### pyOpt local

        # after using global optimization, get more exact solution with
        # gradient based method init optimizer (only local)
        use_parallel = 0
        if use_parallel:
            #SLSQP possibly not thread safe? (there is https://github.com/jacobwilliams/slsqp because of that)
            opt2 = SLSQP(pll_type='POA')   #sequential least squares
        else:
            opt2 = SLSQP()   #sequential least squares
        #opt2.setOption('ACC', 1e-1)
        opt2.setOption('MAXIT', 1)
        opt2.setOption('IPRINT', 0)

        if self.config['useGlobalOptimization']:
            #reuse previous solution
            problem = opt_prob.solution(0)
        else:
            problem = opt_prob

        try:
            #reuse history
            opt2(problem, store_hst=True, hot_start=True, sens_step=1e-2)
        except NameError:
            opt2(problem, store_hst=True, sens_step=1e-2)
        local_sol = problem.solution(0)
        print local_sol
        sol_wf = local_sol.getVar(0).value

        sol_q = list()
        for i in range(self.dofs):
            sol_q.append(local_sol.getVar(1+i).value)

        sol_a = list()
        sol_b = list()
        for i in range(self.dofs):
            a_series = list()
            for j in range(self.nf[0]):
                a_series.append(local_sol.getVar(1+self.dofs+i+j).value)
            sol_a.append(a_series)
        for i in range(self.dofs):
            b_series = list()
            for j in range(self.nf[0]):
                b_series.append(local_sol.getVar(1+self.dofs+self.nf[0]*self.dofs+i+j).value)
            sol_b.append(b_series)

        print("testing final solution")
        self.iter_cnt = 0
        local_sol_vec = np.array([local_sol.getVar(x).value for x in range(0,len(local_sol._variables))])
        self.objfunc(local_sol_vec)
        print("\n")

        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf)

        if self.config['showOptimizationGraph']:
            plt.ioff()

        return self.trajectory
