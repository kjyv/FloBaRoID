from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) >= LooseVersion('1.5'):
    plt.style.use('seaborn-pastel')

from identification.helpers import URDFHelpers
from excitation.trajectoryGenerator import TrajectoryGenerator

class TrajectoryOptimizer(object):
    def __init__(self, config, simulation_func):
        self.config = config
        self.sim_func = simulation_func
        # init some classes
        self.limits = URDFHelpers.getJointLimits(config['model'], use_deg=False)  #will always be compared to rad
        self.trajectory = TrajectoryGenerator(config['N_DOFS'], use_deg = config['useDeg'])

        self.dofs = self.config['N_DOFS']

        ## bounds for parameters
        # number of fourier partial sums (same for all joints atm)
        # (needs to be larger for larger dofs? means a lot more variables)
        self.nf = [4]*self.dofs
        #pulsation
        self.wf_min = self.config['trajectoryPulseMin']
        self.wf_max = self.config['trajectoryPulseMax']
        self.wf_init = self.config['trajectoryPulseInit']
        #angle offsets
        if self.config['trajectoryAngleRanges'][0] is not None:
            self.qmin = []
            self.qmax = []
            self.qinit = []
            for i in range(0, self.dofs):
                low = self.config['trajectoryAngleRanges'][i][0]
                high = self.config['trajectoryAngleRanges'][i][1]
                self.qmin.append(low)
                self.qmax.append(high)
                self.qinit.append((high+low)*0.5)  #set init to middle of range
        else:
            self.qmin = [self.config['trajectoryAngleMin']]*self.dofs
            self.qmax = [self.config['trajectoryAngleMax']]*self.dofs
            self.qinit = [self.config['trajectoryAngleMax']-self.config['trajectoryAngleMin']]*self.dofs

        if not self.config['useDeg']:
            self.qmin = np.deg2rad(self.qmin)
            self.qmax = np.deg2rad(self.qmax)
            self.qinit = np.deg2rad(self.qinit)
        #sin/cos coefficients
        self.amin = self.bmin = self.config['trajectoryCoeffMin']
        self.amax = self.bmax = self.config['trajectoryCoeffMax']
        self.ainit = np.empty((self.dofs, self.nf[0]))
        self.binit = np.empty((self.dofs, self.nf[0]))
        for i in range(0, self.dofs):
            for j in range(0, self.nf[0]):
                #fade out and alternate sign
                #self.ainit[j] = self.config['trajectoryCoeffInit']/ (j+1) * ((j%2)*2-1)
                #self.binit[j] = self.config['trajectoryCoeffInit']/ (j+1) * ((1-j%2)*2-1)
                #self.ainit[i,j] = self.binit[i,j] = self.config['trajectoryCoeffInit']+
                #                  ((self.amax-self.config['trajectoryCoeffInit'])/(self.dofs-i))
                self.ainit[i,j] = self.binit[i,j] = self.config['trajectoryCoeffInit']

        self.last_best_f = float('inf')
        self.last_best_f_f1 = 0
        self.last_best_sol = None

    def initGraph(self):
        # init graphing of objective function value
        self.fig = plt.figure(0)
        self.ax1 = self.fig.add_subplot(1,1,1)
        plt.ion()
        self.xar = []
        self.yar = []
        self.x_constr = []
        self.ax1.plot(self.xar,self.yar)

        self.updateGraphEveryVals = 5

        # 'globals' for objfunc
        self.iter_cnt = 0   #iteration counter
        self.last_g = None

    def updateGraph(self):
        # draw all optimization steps so far (yes, no updating)

        if self.iter_cnt % self.updateGraphEveryVals == 0:
            # go through list of constraint states and draw next line with other color if changed
            i = last_i = 0
            last_state = self.x_constr[0]
            while (i < len(self.x_constr)):
                if self.x_constr[i]: color = 'g'
                else: color = 'r'
                if self.x_constr[i] == last_state:
                    # draw intermediate and at end of data
                    if i-last_i +1 >= self.updateGraphEveryVals:
                        #need to draw one point more to properly connect to next segment
                        if last_i == 0: end = i+1
                        else: end = i+2
                        self.ax1.plot(self.xar[last_i:end], self.yar[last_i:end], marker='p', markerfacecolor=color, color='0.75')
                        last_i = i
                else:
                    #draw line when state has changed
                    last_state = not last_state
                    if last_i == 0: end = i+1
                    else: end = i+2
                    self.ax1.plot(self.xar[last_i:end], self.yar[last_i:end], marker='p', markerfacecolor=color, color='0.75')
                    last_i = i
                i+=1

        if self.iter_cnt == 1: plt.show(block=False)
        plt.pause(0.01)


    def vecToParams(self, x):
        # convert vector of all solution variables to separate parameter variables
        wf = x[0]
        q = x[1:self.dofs+1]
        ab_len = self.dofs*self.nf[0]
        a = np.array(np.split(x[self.dofs+1:self.dofs+1+ab_len], self.dofs))
        b = np.array(np.split(x[self.dofs+1+ab_len:self.dofs+1+ab_len*2], self.dofs))
        return wf, q, a, b


    def approx_jacobian(self, f, x, epsilon, *args):
        """Approximate the Jacobian matrix of callable function func

           * Parameters
             x       - The state vector at which the Jacobian matrix is desired
             func    - A vector-valued function of the form f(x,*args)
             epsilon - The peturbation used to determine the partial derivatives
             *args   - Additional arguments passed to func

           * Returns
             An array of dimensions (lenf, lenx) where lenf is the length
             of the outputs of func, and lenx is the number of

           * Notes
             The approximation is done using forward differences
        """

        x0 = np.asfarray(x)
        f0 = f(*((x0,)+args))
        jac = np.zeros([x0.size,f0.size])
        dx = np.zeros(x0.size)
        for i in range(x0.size):
            dx[i] = epsilon
            jac[i] = (f(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()

    def objective_func(self, x):
        self.iter_cnt += 1
        print("iter #{}/{}".format(self.iter_cnt, self.iter_max))

        wf, q, a, b = self.vecToParams(x)

        if self.config['verbose']:
            print('wf {}'.format(wf))
            print('a {}'.format(np.round(a,5).tolist()))
            print('b {}'.format(np.round(b,5).tolist()))
            print('q {}'.format(np.round(q,5).tolist()))

        #input vars out of bounds, skip call
        if not self.testBounds(x):
            # give penalty obj value for out of bounds (because we shouldn't get here)
            # TODO: for some algorithms (with augemented lagrangian added bounds) this should
            # not be very high as it is added again anyway)
            f = 10000.0
            if self.config['minVelocityConstraint']:
                g = [10.0]*self.dofs*5
            else:
                g = [10.0]*self.dofs*4

            fail = 1.0
            self.iter_cnt-=1
            return f, g, fail

        self.trajectory.initWithParams(a,b,q, self.nf, wf)

        old_verbose = self.config['verbose']
        self.config['verbose'] = 0
        old_floatingBase = self.config['floatingBase']
        self.config['floatingBase'] = 0
        if 'model' in locals():
            trajectory_data, data, model = self.sim_func(self.config, self.trajectory, model)
        else:
            # get model at first run, then reuse
            trajectory_data, data, model = self.sim_func(self.config, self.trajectory)

        self.config['verbose'] = old_verbose
        self.config['floatingBase'] = old_floatingBase

        self.last_trajectory_data = trajectory_data
        if self.plot_func: self.plot_func(trajectory_data)

        f = np.linalg.cond(model.YBase)
        #f = np.log(np.linalg.det(model.YBase.T.dot(model.YBase)))   #fisher information matrix

        #xBaseModel = np.dot(model.Binv, model.xStdModel)
        #f = np.linalg.cond(model.YBase.dot(np.diag(xBaseModel)))    #weighted with CAD params

        print("objective function value: {} (last best: {} + {})".format(f,
                                                            self.last_best_f-self.last_best_f_f1,
                                                            self.last_best_f_f1))

        f1 = 0
        # add constraints  (later tested for all: g(n) <= 0)
        if self.config['minVelocityConstraint']:
            g = [0.0]*self.dofs*5
        else:
            g = [0.0]*self.dofs*4

        # check for joint limits
        jn = self.config['jointNames']
        for n in range(self.dofs):
            # joint pos lower
            if len(self.config['ovrPosLimit'])>=n and self.config['ovrPosLimit'][n]:
                g[n] = np.deg2rad(self.config['ovrPosLimit'][n][0]) - np.min(trajectory_data['positions'][:, n])
            else:
                g[n] = self.limits[jn[n]]['lower'] - np.min(trajectory_data['positions'][:, n])
            # joint pos upper
            if len(self.config['ovrPosLimit'])>=n and self.config['ovrPosLimit'][n]:
                g[self.dofs+n] = np.max(trajectory_data['positions'][:, n]) - np.deg2rad(self.config['ovrPosLimit'][n][1])
            else:
                g[self.dofs+n] = np.max(trajectory_data['positions'][:, n]) - self.limits[jn[n]]['upper']
            # max joint vel
            g[2*self.dofs+n] = np.max(np.abs(trajectory_data['velocities'][:, n])) - self.limits[jn[n]]['velocity']
            # max torques
            g[3*self.dofs+n] = np.nanmax(np.abs(data.samples['torques'][:, n])) - self.limits[jn[n]]['torque']

            if self.config['minVelocityConstraint']:
                # max joint vel of trajectory should at least be 10% of joint limit
                g[4*self.dofs+n] = self.limits[jn[n]]['velocity']*self.config['minVelocityPercentage'] - \
                                    np.max(np.abs(trajectory_data['velocities'][:, n]))

            # highest joint torque should at least be 10% of joint limit
            #g[5*self.dofs+n] = self.limits[jn[n]]['torque']*0.1 - np.max(np.abs(data.samples['torques'][:, n]))
            f_tmp = self.limits[jn[n]]['torque']*0.1 - np.max(np.abs(data.samples['torques'][:, n]))
            if f_tmp > 0:
                f1+=f_tmp
        self.last_g = g

        #add min join torques as second objective
        if f1 > 0:
            f+= f1
            print("added cost: {}".format(f1))

        c = self.testConstraints(g)
        if self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        #TODO: add cartesian/collision constraints, e.g. using fcl

        #keep last best solution (some solvers don't keep it)
        if c and f < self.last_best_f:
            self.last_best_f = f
            self.last_best_f_f1 = f1
            self.last_best_sol = x

        print("\n\n")

        fail = 0.0
        #funcs = {}
        #funcs['opt'] = f
        #funcs['con'] = g
        #return funcs, fail
        return f, g, fail

    def testBounds(self, x):
        #test variable bounds
        wf, q, a, b = self.vecToParams(x)
        wf_t = wf >= self.wf_min and wf <= self.wf_max
        q_t = np.all(q <= self.qmax) and np.all(q >= self.qmin)
        a_t = np.all(a <= self.amax) and np.all(a >= self.amin)
        b_t = np.all(b <= self.bmax) and np.all(b >= self.bmin)
        res = wf_t and q_t and a_t and b_t

        if not res:
            print("bounds violated")

        return res

    def testConstraints(self, g):
        res = np.all(np.array(g) <= self.config['minTolConstr'])
        if not res:
            print("constraints violated:")
            if True in np.in1d(list(range(1,2*self.dofs)), np.where(np.array(g) >= self.config['minTolConstr'])):
                print("angle limits")
                print(np.array(g)[list(range(1,2*self.dofs))])
            if True in np.in1d(list(range(2*self.dofs,3*self.dofs)), np.where(np.array(g) >= self.config['minTolConstr'])):
                print("max velocity limits")
                #print np.array(g)[range(2*self.dofs,3*self.dofs)]
            if True in np.in1d(list(range(3*self.dofs,4*self.dofs)), np.where(np.array(g) >= self.config['minTolConstr'])):
                print("max torque limits")

            if self.config['minVelocityConstraint']:
                if True in np.in1d(list(range(4*self.dofs,5*self.dofs)), np.where(np.array(g) >= self.config['minTolConstr'])):
                    print("min velocity limits")
            #if True in np.in1d(range(5*self.dofs,6*self.dofs), np.where(np.array(g) >= self.config['minTolConstr'])):
            #    print "min torque limits"
            #    print np.array(g)[range(5*self.dofs,6*self.dofs)]
        return res

    def testParams(self, **kwargs):
        x = kwargs['x_new']
        return self.testBounds(x) and self.testConstraints(self.last_g)

    def optimizeTrajectory(self, plot_func=None):
        # use non-linear optimization to find parameters for minimal
        # condition number trajectory

        self.plot_func = plot_func

        if self.config['showOptimizationGraph']:
            self.initGraph()

        ## describe optimization problem with pyOpt classes

        import pyOpt

        # Instanciate Optimization Problem
        opt_prob = pyOpt.Optimization('Trajectory optimization', self.objective_func)
        opt_prob.addObj('f')

        # add variables, define bounds
        # w_f - pulsation
        opt_prob.addVar('wf', 'c', value=self.wf_init, lower=self.wf_min, upper=self.wf_max)

        # q - offsets
        for i in range(self.dofs):
            opt_prob.addVar('q_%d'%i,'c', value=self.qinit[i], lower=self.qmin[i], upper=self.qmax[i])
        # a, b - sin/cos params
        for i in range(self.dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('a{}_{}'.format(i,j), 'c', value=self.ainit[i][j], lower=self.amin, upper=self.amax)
        for i in range(self.dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('b{}_{}'.format(i,j), 'c', value=self.binit[i][j], lower=self.bmin, upper=self.bmax)

        # add constraint vars (constraint functions are in obfunc)
        if self.config['minVelocityConstraint']:
            opt_prob.addConGroup('g', self.dofs*5, type='i')
        else:
            opt_prob.addConGroup('g', self.dofs*4, type='i')
        #print opt_prob

        initial = [v.value for v in list(opt_prob.getVarSet().values())]

        if self.config['useGlobalOptimization']:
            ### optimize using pyOpt (global)
            opt = pyOpt.ALPSO()  #augmented lagrange particle swarm optimization
            opt.setOption('stopCriteria', 0)
            opt.setOption('maxInnerIter', 3)
            opt.setOption('maxOuterIter', self.config['globalOptIterations'])
            opt.setOption('printInnerIters', 1)
            opt.setOption('printOuterIters', 1)
            opt.setOption('SwarmSize', 30)
            opt.setOption('xinit', 1)
            #TODO: how to properly limit max number of function calls?
            # no. func calls = (SwarmSize * inner) * outer + SwarmSize
            self.iter_max = opt.getOption('SwarmSize') * opt.getOption('maxInnerIter') * opt.getOption('maxOuterIter') + opt.getOption('SwarmSize')

            # run fist (global) optimization
            try:
                #reuse history
                opt(opt_prob, store_hst=False, hot_start=True, xstart=initial)
            except NameError:
                opt(opt_prob, store_hst=False, xstart=initial)
            print(opt_prob.solution(0))

        ### pyOpt local

        #TODO: run local optimization for e.g. the three last best results (global solutions could be more or less optimal
        # within their local minima)

        # after using global optimization, get more exact solution with
        # gradient based method init optimizer (only local)
        opt2 = pyOpt.SLSQP()   #sequential least squares
        opt2.setOption('MAXIT', self.config['localOptIterations'])
        if self.config['verbose']:
            opt2.setOption('IPRINT', 0)
        #opt2 = pyOpt.IPOPT()
        #opt2 = pyOpt.PSQP()
        # TODO: amount of function calls depends on amount of variables and iterations to approximate gradient
        # iterations are probably steps along the gradient. How to get proper no. of expected func calls?
        self.iter_max = "(unknown)"

        if self.config['useGlobalOptimization']:
            if self.last_best_sol is not None:
                #use best constrained solution (might be better than what solver thinks)
                for i in range(len(opt_prob.getVarSet())):
                    opt_prob.getVar(i).value = self.last_best_sol[i]

            opt2(opt_prob, store_hst=False, sens_step=0.1)
        else:
            try:
                #reuse history
                opt2(opt_prob, store_hst=True, hot_start=True, sens_step=0.1)
            except NameError:
                opt2(opt_prob, store_hst=True, sens_step=0.1)

        local_sol = opt_prob.solution(0)
        if not self.config['useGlobalOptimization']:
            print(local_sol)
        local_sol_vec = np.array([local_sol.getVar(x).value for x in range(0,len(local_sol.getVarSet()))])

        if self.last_best_sol is not None:
            local_sol_vec = self.last_best_sol
            print("using last best constrained solution instead of given solver solution.")

        sol_wf, sol_q, sol_a, sol_b = self.vecToParams(local_sol_vec)

        print("testing final solution")
        self.iter_cnt = 0
        self.objective_func(local_sol_vec)
        print("\n")

        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf)

        if self.config['showOptimizationGraph']:
            plt.ioff()

        return self.trajectory
