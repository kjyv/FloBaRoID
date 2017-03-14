from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Dict, Tuple, Union

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pyOpt
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from identification.model import Model
from identification.data import Data
from identification.helpers import URDFHelpers
from excitation.trajectoryGenerator import simulateTrajectory, Trajectory, PulsedTrajectory
from excitation.optimizer import plotter, Optimizer


class TrajectoryOptimizer(Optimizer):
    def __init__(self, config, model, simulation_func):
        super(TrajectoryOptimizer, self).__init__(config, model, simulation_func)

        # init some classes
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad
        self.trajectory = PulsedTrajectory(config['num_dofs'], use_deg = config['useDeg'])

        self.dofs = self.config['num_dofs']

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
            self.qmin = []  # type: List[float]
            self.qmax = []  # type: List[float]
            self.qinit = []  # type: List[float]
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

        self.last_best_f_f1 = 0


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
        jac = np.zeros((x0.size, f0.size))
        dx = np.zeros(x0.size)
        for i in range(x0.size):
            dx[i] = epsilon
            jac[i] = (f(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()


    def objectiveFunc(self, x):
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
            f = 1000.0
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
        #old_floatingBase = self.config['floatingBase']
        #self.config['floatingBase'] = 0
        trajectory_data, data = self.sim_func(self.config, self.trajectory, model=self.model)

        self.config['verbose'] = old_verbose
        #self.config['floatingBase'] = old_floatingBase

        self.last_trajectory_data = trajectory_data
        if self.config['showOptimizationTrajs']:
            plotter(self.config, data=trajectory_data)

        f = np.linalg.cond(self.model.YBase)
        #f = np.log(np.linalg.det(model.YBase.T.dot(model.YBase)))   #fisher information matrix

        #xBaseModel = np.dot(model.Binv | K, model.xStdModel)
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
        if self.mpi_rank == 0 and self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        # TODO: add cartesian/collision constraints using fcl
        # for whole trajectory, get closest distance of all link pairs

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


    def addVarsAndConstraints(self, opt_prob):
        # type: (pyOpt.Optimization) -> None
        ''' add variables, define bounds '''

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


    def optimizeTrajectory(self):
        # type: () -> PulsedTrajectory
        # use non-linear optimization to find parameters for minimal
        # condition number trajectory

        # Instanciate Optimization Problem
        opt_prob = pyOpt.Optimization('Trajectory optimization', self.objectiveFunc)
        opt_prob.addObj('f')

        self.addVarsAndConstraints(opt_prob)
        sol_vec = self.runOptimizer(opt_prob)

        sol_wf, sol_q, sol_a, sol_b = self.vecToParams(sol_vec)
        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf)

        if self.config['showOptimizationGraph']:
            plt.ioff()

        return self.trajectory
