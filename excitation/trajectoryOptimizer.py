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
from colorama import Fore

from identification.model import Model
from identification.data import Data
from identification.helpers import URDFHelpers
from excitation.trajectoryGenerator import simulateTrajectory, Trajectory, PulsedTrajectory
from excitation.optimizer import plotter, Optimizer


class TrajectoryOptimizer(Optimizer):
    def __init__(self, config, idf, model, simulation_func, world=None):
        super(TrajectoryOptimizer, self).__init__(config, idf, model, simulation_func, world=world)

        # init some classes
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad
        self.trajectory = PulsedTrajectory(self.num_dofs, use_deg = config['useDeg'])

        ## bounds for parameters
        # number of fourier partial sums (same for all joints atm)
        # (needs to be larger for larger dofs? means a lot more variables)
        self.nf = [4]*self.num_dofs

        #pulsation
        self.wf_min = self.config['trajectoryPulseMin']
        self.wf_max = self.config['trajectoryPulseMax']
        self.wf_init = self.config['trajectoryPulseInit']

        #angle offsets
        if self.config['trajectoryAngleRanges'] and self.config['trajectoryAngleRanges'][0] is not None:
            self.qmin = []  # type: List[float]
            self.qmax = []  # type: List[float]
            self.qinit = []  # type: List[float]
            for i in range(0, self.num_dofs):
                low = self.config['trajectoryAngleRanges'][i][0]
                high = self.config['trajectoryAngleRanges'][i][1]
                self.qmin.append(low)
                self.qmax.append(high)
                self.qinit.append((high+low)*0.5)  #set init to middle of range
        else:
            self.qmin = [self.config['trajectoryAngleMin']]*self.num_dofs
            self.qmax = [self.config['trajectoryAngleMax']]*self.num_dofs
            self.qinit = [0.5*self.config['trajectoryAngleMin'] + 0.5*self.config['trajectoryAngleMax']]*self.num_dofs

        if not self.config['useDeg']:
            self.qmin = np.deg2rad(self.qmin)
            self.qmax = np.deg2rad(self.qmax)
            self.qinit = np.deg2rad(self.qinit)
        #sin/cos coefficients
        self.amin = self.bmin = self.config['trajectoryCoeffMin']
        self.amax = self.bmax = self.config['trajectoryCoeffMax']
        self.ainit = np.empty((self.num_dofs, self.nf[0]))
        self.binit = np.empty((self.num_dofs, self.nf[0]))
        for i in range(0, self.num_dofs):
            for j in range(0, self.nf[0]):
                #fade out and alternate sign
                #self.ainit[j] = self.config['trajectoryCoeffInit']/ (j+1) * ((j%2)*2-1)
                #self.binit[j] = self.config['trajectoryCoeffInit']/ (j+1) * ((1-j%2)*2-1)
                #self.ainit[i,j] = self.binit[i,j] = self.config['trajectoryCoeffInit']+
                #                  ((self.amax-self.config['trajectoryCoeffInit'])/(self.num_dofs-i))
                self.ainit[i,j] = self.binit[i,j] = self.config['trajectoryCoeffInit']

        self.last_best_f_f1 = 0

        self.num_constraints = self.num_dofs*4  # angle, velocity, torque limits
        if self.config['minVelocityConstraint']:
            self.num_constraints += self.num_dofs

        # collision constraints

        self.idyn_model = iDynTree.Model()
        iDynTree.modelFromURDF(self.config['urdf'], self.idyn_model)
        self.neighbors = URDFHelpers.getNeighbors(self.idyn_model)

        # amount of collision checks to be done
        eff_links = self.model.num_links - len(self.config['ignoreLinksForCollision']) + len(self.world_links)
        self.num_samples = int(self.config['excitationFrequency'] * self.trajectory.getPeriodLength())
        self.num_coll_constraints = (eff_links * (eff_links-1) // 2)

        # ignore neighbors
        nb_pairs = []  # type: List[Tuple]
        for link in self.neighbors:
            if link in self.config['ignoreLinksForCollision']:
                continue
            if link not in self.model.linkNames:
                continue
            nb_real = set(self.neighbors[link]['links']).difference(
                self.config['ignoreLinksForCollision']).intersection(self.model.linkNames)
            for l in nb_real:
                if (link, l) not in nb_pairs and (l, link) not in nb_pairs:
                    nb_pairs.append((link, l))
        self.num_coll_constraints -=  (len(nb_pairs) +        # neighbors
                                  len(self.config['ignoreLinkPairsForCollision']))  # custom combinations
        self.num_constraints += self.num_coll_constraints

        self.initVisualizer()

    def vecToParams(self, x):
        # convert vector of all solution variables to separate parameter variables
        wf = x[0]
        q = x[1:self.num_dofs+1]
        ab_len = self.num_dofs*self.nf[0]
        a = np.array(np.split(np.array(x[self.num_dofs+1:self.num_dofs+1+ab_len]), self.num_dofs))
        b = np.array(np.split(np.array(x[self.num_dofs+1+ab_len:self.num_dofs+1+ab_len*2]), self.num_dofs))
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


    def objectiveFunc(self, x, test=False):
        self.iter_cnt += 1
        print("call #{}/{}".format(self.iter_cnt, self.iter_max))

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
                g = [10.0]*self.num_constraints
            else:
                g = [10.0]*self.num_constraints

            fail = 1.0
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

        f1 = 0
        # add constraints  (later tested for all: g(n) <= 0)
        g = [1e10]*self.num_constraints
        jn = self.model.jointNames
        for n in range(self.num_dofs):
            # check for joint limits
            # joint pos lower
            if len(self.config['ovrPosLimit'])>=n and self.config['ovrPosLimit'][n]:
                g[n] = np.deg2rad(self.config['ovrPosLimit'][n][0]) - np.min(trajectory_data['positions'][:, n])
            else:
                g[n] = self.limits[jn[n]]['lower'] - np.min(trajectory_data['positions'][:, n])
            # joint pos upper
            if len(self.config['ovrPosLimit'])>=n and self.config['ovrPosLimit'][n]:
                g[self.num_dofs+n] = np.max(trajectory_data['positions'][:, n]) - np.deg2rad(self.config['ovrPosLimit'][n][1])
            else:
                g[self.num_dofs+n] = np.max(trajectory_data['positions'][:, n]) - self.limits[jn[n]]['upper']
            # max joint vel
            g[2*self.num_dofs+n] = np.max(np.abs(trajectory_data['velocities'][:, n])) - self.limits[jn[n]]['velocity']
            # max torques
            g[3*self.num_dofs+n] = np.nanmax(np.abs(data.samples['torques'][:, n])) - self.limits[jn[n]]['torque']

            if self.config['minVelocityConstraint']:
                # max joint vel of trajectory should at least be 10% of joint limit
                g[4*self.num_dofs+n] = self.limits[jn[n]]['velocity']*self.config['minVelocityPercentage'] - \
                                    np.max(np.abs(trajectory_data['velocities'][:, n]))

            # highest joint torque should at least be 10% of joint limit
            #g[5*self.num_dofs+n] = self.limits[jn[n]]['torque']*0.1 - np.max(np.abs(data.samples['torques'][:, n]))
            f_tmp = self.limits[jn[n]]['torque']*0.1 - np.max(np.abs(data.samples['torques'][:, n]))
            if f_tmp > 0:
                f1+=f_tmp

        # check collision constraints
        # (for whole trajectory but only get closest distance as constraint value)
        c_s = self.num_constraints - self.num_coll_constraints  # start where collision constraints start
        if self.config['verbose'] > 1:
            print('checking collisions')
        for p in range(0, trajectory_data['positions'].shape[0], 10):
            g_cnt = 0
            if self.config['verbose'] > 1:
                print("Sample {}".format(p))
            q = trajectory_data['positions'][p]

            for l0 in range(self.model.num_links + len(self.world_links)):
                for l1 in range(self.model.num_links + len(self.world_links)):
                    l0_name = (self.model.linkNames + self.world_links)[l0]
                    l1_name = (self.model.linkNames + self.world_links)[l1]

                    if (l0 >= l1):  # don't need, distance is the same in both directions; same link never collides
                        continue
                    if l0_name in self.config['ignoreLinksForCollision'] \
                            or l1_name in self.config['ignoreLinksForCollision']:
                        continue
                    if [l0_name, l1_name] in self.config['ignoreLinkPairsForCollision'] or \
                       [l1_name, l0_name] in self.config['ignoreLinkPairsForCollision']:
                        continue

                    # neighbors can't collide with a proper joint range, so ignore
                    if l0 < self.model.num_links and l1 < self.model.num_links:
                        if l0_name in self.neighbors[l1_name]['links'] or l1_name in self.neighbors[l0_name]['links']:
                            continue

                    if l0 < l1:
                        d = self.getLinkDistance(l0_name, l1_name, q)
                        if d < g[c_s+g_cnt]:
                            g[c_s+g_cnt] = d
                        g_cnt += 1

        self.last_g = g

        #add min join torques as second objective
        if f1 > 0:
            f+= f1
            print("added cost: {}".format(f1))

        c = self.testConstraints(g)
        if c:
            print(Fore.GREEN, end=' ')
        else:
            print(Fore.YELLOW, end=' ')

        print("objective function value: {} (last best: {} + {})".format(f,
                                                            self.last_best_f-self.last_best_f_f1,
                                                            self.last_best_f_f1), end=' ')
        print(Fore.RESET)

        if self.config['verbose']:
            if self.opt_prob.is_gradient:
                print("(Gradient evaluation)")

        if self.mpi_rank == 0 and not self.opt_prob.is_gradient and self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        self.showVisualizerTrajectory(self.trajectory)

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
        g = np.array(g)
        c_s = self.num_constraints - self.num_coll_constraints  # start where collision constraints start
        res = np.all(g[:c_s] <= self.config['minTolConstr'])
        res_c = np.all(g[c_s:] > 0)
        if not res:
            print("constraints violated:")
            if True in np.in1d(list(range(1, 2*self.num_dofs)), np.where(g >= self.config['minTolConstr'])):
                print("- angle limits")
                print(np.array(g)[list(range(1, 2*self.num_dofs))])
            if True in np.in1d(list(range(2*self.num_dofs, 3*self.num_dofs)), np.where(g >= self.config['minTolConstr'])):
                print("- max velocity limits")
                #print np.array(g)[range(2*self.num_dofs,3*self.num_dofs)]
            if True in np.in1d(list(range(3*self.num_dofs, 4*self.num_dofs)), np.where(g >= self.config['minTolConstr'])):
                print("- max torque limits")

            if self.config['minVelocityConstraint']:
                if True in np.in1d(list(range(4*self.num_dofs, 5*self.num_dofs)), np.where(g >= self.config['minTolConstr'])):
                    print("- min velocity limits")

            if not res_c:
                print("- collision constraints")

            #if True in np.in1d(range(5*self.num_dofs,6*self.num_dofs), np.where(g >= self.config['minTolConstr'])):
            #    print "- min torque limits"
            #    print g[range(5*self.num_dofs,6*self.num_dofs)]
        return res and res_c


    def testParams(self, **kwargs):
        x = kwargs['x_new']
        return self.testBounds(x) and self.testConstraints(self.last_g)


    def addVarsAndConstraints(self, opt_prob):
        # type: (pyOpt.Optimization) -> None
        ''' add variables, define bounds '''

        # w_f - pulsation
        opt_prob.addVar('wf', 'c', value=self.wf_init, lower=self.wf_min, upper=self.wf_max)

        # q - offsets
        for i in range(self.num_dofs):
            opt_prob.addVar('q_%d'%i,'c', value=self.qinit[i], lower=self.qmin[i], upper=self.qmax[i])
        # a, b - sin/cos params
        for i in range(self.num_dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('a{}_{}'.format(i,j), 'c', value=self.ainit[i][j], lower=self.amin, upper=self.amax)
        for i in range(self.num_dofs):
            for j in range(self.nf[0]):
                opt_prob.addVar('b{}_{}'.format(i,j), 'c', value=self.binit[i][j], lower=self.bmin, upper=self.bmax)

        # add constraint vars (constraint functions are in obfunc)
        opt_prob.addConGroup('g', self.num_constraints, type='i', lower=0.0, upper=np.inf)
        #print opt_prob


    def optimizeTrajectory(self):
        # type: () -> PulsedTrajectory
        # use non-linear optimization to find parameters for minimal
        # condition number trajectory

        # Instanciate Optimization Problem
        opt_prob = pyOpt.Optimization('Trajectory optimization', self.objectiveFunc)
        opt_prob.addObj('f')
        self.opt_prob = opt_prob

        self.addVarsAndConstraints(opt_prob)
        sol_vec = self.runOptimizer(opt_prob)

        sol_wf, sol_q, sol_a, sol_b = self.vecToParams(sol_vec)
        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf)

        if self.config['showOptimizationGraph']:
            plt.ioff()

        return self.trajectory
