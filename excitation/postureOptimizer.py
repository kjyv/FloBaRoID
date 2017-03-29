from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import pyOpt
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from identification.helpers import URDFHelpers
from excitation.trajectoryGenerator import FixedPositionTrajectory
from excitation.optimizer import plotter, Optimizer


class PostureOptimizer(Optimizer):
    ''' find angles of n static positions for identification of gravity parameters '''

    def __init__(self, config, idf, model, simulation_func, world=None):
        # type: (Dict[str, Any], Identification, Model, Callable[[Dict, Trajectory, Model, np._ArrayLike], Tuple[Dict, Data]], str) -> None
        super(PostureOptimizer, self).__init__(config, idf, model, simulation_func, world=world)

        self.idf = idf

        # get joint ranges
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad
        self.trajectory = FixedPositionTrajectory(self.config)

        self.num_postures = self.config['numStaticPostures']
        self.posture_time = self.config['staticPostureTime']

        self.idyn_model = iDynTree.Model()
        iDynTree.modelFromURDF(self.config['urdf'], self.idyn_model)

        self.neighbors = URDFHelpers.getNeighbors(self.idyn_model)

        # amount of collision checks to be done
        eff_links = self.model.num_links - len(self.config['ignoreLinksForCollision']) + len(self.world_links)
        self.num_constraints = self.num_postures * (eff_links * (eff_links-1) // 2)

        #get neighbors
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
        self.num_constraints -= self.num_postures * (len(nb_pairs) +        # neighbors
                                  len(self.config['ignoreLinkPairsForCollision']))  # custom combinations

        # only generate output from main process
        if self.mpi_rank > 0:
            self.config['verbose'] = 0

        self.initVisualizer()

    def testConstraints(self, g):
        return np.all(g > 0.0)

    def objectiveFunc(self, x, test=False):
        self.iter_cnt += 1
        if self.mpi_size > 1:
            print("process {}, iter #{}/{}".format(self.mpi_rank, self.iter_cnt, self.iter_max))
        else:
            print("call #{}/{}".format(self.iter_cnt, self.iter_max))

        # init vars
        fail = False
        f = 0.0
        #g = np.zeros((self.num_postures, self.model.num_links, self.model.num_links))
        #assert(g.size == self.num_constraints)  # needs to stay in sync
        g = np.zeros(self.num_constraints)

        # test constraints
        # check for each link that it does not collide with any other link (parent/child shouldn't be possible)

        g_cnt = 0
        if self.config['verbose'] > 1:
            print('checking collisions')
        for p in range(self.num_postures):
            if self.config['verbose'] > 1:
                print("Posture {}".format(p))
            q = x[p*self.num_dofs:(p+1)*self.num_dofs]

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
                        g[g_cnt] = self.getLinkDistance(l0_name, l1_name, q)
                        g_cnt += 1

        # check those links that are very close or collide again with mesh (simplified versions or full)
        # TODO: possibly limit distance of overall COM from hip (simple balance?)

        # simulate with current angles
        angles = self.vecToParam(x)
        self.trajectory.initWithAngles(angles)
        old_verbose = self.config['verbose']
        self.config['verbose'] = 0
        trajectory_data, data = self.sim_func(self.config, self.trajectory, model=self.model)

        # identify parameters with this trajectory
        self.idf.data.init_from_data(trajectory_data)
        self.idf.estimateParameters()
        self.config['verbose'] = old_verbose

        if self.config['showOptimizationTrajs']:
            plotter(self.config, data=trajectory_data)

        '''
        # get objective function value: identified parameter distance (from 'real')
        id_grav = self.model.identified_params
        id_grav_id = list(range(0, len(self.idf.model.xStd)))
        if self.config['identifyGravityParamsOnly']:
            id_grav = []
            id_grav_id = []
            # only compare COM params
            for i in range(self.model.num_links):
                id_grav.append(i*10+1)
                id_grav.append(i*10+2)
                id_grav.append(i*10+3)
                id_grav_id.append(i*4+1)
                id_grav_id.append(i*4+6)
                id_grav_id.append(i*4+7)
        param_error = self.idf.xStdReal[id_grav] - self.idf.model.xStd[id_grav_id]
        '''
        param_error = self.idf.xBaseReal - self.idf.model.xBase
        f = np.linalg.norm(param_error)**2 #+ np.std(param_error)

        c = self.testConstraints(g)
        if self.config['showOptimizationGraph'] and not self.opt_prob.is_gradient and self.mpi_rank == 0:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        self.showVisualizerAngles(x)

        print("Objective function value: {} (last best: {})".format(f, self.last_best_f))

        if self.config['verbose']:
            if self.opt_prob.is_gradient:
                print("(Gradient evaluation)")
            print("Parameter error: {}".format(param_error))
            if self.config['verbose'] > 1:
                print("Angles: {}".format(angles))
                print("Constraints ({} link distances): {}".format(len(g), list(g)))

        #keep last best solution (some solvers don't properly keep it or don't consider gradient values)
        if c and f < self.last_best_f:
            self.last_best_f = f
            self.last_best_sol = x
        elif not c:
            print('Constraints not met.')

        return f, g, fail


    def addVarsAndConstraints(self, opt_prob):
        # type: (pyOpt.Optimization) -> None
        ''' add variables, define bounds
            variable type: 'c' - continuous, 'i' - integer, 'd' - discrete (choices)
            constraint types: 'i' - inequality, 'e' - equality
        '''

        # add objective
        opt_prob.addObj('f')

        # add variables: angles for each posture
        for p in range(self.num_postures):
            for d in range(self.num_dofs):
                d_n = self.model.jointNames[d]
                if len(self.config['trajectoryAngleRanges']) > d and \
                        self.config['trajectoryAngleRanges'][d] is not None:
                    low = self.config['trajectoryAngleRanges'][d][0]
                    high = self.config['trajectoryAngleRanges'][d][1]
                    if self.config['useDeg']:
                        low = np.deg2rad(low)
                        high = np.deg2rad(high)
                else:
                    low = self.limits[d_n]['lower']
                    high = self.limits[d_n]['upper']

                #initial = (high - low) / 2
                #initial = 0.0
                if len(self.config['initialPostures']) > p:
                    initial = self.config['initialPostures'][p][d]
                    if self.config['useDeg']:
                        initial = np.deg2rad(initial)
                else:
                    initial = 0.0

                opt_prob.addVar('p_{} q_{}'.format(p, d), type='c', value=initial,
                                lower=low, upper=high)

        # add constraints (functions are calculated in objectiveFunc())
        # for each link mesh distance to each other link, should be >0
        opt_prob.addConGroup('g', self.num_constraints, type='i', lower=0.0, upper=np.inf)

    def vecToParam(self, x):
        # type: (np._ArrayLike[float]) -> List[Dict[str, Any]]
        # put solution vector into form for trajectory class
        angles = []    # type: List[Dict[str, Any]]     # matrix angles for each posture
        for n in range(self.num_postures):
            angles.append({'start_time': n*self.posture_time,
                           'angles': x[n*self.num_dofs:(n+1)*self.num_dofs],
                          })
        return angles

    def optimizeTrajectory(self):
        # type: () -> FixedPositionTrajectory
        # use non-linear optimization to find parameters

        ## describe optimization problem with pyOpt classes

        # Instanciate Optimization Problem
        self.opt_prob = pyOpt.Optimization('Posture optimization', self.objectiveFunc)

        # set if the available pyOpt doesn't have gradient flag (telling when objfunc is called for gradient)
        if 'is_gradient' not in self.opt_prob.__dict__:
            self.opt_prob.is_gradient = False

        self.addVarsAndConstraints(self.opt_prob)
        #print(self.opt_prob)

        if self.config['localSolver'] in ['SLSQP', 'PSQP']:
            #slsqp/psqp

            num_vars = self.num_postures * self.num_dofs
            # num of gradient evals divided by parallel processes times iterations
            self.local_iter_max = (num_vars*2  // self.mpi_size) * self.config['localOptIterations']
        else:
            #ipopt, not really correct
            num_vars = self.num_postures * self.num_dofs
            self.local_iter_max = ((num_vars + self.num_constraints)  // self.mpi_size) * self.config['localOptIterations']

        sol_vec = self.runOptimizer(self.opt_prob)

        angles = self.vecToParam(sol_vec)
        self.trajectory.initWithAngles(angles)

        # keep plot windows open (if any)
        plt.ioff()
        plt.show(block=True)

        return self.trajectory

