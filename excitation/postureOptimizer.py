from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import pyOpt
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from fcl import fcl, collision_data, transform

from identification.model import Model
from identification.data import Data
from identification.helpers import URDFHelpers
from excitation.trajectoryGenerator import Trajectory, FixedPositionTrajectory
from excitation.optimizer import plotter, Optimizer


class PostureOptimizer(Optimizer):
    ''' find angles of n static positions for identification of gravity parameters '''

    def __init__(self, config, idf, model, simulation_func):
        super(PostureOptimizer, self).__init__(config, model, simulation_func)

        self.idf = idf

        # get joint ranges
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad

        self.trajectory = FixedPositionTrajectory(self.config)

        self.num_dofs = self.config['num_dofs']
        self.num_postures = self.config['numStaticPostures']

        #self.model.num_links**2 * self.num_postures
        self.num_constraints = self.num_postures * (self.model.num_links * (self.model.num_links-1) // 2)
        self.posture_time = 0.05  # time in s per posture

        self.link_cuboid_hulls = []  # type: List[np.ndarray]
        for i in range(self.model.num_links):
            self.link_cuboid_hulls.append(np.array(
                idf.urdfHelpers.getBoundingBox(
                    input_urdf = idf.model.urdf_file,
                    old_com = idf.model.xStdModel[i*10+1:i*10+4] / idf.model.xStdModel[i*10],
                    link_nr = i
                )
            ))

        vel = [0.0]*self.num_dofs
        self.dq_zero = iDynTree.VectorDynSize.fromList(vel)
        self.world_gravity = iDynTree.SpatialAcc.fromList(self.model.gravity)

    def testConstraints(self, g):
        return np.all(g > 0.0)

    def getLinkDistance(self, l0, l1, joint_q):
        '''get distance from link with id l0 to link with id l1 for posture joint_q'''

        #get link rotation and position in world frame
        q = iDynTree.VectorDynSize.fromList(joint_q)
        self.model.dynComp.setRobotState(q, self.dq_zero, self.dq_zero, self.world_gravity)

        f0 = self.model.dynComp.getFrameIndex(self.model.linkNames[l0])
        t0 = self.model.dynComp.getWorldTransform(f0)
        rot0 = t0.getRotation().toNumPy()
        pos0 = t0.getPosition().toNumPy()

        f1 = self.model.dynComp.getFrameIndex(self.model.linkNames[l1])
        t1 = self.model.dynComp.getWorldTransform(f1)
        rot1 = t1.getRotation().toNumPy()
        pos1 = t1.getPosition().toNumPy()

        b = self.link_cuboid_hulls[l0]
        b0 = fcl.Box(b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0])

        b = self.link_cuboid_hulls[l1]
        b1 = fcl.Box(b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0])

        o0 = fcl.CollisionObject(b0, transform.Transform(rot0, pos0))
        o1 = fcl.CollisionObject(b1, transform.Transform(rot1, pos1))

        distance, d_result = fcl.distance(o0, o1, collision_data.DistanceRequest(True))

        if distance < 0:
            print("Collision of {} and {}".format(self.model.linkNames[l0], self.model.linkNames[l1]))

            # get proper collision and depth since optimization should also know how much constraint is violated
            cr = collision_data.CollisionRequest()
            cr.enable_contact = True
            cr.enable_cost = True
            collision, c_result = fcl.collide(o0, o1, cr)

            # sometimes no collision is found?
            if len(c_result.contacts):
                distance = c_result.contacts[0].penetration_depth


        return distance


    def objectiveFunc(self, x):
        self.iter_cnt += 1
        print("iter #{}/{}".format(self.iter_cnt, self.iter_max))

        # init vars
        fail = False
        f = 0.0
        #g = np.zeros((self.num_postures, self.model.num_links, self.model.num_links))
        #assert(g.size == self.num_constraints)  # needs to stay in sync
        g = np.zeros(self.num_constraints)

        # test constraints
        # check for each link that it does not collide with any other link (parent/child shouldn't be possible)
        g_cnt = 0
        for p in range(self.num_postures):
            q = x[p*self.num_dofs:(p+1)*self.num_dofs]
            for l0 in range(self.model.num_links-1):
                for l1 in range(self.model.num_links):
                    if (l0 == l1): # same link never collides
                        continue

                    if (l1 - l0) == 1:    # neighbors should not be able to collide because of joint range
                        g[g_cnt] = 10.0
                        g_cnt += 1
                        continue

                    if l0 < l1:
                        g[g_cnt] = self.getLinkDistance(l0, l1, q)
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

        # get objective function value: identified parameter distance (from 'real')
        id_grav = self.model.identified_params
        param_error = self.idf.xStdReal[id_grav] - self.idf.model.xStd
        f = np.linalg.norm(param_error)**2

        c = self.testConstraints(g)
        if not self.opt_prob.is_gradient and self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        print("Objective function value: {} (last best: {})".format(f, self.last_best_f))
        if self.opt_prob.is_gradient:
            print("(Gradient evaluation)")
        print("Parameter error: {}".format(param_error))

        if self.config['verbose']:
            #print("Angles: {}".format(angles))
            print("Constraints (link distances): {}".format(g))

        #keep last best solution (some solvers don't keep it)
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
                else:
                    low = self.limits[d_n]['lower']
                    high = self.limits[d_n]['upper']
                #initial = (high - low) / 2
                initial = 0.0

                opt_prob.addVar('p_{} q_{}'.format(p, d), type='c', value=initial,
                                lower=low, upper=high)

        # add constraints (functions are calculated in objectiveFunc())
        # for each link mesh distance to each other link, should be >0
        # TODO: reduce this to physically possible collisions from table
        opt_prob.addConGroup('g', self.num_constraints, type='i', lower=0.0, upper=np.inf)

    def vecToParam(self, x):
        # type: (np.ndarray) -> List[Dict[str, Any]]
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

        #slsqp/psqp
        #self.local_iter_max = self.num_postures * self.num_dofs * self.config['localOptIterations']  # num of gradient evals
        #self.local_iter_max += self.config['localOptIterations']*2  # some steps for each iter?

        #ipopt, not really correct
        num_vars = self.num_postures * self.num_dofs
        self.local_iter_max = (2*num_vars  + self.num_constraints) * self.config['localOptIterations'] + 2*num_vars

        sol_vec = self.runOptimizer(self.opt_prob)

        angles = self.vecToParam(sol_vec)
        self.trajectory.initWithAngles(angles)

        # keep plot windows open (if any)
        plt.ioff()
        plt.show(block=True)

        return self.trajectory
