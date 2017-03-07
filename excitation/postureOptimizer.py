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

from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) >= LooseVersion('1.5'):
    plt.style.use('seaborn-pastel')

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

        # init some classes
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad
        self.trajectory = FixedPositionTrajectory(self.config)

        self.num_dofs = self.config['num_dofs']
        self.num_postures = self.config['numStaticPostures']
        self.num_constraints = self.model.num_links**2
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

    def testConstraints(self, g):
        return np.all(np.array(g) > 0.0)

    def getLinkDistance(self, l0, l1):
        '''get distance from link with id l0 to link with id l1'''

        #get link rotation and position in world frame
        #TODO: check that this is correct (dependent on angle?)
        #self.model.dynComp.setRobotState()
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

        distance, result = fcl.distance(o0, o1, collision_data.DistanceRequest(True))

        return distance

    def objectiveFunc(self, x):
        self.iter_cnt += 1
        print("iter #{}/{}".format(self.iter_cnt, self.iter_max))

        # init vars
        fail = False
        f = 0.0
        g = [0.0]*self.num_constraints

        # test constraints
        # check for each link that it does not collide with any other link (parent/child shouldn't be possible)
        for l0 in range(self.model.num_links):
            for l1 in range(self.model.num_links):
                if np.abs(l0 - l1) <= 2:  # same link or neighbors won't collide
                    g[l0*self.model.num_links + l1] = 10.0
                    continue

                # only need upper triangular part of coefficient matrix (distance l0,l1 = l1,l0)
                if l0 < l1:
                    g[l0*self.model.num_links + l1] = self.getLinkDistance(l0, l1)
                else:
                    # get symmetrical entry (already calculated)
                    g[l0*self.model.num_links + l1] = g[l1*self.model.num_links + l0]

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
        if self.config['showOptimizationGraph']:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        #if self.config['verbose']:
        print("Angles: {}".format(angles))
        print("Parameters: {}".format(param_error))
        print("Constraints (link distances): {}".format(np.reshape(g, (self.model.num_links, self.model.num_links))))
        print("objective function value: {} (last best: {})".format(f, self.last_best_f))

        #keep last best solution (some solvers don't keep it)
        if c and f < self.last_best_f:
            self.last_best_f = f
            self.last_best_sol = x

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
                initial = (self.limits[d_n]['upper'] - self.limits[d_n]['lower']) / 2
                opt_prob.addVar('p_{} q_{}'.format(p, d), type='c', value=initial,
                                lower=self.limits[d_n]['lower'], upper=self.limits[d_n]['upper'])

        # add constraints (functions are calculated in objectiveFunc())
        # for each link mesh distance to each other link, should be >0
        # TODO: reduce this to physically possible collisions
        opt_prob.addConGroup('g', self.num_constraints, type='i', lower=0.0)

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

        self.addVarsAndConstraints(self.opt_prob)
        sol_vec = self.runOptimizer(self.opt_prob)

        angles = self.vecToParam(sol_vec)
        self.trajectory.initWithAngles(angles)

        #if self.config['showOptimizationGraph']:
        #plt.ioff()

        return self.trajectory
