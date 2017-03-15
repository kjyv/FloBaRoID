from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import pyOpt
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
from fcl import fcl, collision_data, transform

from identification.helpers import URDFHelpers, eulerAnglesToRotationMatrix
from excitation.trajectoryGenerator import FixedPositionTrajectory
from excitation.optimizer import plotter, Optimizer
from visualizer import Visualizer

from colorama import Fore

class PostureOptimizer(Optimizer):
    ''' find angles of n static positions for identification of gravity parameters '''

    def __init__(self, config, idf, model, simulation_func, world=None):
        super(PostureOptimizer, self).__init__(config, model, simulation_func)

        self.idf = idf

        # get joint ranges
        self.limits = URDFHelpers.getJointLimits(config['urdf'], use_deg=False)  #will always be compared to rad

        self.trajectory = FixedPositionTrajectory(self.config)

        self.num_dofs = self.config['num_dofs']
        self.num_postures = self.config['numStaticPostures']

        #self.model.num_links**2 * self.num_postures
        self.posture_time = 0.01  # time in s per posture (*freq = equal samples)

        self.link_cuboid_hulls = {}  # type: Dict[str, List]
        for i in range(self.model.num_links):
            link_name = self.model.linkNames[i]
            box, pos, rot = idf.urdfHelpers.getBoundingBox(
                    input_urdf = idf.model.urdf_file,
                    old_com = idf.model.xStdModel[i*10+1:i*10+4] / idf.model.xStdModel[i*10],
                    link_name = link_name
            )
            self.link_cuboid_hulls[link_name] = [box, pos, rot]

        self.world = world
        if world:
            self.world_links = idf.urdfHelpers.getLinkNames(world)
            print('World links: {}'.format(self.world_links))
            for link_name in self.world_links:
                box, pos, rot = idf.urdfHelpers.getBoundingBox(
                        input_urdf = world,
                        old_com = [0,0,0],
                        link_name = link_name
                )
            # make sure no name collision happens
            if link_name not in self.link_cuboid_hulls:
                self.link_cuboid_hulls[link_name] = [box, pos, rot]
            else:
                print(Fore.RED+'Warning: link {} declared in model and world file!'.format(link_name) + Fore.RESET)

        vel = [0.0]*self.num_dofs
        self.dq_zero = iDynTree.VectorDynSize.fromList(vel)
        self.world_gravity = iDynTree.SpatialAcc.fromList(self.model.gravity)

        self.idyn_model = iDynTree.Model()
        iDynTree.modelFromURDF(self.config['urdf'], self.idyn_model)

        # get neighbors for each link
        self.neighbors = {}   # type: Dict[str, Dict[str, List[int]]]
        for l in range(self.idyn_model.getNrOfLinks()):
            link_name = self.idyn_model.getLinkName(l)
            #if link_name not in self.model.linkNames:  # ignore links that are ignored in the generator
            #    continue
            self.neighbors[link_name] = {'links':[], 'joints':[]}
            num_neighbors = self.idyn_model.getNrOfNeighbors(l)
            for n in range(num_neighbors):
                nb = self.idyn_model.getNeighbor(l, n)
                self.neighbors[link_name]['links'].append(self.idyn_model.getLinkName(nb.neighborLink))
                self.neighbors[link_name]['joints'].append(self.idyn_model.getJointName(nb.neighborJoint))

        # for each neighbor link, add links connected via a fixed joint also as neighbors
        self.neighbors_tmp = self.neighbors.copy()  # don't modify in place so no recursive loops happen
        for l in range(self.idyn_model.getNrOfLinks()):
            link_name = self.idyn_model.getLinkName(l)
            for nb in self.neighbors_tmp[link_name]['links']:  # look at all neighbors of l
                for j_name in self.neighbors_tmp[nb]['joints']:  # check each joint of a neighbor of l
                    j = self.idyn_model.getJoint(self.idyn_model.getJointIndex(j_name))
                    # check all connected joints if they are fixed, if so add connected link as neighbor
                    if j.isFixedJoint():
                        j_l0 = j.getFirstAttachedLink()
                        j_l1 = j.getSecondAttachedLink()
                        if j_l0 == self.idyn_model.getLinkIndex(nb):
                            nb_fixed = j_l1
                        else:
                            nb_fixed = j_l0
                        nb_fixed_name = self.idyn_model.getLinkName(nb_fixed)
                        if nb_fixed != l and nb_fixed_name not in self.neighbors[link_name]['links']:
                            self.neighbors[link_name]['links'].append(nb_fixed_name)

        # amount of collision checks to be done
        eff_links = self.model.num_links - len(self.config['ignoreLinksForCollision'])
        self.num_constraints = self.num_postures * ((eff_links * (eff_links-1) // 2) + len(self.world_links)*eff_links)

        #subtract neighbors
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

        if self.config['showModelVisualization'] and self.mpi_rank == 0:
            self.visualizer = Visualizer(self.config)


    def testConstraints(self, g):
        return np.all(g > 0.0)

    def getLinkDistance(self, l0_name, l1_name, joint_q):
        '''get distance from link with id l0 to link with id l1 for posture joint_q'''

        #get link rotation and position in world frame
        q = iDynTree.VectorDynSize.fromList(joint_q)
        self.model.dynComp.setRobotState(q, self.dq_zero, self.dq_zero, self.world_gravity)

        if l0_name in self.model.linkNames:    # if robot link
            f0 = self.model.dynComp.getFrameIndex(l0_name)
            t0 = self.model.dynComp.getWorldTransform(f0)
            rot0 = t0.getRotation().toNumPy()
            pos0 = t0.getPosition().toNumPy()
            s0 = self.config['scaleCollisionHull']
        else:   # if world link
            pos0 = self.link_cuboid_hulls[l0_name][1]
            rot0 = eulerAnglesToRotationMatrix(self.link_cuboid_hulls[l0_name][2])
            s0 = 1

        if l1_name in self.model.linkNames:    # if robot link
            f1 = self.model.dynComp.getFrameIndex(l1_name)
            t1 = self.model.dynComp.getWorldTransform(f1)
            rot1 = t1.getRotation().toNumPy()
            pos1 = t1.getPosition().toNumPy()
            s1 = self.config['scaleCollisionHull']
        else:   # if world link
            pos1 = self.link_cuboid_hulls[l1_name][1]
            rot1 = eulerAnglesToRotationMatrix(self.link_cuboid_hulls[l1_name][2])
            s1 = 1

        # TODO: use pos and rot of boxes for vals from geometry tags
        # self.link_cuboid_hulls[l0_name][1], [2]

        b = np.array(self.link_cuboid_hulls[l0_name][0]) * s0
        b0_center = 0.5*np.array([np.abs(b[0][1])-np.abs(b[0][0]),
                                  np.abs(b[1][1])-np.abs(b[1][0]),
                                  np.abs(b[2][1])-np.abs(b[2][0])])
        b0 = fcl.Box(b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0])

        b = np.array(self.link_cuboid_hulls[l1_name][0]) * s1
        b1_center = 0.5*np.array([np.abs(b[0][1])-np.abs(b[0][0]),
                                  np.abs(b[1][1])-np.abs(b[1][0]),
                                  np.abs(b[2][1])-np.abs(b[2][0])])
        b1 = fcl.Box(b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0])

        # move box to pos + box center pos (model has pos in link origin, box has zero at center)
        o0 = fcl.CollisionObject(b0, transform.Transform(rot0, pos0+b0_center))
        o1 = fcl.CollisionObject(b1, transform.Transform(rot1, pos1+b1_center))

        distance, d_result = fcl.distance(o0, o1, collision_data.DistanceRequest(True))

        if distance < 0:
            if self.config['verbose'] > 1:
                print("Collision of {} and {}".format(l0_name, l1_name))

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
        if self.mpi_size > 1:
            print("process {}, iter #{}/{}".format(self.mpi_rank, self.iter_cnt, self.iter_max))
        else:
            print("iter #{}/{}".format(self.iter_cnt, self.iter_max))

        # init vars
        fail = False
        f = 0.0
        #g = np.zeros((self.num_postures, self.model.num_links, self.model.num_links))
        #assert(g.size == self.num_constraints)  # needs to stay in sync
        g = np.zeros(self.num_constraints)

        # test constraints
        # check for each link that it does not collide with any other link (parent/child shouldn't be possible)

        if self.config['showModelVisualization'] and self.mpi_rank == 0:
            def draw_model():
                p_id = self.visualizer.display_index
                q0 = x[p_id*self.num_dofs:(p_id+1)*self.num_dofs]
                q = iDynTree.VectorDynSize.fromList(q0)
                self.model.dynComp.setRobotState(q, self.dq_zero, self.dq_zero, self.world_gravity)
                self.visualizer.addIDynTreeModel(self.model.dynComp, self.link_cuboid_hulls,
                                                 self.model.linkNames, self.config['ignoreLinksForCollision'])
                if self.world:
                    world_boxes = {link: self.link_cuboid_hulls[link] for link in self.world_links}
                    self.visualizer.addWorld(world_boxes)
                self.visualizer.run()
            self.visualizer.display_max = self.num_postures
            self.visualizer.event_callback = draw_model
            self.visualizer.event_callback()

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

        # get objective function value: identified parameter distance (from 'real')
        #id_grav = self.model.identified_params
        id_grav = []
        id_grav_id = []
        for i in range(self.model.num_links):
            id_grav.append(i*10+1)
            id_grav.append(i*10+2)
            id_grav.append(i*10+3)
            id_grav_id.append(i*4+1)
            id_grav_id.append(i*4+6)
            id_grav_id.append(i*4+7)
        param_error = self.idf.xStdReal[id_grav] - self.idf.model.xStd[id_grav_id]
        f = np.linalg.norm(param_error)**2

        c = self.testConstraints(g)
        if self.config['showOptimizationGraph'] and not self.opt_prob.is_gradient and self.mpi_rank == 0:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        print("Objective function value: {} (last best: {})".format(f, self.last_best_f))

        if self.config['verbose']:
            if self.opt_prob.is_gradient:
                print("(Gradient evaluation)")
            print("Parameter error: {}".format(param_error))
            if self.config['verbose'] > 1:
                print("Angles: {}".format(angles))
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
                if self.config['useDeg']:
                    low = np.deg2rad(low)
                    high = np.deg2rad(high)

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
        self.local_iter_max = ((num_vars  + self.num_constraints) * self.config['localOptIterations'] + 2*num_vars)
        if self.parallel:
            self.local_iter_max = self.local_iter_max // self.mpi_size

        sol_vec = self.runOptimizer(self.opt_prob)

        angles = self.vecToParam(sol_vec)
        self.trajectory.initWithAngles(angles)

        # keep plot windows open (if any)
        plt.ioff()
        plt.show(block=True)

        return self.trajectory
