from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Tuple, Dict
import sys
import random

import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
if LooseVersion(matplotlib.__version__) >= LooseVersion('1.5'):
    plt.style.use('seaborn-pastel')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    parallel = nprocs > 1
except:
    parallel = False

from colorama import Fore
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from identification.helpers import eulerAnglesToRotationMatrix

def plotter(config, data=None, filename=None):
    #type: (Dict, np._ArrayLike, str) -> None
    fig = plt.figure(1)
    fig.clear()
    if False:
        from random import sample
        from itertools import permutations

        # get a random color wheel
        Nlines = 200
        color_lvl = 8
        rgb = np.array(list(permutations(list(range(0,256,color_lvl),3))))/255.0
        colors = sample(rgb,Nlines)
        print(colors[0:config['num_dofs']])
    else:
        # set some fixed colors
        colors = [[ 0.97254902,  0.62745098,  0.40784314],
                  [ 0.0627451 ,  0.53333333,  0.84705882],
                  [ 0.15686275,  0.75294118,  0.37647059],
                  [ 0.90980392,  0.37647059,  0.84705882],
                  [ 0.84705882,  0.        ,  0.1254902 ],
                  [ 0.18823529,  0.31372549,  0.09411765],
                  [ 0.50196078,  0.40784314,  0.15686275]
                 ]

        from palettable.tableau import Tableau_10, Tableau_20
        colors += Tableau_10.mpl_colors[0:6] + Tableau_20.mpl_colors + Tableau_20.mpl_colors

    if not data:
        # reload measurements from this or last run (if run dry)
        measurements = np.load(filename)
        Q = measurements['positions']
        Qraw = measurements['positions_raw']
        V = measurements['velocities']
        Vraw = measurements['velocities_raw']
        dV = measurements['accelerations']
        Tau = measurements['torques']
        TauRaw = measurements['torques_raw']
        if 'plot_targets' in config and config['plot_targets']:
            Q_t = measurements['target_positions']
            V_t = measurements['target_velocities']
            dV_t = measurements['target_accelerations']
        T = measurements['times']
        num_samples = measurements['positions'].shape[0]
    else:
        Q = data['positions']
        Qraw = data['positions']
        Q_t = data['target_positions']
        V = data['velocities']
        Vraw = data['velocities']
        V_t = data['target_velocities']
        dV = data['accelerations']
        dV_t = data['target_accelerations']
        Tau = data['torques']
        TauRaw = data['torques']
        T = data['times']
        num_samples = data['positions'].shape[0]

    if config['verbose']:
        print('loaded {} measurement samples'.format(num_samples))

    if 'plot_targets' in config and config['plot_targets']:
        print("tracking error per joint:")
        for i in range(0, config['num_dofs']):
            sse = np.sum((Q[:, i] - Q_t[:, i]) ** 2)
            print("joint {}: {}".format(i, sse))

    #print("histogram of time diffs")
    #dT = np.diff(T)
    #H, B = np.histogram(dT)
    #plt.hist(H, B)
    #late_msgs = (1 - float(np.sum(H)-np.sum(H[1:])) / float(np.sum(H))) * 100
    #print("bins: {}".format(B))
    #print("sums: {}".format(H))
    #print("({}% messages too late)\n".format(late_msgs))

    # what to plot (each tuple has a title and one or multiple data arrays)
    if 'plot_targets' in config and config['plot_targets']:
        datasets = [
            ([Q_t,], 'Target Positions'),
            ([V_t,], 'Target Velocities'),
            ([dV_t,], 'Target Accelerations')
            ]
    else:   #plot measurements and raw data (from measurements file)
        if np.sum(Qraw - Q) != 0:
            datasets = [
                ([Q, Qraw], 'Positions'),
                ([V, Vraw],'Velocities'),
                ([dV,], 'Accelerations'),
                ([Tau, TauRaw],'Measured Torques')
                ]
        else:
            datasets = [
                ([Q], 'Positions'),
                ([V],'Velocities'),
                ([dV,], 'Accelerations'),
                ([Tau],'Measured Torques')
                ]

    d = 0
    cols = 2.0
    rows = round(len(datasets)/cols)
    for (dat, title) in datasets:
        plt.subplot(rows, cols, d+1)
        plt.title(title)
        lines = list()
        labels = list()
        for d_i in range(0, len(dat)):
            if len(dat[d_i].shape) > 1:
                for i in range(0, config['num_dofs']):
                    l = config['jointNames'][i] if d_i == 0 else ''  #only put joint names in the legend once
                    labels.append(l)
                    line = plt.plot(T, dat[d_i][:, i], color=colors[i], alpha=1-(d_i/2.0))
                    lines.append(line[0])
            else:
                #dat vector
                plt.plot(T, dat[d_i], label=title, color=colors[0], alpha=1-(d_i/2.0))
        d+=1
    leg = plt.figlegend(lines, labels, 'upper right', fancybox=True, fontsize=10)
    leg.draggable()

    plt.subplots_adjust(hspace=2)
    plt.tight_layout()

    plt.show()


class Optimizer(object):
    '''base class for different optimizers'''
    def __init__(self, config, idf, model, simulation_func, world=None):
        # type: (Dict[str, Any], Identification, Model, Callable[[Dict, Trajectory, Model, np._ArrayLike], Tuple[Dict, Data]], str) -> None

        self.config = config
        self.sim_func = simulation_func
        self.model = model
        self.idf = idf
        self.world = world

        self.num_dofs = self.config['num_dofs']

        # optimization status vars
        self.last_best_f = np.inf
        self.last_best_sol = np.array([])

        self.iter_cnt = 0   # iteration counter
        self.last_g = None  # type: List[float]    # last constraint values
        self.is_global = False
        self.local_iter_max = "(unknown)"

        # init parallel runs
        self.parallel = parallel
        if parallel:
            self.mpi_size = nprocs   # number of processes
            self.mpi_rank = comm.Get_rank()  # current process
            self.comm = comm
        else:
            self.mpi_size = 1
            self.mpi_rank = 0

        # init plotting progress
        if self.config['showOptimizationGraph']:
            self.initGraph()

        # init link data
        self.link_cuboid_hulls = {}  # type: Dict[str, List]
        for i in range(self.model.num_links):
            link_name = self.model.linkNames[i]
            box, pos, rot = idf.urdfHelpers.getBoundingBox(
                    input_urdf = self.model.urdf_file,
                    old_com = self.model.xStdModel[i*10+1:i*10+4] / self.model.xStdModel[i*10],
                    link_name = link_name,
                    scaling = False
            )
            self.link_cuboid_hulls[link_name] = [box, pos, rot]

        self.world = world
        self.world_links = []  # type: List[str]
        if world:
            self.world_links = idf.urdfHelpers.getLinkNames(world)
            if self.config['verbose']:
                print('World links: {}'.format(self.world_links))
            for link_name in self.world_links:
                box, pos, rot = idf.urdfHelpers.getBoundingBox(
                        input_urdf = world,
                        old_com = [0,0,0],
                        link_name = link_name,
                        scaling = False
                )
                # make sure no name collision happens
                if link_name not in self.link_cuboid_hulls:
                    self.link_cuboid_hulls[link_name] = [box, pos, rot]
                else:
                    print(Fore.RED+'Warning: link {} declared in model and world file!'.format(link_name) + Fore.RESET)

        self.world_boxes = {link: self.link_cuboid_hulls[link] for link in self.world_links}

        # init some vars for link distance
        vel = [0.0]*self.num_dofs
        self.dq_zero = iDynTree.VectorDynSize.fromList(vel)
        self.world_gravity = iDynTree.SpatialAcc.fromList(self.model.gravity)


    def testBounds(self, x):
        # type: (np._ArrayLike) -> bool
        raise NotImplementedError

    def testConstraints(self, g):
        # type: (np._ArrayLike) -> bool
        raise NotImplementedError

    def getLinkDistance(self, l0_name, l1_name, joint_q):
        # type: (str, str, np._ArrayLike[float]) -> float
        '''get shortest distance from link with id l0 to link with id l1 for posture joint_q'''

        from fcl import fcl, collision_data, transform

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
        p = np.array(self.link_cuboid_hulls[l0_name][1])
        b0_center = 0.5*np.array([(b[1][0])+(b[0][0]) + p[0],
                                  (b[1][1])+(b[0][1]) + p[1],
                                  (b[1][2])+(b[0][2]) + p[2]])
        b0 = fcl.Box(b[1][0]-b[0][0], b[1][1]-b[0][1], b[1][2]-b[0][2])

        b = np.array(self.link_cuboid_hulls[l1_name][0]) * s1
        p = np.array(self.link_cuboid_hulls[l1_name][1])
        b1_center = 0.5*np.array([(b[1][0])+(b[0][0]) + p[0],
                                  (b[1][1])+(b[0][1]) + p[1],
                                  (b[1][2])+(b[0][2]) + p[2]])
        b1 = fcl.Box(b[1][0]-b[0][0], b[1][1]-b[0][1], b[1][2]-b[0][2])

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

            # sometimes no contact is found even though distance is less than 0?
            if len(c_result.contacts):
                distance = c_result.contacts[0].penetration_depth

        return distance

    def initVisualizer(self):
        if self.config['showModelVisualization'] and self.mpi_rank == 0:
            from visualizer import Visualizer
            self.visualizer = Visualizer(self.config)
            self.visualizer.loadMeshes(self.model.urdf_file, self.model.linkNames, self.idf.urdfHelpers)

            # set draw method for visualizer. This taps into local variables here, a bit unclean...
            def draw_model():
                if self.trajectory:
                    # get data of trajectory
                    self.visualizer.trajectory.setTime(self.visualizer.display_index/self.visualizer.fps)
                    q0 = [self.visualizer.trajectory.getAngle(d) for d in range(self.num_dofs)]
                else:
                    p_id = self.visualizer.display_index
                    q0 = self.visualizer.angles[p_id*self.num_dofs:(p_id+1)*self.num_dofs]

                q = iDynTree.VectorDynSize.fromList(q0)
                dq = iDynTree.VectorDynSize.fromList([0.0]*self.num_dofs)
                self.model.dynComp.setRobotState(q, dq, dq, self.world_gravity)
                self.visualizer.addIDynTreeModel(self.model.dynComp, self.link_cuboid_hulls,
                        self.model.linkNames, self.config['ignoreLinksForCollision'])
                if self.world:
                    self.visualizer.addWorld(self.world_boxes)
                self.visualizer.updateLabels()
            self.visualizer.event_callback = draw_model

    def showVisualizerAngles(self, x):
        '''show visualizer for current joint angles x'''
        if self.config['showModelVisualization'] and self.mpi_rank == 0: #and c:
            self.visualizer.display_max = self.num_postures
            self.visualizer.angles = x
            self.visualizer.event_callback()
            self.visualizer.run()

    def showVisualizerTrajectory(self, t):
        '''show visualizer for joint trajectory t'''
        if self.config['showModelVisualization'] and self.mpi_rank == 0: #and c:
            self.visualizer.setModelTrajectory(t)
            freq = self.config['excitationFrequency']
            self.visualizer.display_max = t.getPeriodLength()*self.visualizer.fps # length of trajectory
            self.visualizer.trajectory = t
            self.visualizer.playable = True
            self.visualizer.event_callback()
            self.visualizer.run()

    def objectiveFunc(self, x, test=False):
        # type: (np._ArrayLike[float], bool) -> Tuple[float, np._ArrayLike, bool]
        ''' calculate objective function and return objective function value f, constraint values g
        and a fail flag'''
        raise NotImplementedError

    def initGraph(self):
        if self.mpi_rank > 0:
            return
        # init graphing of objective function value
        self.fig = plt.figure(0)
        self.ax1 = self.fig.add_subplot(1,1,1)
        plt.ion()
        self.xar = []           # type: List[int]    # x value, i.e. iteration count
        self.yar = []           # type: List[float]  # y value, i.e. obj func value
        self.x_constr = []      # type: List[bool]   # within constraints or not (feasible)
        self.ax1.plot(self.xar, self.yar)
        self.ax1.set_xlabel('Function evaluation #')
        self.ax1.set_ylabel('Objective function value')

        self.updateGraphEveryVals = 5

    def updateGraph(self):
        if self.mpi_rank > 0:
            return
        # draw all optimization steps, mark the ones that are within constraints
        #if (self.iter_cnt % self.updateGraphEveryVals) == 0:
        color = 'g'
        line = self.ax1.plot(self.xar, self.yar, marker='.', markeredgecolor=color, markerfacecolor=color, color="0.75")
        markers = np.where(self.x_constr)[0]
        line[0].set_markevery(list(markers))

        if self.iter_cnt == 1: plt.show(block=False)

        # allow some interaction
        plt.pause(0.01)


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
        f0 = f(*((x0,) + args))
        jac = np.zeros((x0.size, f0.size))
        dx = np.zeros(x0.size)
        for i in range(x0.size):
            dx[i] = epsilon
            jac[i] = (f(*((x0 + dx,) + args)) - f0) / epsilon
            dx[i] = 0.0
        return jac.transpose()

    def gather_solutions(self):
        # send best solutions to node 0
        if self.parallel:
            if self.config['verbose']:
                print('Collecting best solutions from processes')
            send_obj = [self.last_best_f, self.last_best_sol, self.mpi_rank]
            received_objs = self.comm.gather(send_obj, root=0)

            #receive solutions from other instances
            if self.mpi_rank == 0:
                for proc in range(0, self.mpi_size):
                    other_best_f, other_best_sol, rank = received_objs[proc]

                    if other_best_f < self.last_best_f:
                        print('received better solution from {}'.format(rank))
                        self.last_best_f = other_best_f
                        self.last_best_sol = other_best_sol

    def runOptimizer(self, opt_prob):
        # type: (pyOpt.Optimization) -> np._ArrayLike[float]
        ''' call global followed by local optimizer, return solution '''

        import pyOpt

        initial = [v.value for v in list(opt_prob.getVarSet().values())]

        if self.config['useGlobalOptimization']:
            ### optimize using pyOpt (global)
            sr = random.SystemRandom()
            if self.config['globalSolver'] == 'NSGA2':
                if parallel:
                    opt = pyOpt.NSGA2(pll_type='POA') # genetic algorithm
                else:
                    opt = pyOpt.NSGA2()
                if self.config['globalOptSize'] % 4:
                    raise IOError("globalOptSize needs to be a multiple of 4 for NSGA2")
                opt.setOption('PopSize', self.config['globalOptSize'])   # Population Size (a Multiple of 4)
                opt.setOption('maxGen', self.config['globalOptIterations'])   # Maximum Number of Generations
                opt.setOption('PrintOut', 0)    # Flag to Turn On Output to files (0-None, 1-Subset, 2-All)
                opt.setOption('xinit', 1)       # Use Initial Solution Flag (0 - random population, 1 - use given solution)
                opt.setOption('seed', sr.random())   # Random Number Seed 0..1 (0 - Auto based on time clock)
                #pCross_real    0.6     Probability of Crossover of Real Variable (0.6-1.0)
                opt.setOption('pMut_real', 0.5)   # Probablity of Mutation of Real Variables (1/nreal)
                #eta_c  10.0    # Distribution Index for Crossover (5-20) must be > 0
                #eta_m  20.0    # Distribution Index for Mutation (5-50) must be > 0
                #pCross_bin     0.0     # Probability of Crossover of Binary Variable (0.6-1.0)
                #pMut_real      0.0     # Probability of Mutation of Binary Variables (1/nbits)
                self.iter_max = self.config['globalOptSize']*self.config['globalOptIterations']
            elif self.config['globalSolver'] == 'ALPSO':
                if parallel:
                    opt = pyOpt.ALPSO(pll_type='SPM')  #augmented lagrange particle swarm optimization
                else:
                    opt = pyOpt.ALPSO()  #augmented lagrange particle swarm optimization
                opt.setOption('stopCriteria', 0)   # stop at max iters
                opt.setOption('dynInnerIter', 1)   # dynamic inner iter number
                opt.setOption('maxInnerIter', 5)
                opt.setOption('maxOuterIter', self.config['globalOptIterations'])
                opt.setOption('printInnerIters', 1)
                opt.setOption('printOuterIters', 1)
                opt.setOption('SwarmSize', self.config['globalOptSize'])
                opt.setOption('xinit', 1)
                opt.setOption('seed', sr.random()*self.mpi_size) #(self.mpi_rank+1)/self.mpi_size)
                #opt.setOption('vcrazy', 1e-2)
                #TODO: how to properly limit max number of function calls?
                # no. func calls = (SwarmSize * inner) * outer + SwarmSize
                self.iter_max = opt.getOption('SwarmSize') * opt.getOption('maxInnerIter') * \
                    opt.getOption('maxOuterIter') + opt.getOption('SwarmSize')
                self.iter_max = self.iter_max // self.mpi_size
            else:
                print("Solver {} not defined".format(self.config['globalSolver']))
                sys.exit(1)

            # run global optimization

            #try:
                #reuse history
            #    opt(opt_prob, store_hst=False, hot_start=True) #, xstart=initial)
            #except NameError:

            if self.config['verbose']:
                print('Running global optimization with {}'.format(self.config['globalSolver']))
            self.is_global = True
            opt(opt_prob, store_hst=False) #, xstart=initial)

            if self.mpi_rank == 0:
                print(opt_prob.solution(0))

            self.gather_solutions()

        ### pyOpt local
        if self.config['useLocalOptimization']:
            print("Runnning local gradient based solver")

            # TODO: run local optimization for e.g. the three last best results (global solutions
            # could be more or less optimal within their local minima)

            # after using global optimization, refine solution with gradient based method init
            # optimizer (more or less local)
            if self.config['localSolver'] == 'SLSQP':
                opt2 = pyOpt.SLSQP()   #sequential least squares
                opt2.setOption('MAXIT', self.config['localOptIterations'])
                if self.config['verbose']:
                    opt2.setOption('IPRINT', 0)
            elif self.config['localSolver'] == 'IPOPT':
                opt2 = pyOpt.IPOPT()
                opt2.setOption('linear_solver', 'ma57')  #mumps or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
                opt2.setOption('max_iter', self.config['localOptIterations'])
                if self.config['verbose']:
                    opt2.setOption('print_level', 4)  #0 none ... 5 max
                else:
                    opt2.setOption('print_level', 0)  #0 none ... 5 max
            elif self.config['localSolver'] == 'PSQP':
                opt2 = pyOpt.PSQP()
                opt2.setOption('MIT', self.config['localOptIterations'])  # max iterations
                #opt2.setOption('MFV', ??)  # max function evaluations
            elif self.config['localSolver'] == 'COBYLA':
                if parallel:
                    opt2 = pyOpt.COBYLA(pll_type='POA')
                else:
                    opt2 = pyOpt.COBYLA()
                opt2.setOption('MAXFUN', self.config['localOptIterations'])  # max iterations
                opt2.setOption('RHOBEG', 0.1)  # initial step size
                if self.config['verbose']:
                    opt2.setOption('IPRINT', 2)

            self.iter_max = self.local_iter_max

            # use best constrained solution from last run (might be better than what solver thinks)
            if len(self.last_best_sol) > 0:
                for i in range(len(opt_prob.getVarSet())):
                    opt_prob.getVar(i).value = self.last_best_sol[i]

            if self.config['verbose']:
                print('Runing local optimization with {}'.format(self.config['localSolver']))
            self.is_global = False
            if self.config['localSolver'] in ['COBYLA', 'CONMIN']:
                opt2(opt_prob, store_hst=False)
            else:
                if parallel:
                    opt2(opt_prob, sens_step=0.1, sens_mode='pgc', store_hst=False)
                else:
                    opt2(opt_prob, sens_step=0.1, store_hst=False)

            self.gather_solutions()

        if self.mpi_rank == 0:
            sol = opt_prob.solution(0)
            print(sol)
            #sol_vec = np.array([sol.getVar(x).value for x in range(0,len(sol.getVarSet()))])

            if len(self.last_best_sol) > 0:
                print("using last best constrained solution instead of given solver solution.")

                print("testing final solution")
                #self.iter_cnt = 0
                self.objectiveFunc(self.last_best_sol, test=True)
                print("\n")
                return self.last_best_sol
            else:
                print("No feasible solution found!")
                sys.exit(-1)
        else:
            # parallel sub-processes, close
            sys.exit(0)

