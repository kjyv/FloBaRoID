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

import pyOpt
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    parallel = nprocs > 1
except:
    parallel = False

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
    def __init__(self, config, model, simulation_func):
        # type: (Dict, Model, Callable[[Dict, Trajectory, Model, np._ArrayLike], Tuple[Dict, Data]]) -> None
        self.config = config
        self.sim_func = simulation_func
        self.model = model

        self.last_best_f = np.inf
        self.last_best_sol = np.array([])

        self.iter_cnt = 0   # iteration counter
        self.last_g = None  # type: List[float]    # last constraint values

        self.parallel = parallel
        if parallel:
            self.mpi_size = nprocs   # number of processes
            self.mpi_rank = comm.Get_rank()  # current process
            self.comm = comm
        else:
            self.mpi_size = 1
            self.mpi_rank = 0

        if self.config['showOptimizationGraph']:
            self.initGraph()

        self.local_iter_max = "(unknown)"
        self.is_global = False

    def testBounds(self, x):
        # type: (np._ArrayLike) -> bool
        raise NotImplementedError

    def testConstraints(self, g):
        # type: (np._ArrayLike) -> bool
        raise NotImplementedError

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

        initial = [v.value for v in list(opt_prob.getVarSet().values())]

        if self.config['useGlobalOptimization']:
            ### optimize using pyOpt (global)
            sr = random.SystemRandom()
            if self.config['globalSolver'] == 'NSGA2':
                if parallel:
                    opt = pyOpt.NSGA2(pll_type='POA') # genetic algorithm
                else:
                    opt = pyOpt.NSGA2()
                opt.setOption('PopSize', 100)   # Population Size (a Multiple of 4)
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
                self.iter_max = '(unknown)'
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
                opt.setOption('SwarmSize', 200)
                opt.setOption('xinit', 1)
                opt.setOption('seed', sr.random()*self.mpi_size) #(self.mpi_rank+1)/self.mpi_size)
                #opt.setOption('vcrazy', 1e-2)
                #TODO: how to properly limit max number of function calls?
                # no. func calls = (SwarmSize * inner) * outer + SwarmSize
                self.iter_max = opt.getOption('SwarmSize') * opt.getOption('maxInnerIter') * \
                    opt.getOption('maxOuterIter') + opt.getOption('SwarmSize')
                self.iter_max = self.iter_max // self.mpi_size

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

            #TODO: run local optimization for e.g. the three last best results (global solutions could be more or less optimal
            # within their local minima)

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

            # TODO: amount of function calls depends on amount of variables and iterations to
            # approximate gradient ('iterations' are probably the actual steps along the gradient). How
            # to get proper no. of expected func calls? (one call per dimension for each iteration?)
            self.iter_max = self.local_iter_max

            #use best constrained solution from last run (might be better than what solver thinks)
            if len(self.last_best_sol) > 0:
                for i in range(len(opt_prob.getVarSet())):
                    opt_prob.getVar(i).value = self.last_best_sol[i]

            if self.config['verbose']:
                print('Runing local optimization with {}'.format(self.config['localSolver']))
            self.is_global = False
            if parallel:
                opt2(opt_prob, sens_step=0.1, sens_mode='pgc', store_hst=False)
            else:
                opt2(opt_prob, sens_step=0.1, store_hst=False)

        self.gather_solutions()

        if self.mpi_rank == 0:
            sol = opt_prob.solution(0)
            print(sol)
            #sol_vec = np.array([sol.getVar(x).value for x in range(0,len(sol.getVarSet()))])

            if self.last_best_sol.size > 0:
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

