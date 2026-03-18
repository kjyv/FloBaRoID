from __future__ import annotations

import random
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from identification.model import Model
    from identifier import Identification

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-pastel")
except OSError:
    plt.style.use("seaborn-pastel")

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    parallel = nprocs > 1
except:
    parallel = False

from colorama import Fore
from idyntree import bindings as iDynTree

from identification.helpers import eulerAnglesToRotationMatrix


def plotter(config: dict, data: dict | None = None, filename: str | None = None) -> None:
    fig = plt.figure(1)
    fig.clear()
    if False:
        from itertools import permutations
        from random import sample

        # get a random color wheel
        Nlines = 200
        color_lvl = 8
        rgb = np.array(list(permutations(list(range(0, 256, color_lvl), 3)))) / 255.0
        colors = sample(rgb, Nlines)
        print(colors[0 : config["num_dofs"]])
    else:
        # set some fixed colors
        colors = [
            [0.97254902, 0.62745098, 0.40784314],
            [0.0627451, 0.53333333, 0.84705882],
            [0.15686275, 0.75294118, 0.37647059],
            [0.90980392, 0.37647059, 0.84705882],
            [0.84705882, 0.0, 0.1254902],
            [0.18823529, 0.31372549, 0.09411765],
            [0.50196078, 0.40784314, 0.15686275],
        ]

        from palettable.tableau import Tableau_10, Tableau_20

        colors += Tableau_10.mpl_colors[0:6] + Tableau_20.mpl_colors + Tableau_20.mpl_colors

    if not data:
        # reload measurements from this or last run (if run dry)
        if filename is None:
            return
        measurements = np.load(filename)
        Q = measurements["positions"]
        Qraw = measurements["positions_raw"]
        V = measurements["velocities"]
        Vraw = measurements["velocities_raw"]
        dV = measurements["accelerations"]
        Tau = measurements["torques"]
        TauRaw = measurements["torques_raw"]
        if "plot_targets" in config and config["plot_targets"]:
            Q_t = measurements["target_positions"]
            V_t = measurements["target_velocities"]
            dV_t = measurements["target_accelerations"]
        T = measurements["times"]
        num_samples = measurements["positions"].shape[0]
    else:
        Q = data["positions"]
        Qraw = data["positions"]
        Q_t = data["target_positions"]
        V = data["velocities"]
        Vraw = data["velocities"]
        V_t = data["target_velocities"]
        dV = data["accelerations"]
        dV_t = data["target_accelerations"]
        Tau = data["torques"]
        TauRaw = data["torques"]
        T = data["times"]
        num_samples = data["positions"].shape[0]

    if config["verbose"]:
        print(f"loaded {num_samples} measurement samples")

    if "plot_targets" in config and config["plot_targets"]:
        print("tracking error per joint:")
        for i in range(0, config["num_dofs"]):
            sse = np.sum((Q[:, i] - Q_t[:, i]) ** 2)
            print(f"joint {i}: {sse}")

    # print("histogram of time diffs")
    # dT = np.diff(T)
    # H, B = np.histogram(dT)
    # plt.hist(H, B)
    # late_msgs = (1 - float(np.sum(H)-np.sum(H[1:])) / float(np.sum(H))) * 100
    # print("bins: {}".format(B))
    # print("sums: {}".format(H))
    # print("({}% messages too late)\n".format(late_msgs))

    # what to plot (each tuple has a title and one or multiple data arrays)
    if "plot_targets" in config and config["plot_targets"]:
        datasets = [
            (
                [
                    Q_t,
                ],
                "Target Positions",
            ),
            (
                [
                    V_t,
                ],
                "Target Velocities",
            ),
            (
                [
                    dV_t,
                ],
                "Target Accelerations",
            ),
        ]
    else:  # plot measurements and raw data (from measurements file)
        if np.sum(Qraw - Q) != 0:
            datasets = [
                ([Q, Qraw], "Positions"),
                ([V, Vraw], "Velocities"),
                (
                    [
                        dV,
                    ],
                    "Accelerations",
                ),
                ([Tau, TauRaw], "Measured Torques"),
            ]
        else:
            datasets = [
                ([Q], "Positions"),
                ([V], "Velocities"),
                (
                    [
                        dV,
                    ],
                    "Accelerations",
                ),
                ([Tau], "Measured Torques"),
            ]

    d = 0
    cols = 2.0
    rows = round(len(datasets) / cols)
    for dat, title in datasets:
        plt.subplot(rows, cols, d + 1)
        plt.title(title)
        lines = list()
        labels = list()
        for d_i in range(0, len(dat)):
            if len(dat[d_i].shape) > 1:
                for i in range(0, config["num_dofs"]):
                    l = config["jointNames"][i] if d_i == 0 else ""  # only put joint names in the legend once
                    labels.append(l)
                    line = plt.plot(T, dat[d_i][:, i], color=colors[i], alpha=1 - (d_i / 2.0))
                    lines.append(line[0])
            else:
                # dat vector
                plt.plot(T, dat[d_i], label=title, color=colors[0], alpha=1 - (d_i / 2.0))
        d += 1
    leg = plt.figlegend(lines, labels, "upper right", fancybox=True, fontsize=10)
    leg.set_draggable(True)

    plt.subplots_adjust(hspace=2)
    plt.tight_layout()

    plt.show()


class Optimizer:
    """base class for different optimizers"""

    def __init__(
        self,
        config: dict[str, Any],
        idf: Identification,
        model: Model,
        simulation_func: Callable | None,
        world: str | None = None,
    ) -> None:

        self.config = config
        self.sim_func = simulation_func
        self.model = model
        self.idf = idf
        self.world = world

        self.num_dofs = self.config["num_dofs"]

        # optimization status vars
        self.last_best_f = np.inf
        self.last_best_sol = np.array([])

        self.iter_cnt = 0  # iteration counter
        self.last_g: list[float] | None = None  # last constraint values
        self.is_global = False
        self.local_iter_max = "(unknown)"

        # attributes declared here for type checking; set by subclasses
        self.trajectory: Any = None
        self.world_gravity: Any = None
        self.num_postures: int = 0
        self._var_names: list[str] = []

        # init parallel runs
        self.parallel = parallel
        if parallel:
            self.mpi_size = nprocs  # number of processes
            self.mpi_rank = comm.Get_rank()  # current process
            self.comm = comm
        else:
            self.mpi_size = 1
            self.mpi_rank = 0

        # init plotting progress
        if self.config["showOptimizationGraph"]:
            self.initGraph()

        # init link data
        self.link_cuboid_hulls: dict[str, list] = {}
        for i in range(self.model.num_links):
            link_name = self.model.linkNames[i]
            box, pos, rot = idf.urdfHelpers.getBoundingBox(
                input_urdf=self.model.urdf_file,
                old_com=self.model.xStdModel[i * 10 + 1 : i * 10 + 4] / self.model.xStdModel[i * 10],
                link_name=link_name,
                scaling=False,
            )
            self.link_cuboid_hulls[link_name] = [box, pos, rot]

        self.world = world
        self.world_links: list[str] = []
        if world:
            self.world_links = idf.urdfHelpers.getLinkNames(world)
            if self.config["verbose"]:
                print(f"World links: {self.world_links}")
            for link_name in self.world_links:
                box, pos, rot = idf.urdfHelpers.getBoundingBox(
                    input_urdf=world,
                    old_com=[0, 0, 0],
                    link_name=link_name,
                    scaling=False,
                )
                # make sure no name collision happens
                if link_name not in self.link_cuboid_hulls:
                    self.link_cuboid_hulls[link_name] = [box, pos, rot]
                else:
                    print(Fore.RED + f"Warning: link {link_name} declared in model and world file!" + Fore.RESET)

        self.world_boxes = {link: self.link_cuboid_hulls[link] for link in self.world_links}

    def testBounds(self, x: np.ndarray) -> bool:
        raise NotImplementedError

    def testConstraints(self, g: np.ndarray) -> bool:
        raise NotImplementedError

    def getLinkDistance(self, l0_name: str, l1_name: str, joint_q: np.ndarray) -> float:
        """get shortest distance from link with id l0 to link with id l1 for posture joint_q"""

        import fcl

        # get link rotation and position in world frame
        s = iDynTree.JointPosDoubleArray(self.num_dofs)
        ds = iDynTree.JointDOFsDoubleArray(self.num_dofs)
        for j in range(self.num_dofs):
            s.setVal(j, float(joint_q[j]))
        self.model.kinDyn.setRobotState(s, ds, self.model.gravity_vec)

        if l0_name in self.model.linkNames:  # if robot link
            t0 = self.model.kinDyn.getWorldTransform(l0_name)
            rot0 = t0.getRotation().toNumPy()
            pos0 = t0.getPosition().toNumPy()
            s0 = self.config["scaleCollisionHull"]
        else:  # if world link
            pos0 = self.link_cuboid_hulls[l0_name][1]
            rot0 = eulerAnglesToRotationMatrix(self.link_cuboid_hulls[l0_name][2])
            s0 = 1

        if l1_name in self.model.linkNames:  # if robot link
            t1 = self.model.kinDyn.getWorldTransform(l1_name)
            rot1 = t1.getRotation().toNumPy()
            pos1 = t1.getPosition().toNumPy()
            s1 = self.config["scaleCollisionHull"]
        else:  # if world link
            pos1 = self.link_cuboid_hulls[l1_name][1]
            rot1 = eulerAnglesToRotationMatrix(self.link_cuboid_hulls[l1_name][2])
            s1 = 1

        # TODO: use pos and rot of boxes for vals from geometry tags
        # self.link_cuboid_hulls[l0_name][1], [2]

        b = np.array(self.link_cuboid_hulls[l0_name][0]) * s0
        p = np.array(self.link_cuboid_hulls[l0_name][1])
        b0_center = 0.5 * np.array(
            [
                b[1][0] + b[0][0] + p[0],
                b[1][1] + b[0][1] + p[1],
                b[1][2] + b[0][2] + p[2],
            ]
        )
        b0 = fcl.Box(b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2])

        b = np.array(self.link_cuboid_hulls[l1_name][0]) * s1
        p = np.array(self.link_cuboid_hulls[l1_name][1])
        b1_center = 0.5 * np.array(
            [
                b[1][0] + b[0][0] + p[0],
                b[1][1] + b[0][1] + p[1],
                b[1][2] + b[0][2] + p[2],
            ]
        )
        b1 = fcl.Box(b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2])

        # move box to pos + box center pos (model has pos in link origin, box has zero at center)
        o0 = fcl.CollisionObject(b0, fcl.Transform(rot0, pos0 + b0_center))
        o1 = fcl.CollisionObject(b1, fcl.Transform(rot1, pos1 + b1_center))

        d_request = fcl.DistanceRequest(True)
        d_result = fcl.DistanceResult()
        distance = fcl.distance(o0, o1, d_request, d_result)

        if distance < 0:
            if self.config["verbose"] > 1:
                print(f"Collision of {l0_name} and {l1_name}")

            # get proper collision and depth since optimization should also know how much constraint is violated
            cr = fcl.CollisionRequest()
            cr.enable_contact = True
            c_result = fcl.CollisionResult()
            fcl.collide(o0, o1, cr, c_result)

            # sometimes no contact is found even though distance is less than 0?
            if len(c_result.contacts):
                distance = c_result.contacts[0].penetration_depth

        return distance

    def initVisualizer(self):
        if self.config["showModelVisualization"] and self.mpi_rank == 0:
            from visualizer import Visualizer

            self.visualizer = Visualizer(self.config)
            self.visualizer.loadMeshes(self.model.urdf_file, self.model.linkNames, self.idf.urdfHelpers)

            # set draw method for visualizer. This taps into local variables here, a bit unclean...
            def draw_model():
                if self.trajectory and self.visualizer.trajectory:
                    # get data of trajectory
                    self.visualizer.trajectory.setTime(self.visualizer.display_index / self.visualizer.fps)
                    q0 = [self.visualizer.trajectory.getAngle(d) for d in range(self.num_dofs)]
                else:
                    p_id = self.visualizer.display_index
                    angles = self.visualizer.angles
                    if angles is None:
                        return
                    q0 = angles[p_id * self.num_dofs : (p_id + 1) * self.num_dofs]

                q = iDynTree.VectorDynSize.FromPython(q0)
                dq = iDynTree.VectorDynSize.FromPython([0.0] * self.num_dofs)
                self.model.kinDyn.setRobotState(q, dq, dq, self.world_gravity)
                self.visualizer.addIDynTreeModel(
                    self.model.kinDyn,
                    self.link_cuboid_hulls,
                    self.model.linkNames,
                    self.config["ignoreLinksForCollision"],
                )
                if self.world:
                    self.visualizer.addWorld(self.world_boxes)
                self.visualizer.updateLabels()

            self.visualizer.event_callback = draw_model

    def showVisualizerAngles(self, x):
        """show visualizer for current joint angles x"""
        if self.config["showModelVisualization"] and self.mpi_rank == 0:  # and c:
            self.visualizer.display_max = self.num_postures
            self.visualizer.angles = x
            if self.visualizer.event_callback:
                self.visualizer.event_callback()
            self.visualizer.run()

    def showVisualizerTrajectory(self, t):
        """show visualizer for joint trajectory t"""
        if self.config["showModelVisualization"] and self.mpi_rank == 0:  # and c:
            self.visualizer.setModelTrajectory(t)
            self.config["excitationFrequency"]
            self.visualizer.display_max = t.getPeriodLength() * self.visualizer.fps  # length of trajectory
            self.visualizer.trajectory = t
            self.visualizer.playable = True
            if self.visualizer.event_callback:
                self.visualizer.event_callback()
            self.visualizer.run()

    def objectiveFunc(self, x: np.ndarray, test: bool = False) -> tuple[float, np.ndarray, bool]:
        """calculate objective function and return objective function value f, constraint values g
        and a fail flag"""
        raise NotImplementedError

    def initGraph(self):
        if self.mpi_rank > 0:
            return
        # init graphing of objective function value
        self.fig = plt.figure(0)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        plt.ion()
        self.xar: list[int] = []  # x value, i.e. iteration count
        self.yar: list[float] = []  # y value, i.e. obj func value
        self.x_constr: list[bool] = []  # within constraints or not (feasible)
        self.ax1.plot(self.xar, self.yar)
        self.ax1.set_xlabel("Function evaluation #")
        self.ax1.set_ylabel("Objective function value")

        self.updateGraphEveryVals = 5

    def updateGraph(self):
        if self.mpi_rank > 0:
            return
        # draw all optimization steps, mark the ones that are within constraints
        # if (self.iter_cnt % self.updateGraphEveryVals) == 0:
        color = "g"
        line = self.ax1.plot(
            self.xar,
            self.yar,
            marker=".",
            markeredgecolor=color,
            markerfacecolor=color,
            color="0.75",
        )
        markers = np.where(self.x_constr)[0]
        line[0].set_markevery(list(markers))

        if self.iter_cnt == 1:
            plt.show(block=False)

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

        x0 = np.asarray(x, dtype=float)
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
            if self.config["verbose"]:
                print("Collecting best solutions from processes")
            send_obj = [self.last_best_f, self.last_best_sol, self.mpi_rank]
            received_objs = self.comm.gather(send_obj, root=0)

            # receive solutions from other instances
            if self.mpi_rank == 0:
                for proc in range(0, self.mpi_size):
                    other_best_f, other_best_sol, rank = received_objs[proc]

                    if other_best_f < self.last_best_f:
                        print(f"received better solution from {rank}")
                        self.last_best_f = other_best_f
                        self.last_best_sol = other_best_sol

    def runOptimizer(self, opt_prob: Any) -> np.ndarray:
        """call global followed by local optimizer, return solution"""

        from pyoptsparse import ALPSO, IPOPT, NSGA2, PSQP, SLSQP

        if self.config["useGlobalOptimization"]:
            ### global optimization
            sr = random.SystemRandom()
            if self.config["globalSolver"] == "NSGA2":
                opt = NSGA2()  # genetic algorithm
                if parallel:
                    opt.setOption("parallelType", "POA")
                if self.config["globalOptSize"] % 4:
                    raise OSError("globalOptSize needs to be a multiple of 4 for NSGA2")
                opt.setOption("PopSize", self.config["globalOptSize"])  # Population Size (a Multiple of 4)
                opt.setOption("maxGen", self.config["globalOptIterations"])  # Maximum Number of Generations
                opt.setOption("PrintOut", 0)  # Flag to Turn On Output to files (0-None, 1-Subset, 2-All)
                opt.setOption("xinit", 1)  # Use Initial Solution Flag (0 - random population, 1 - use given solution)
                opt.setOption("seed", sr.randint(1, 2**31))  # Random Number Seed
                # pCross_real    0.6     Probability of Crossover of Real Variable (0.6-1.0)
                opt.setOption("pMut_real", 0.5)  # Probablity of Mutation of Real Variables (1/nreal)
                # eta_c  10.0    # Distribution Index for Crossover (5-20) must be > 0
                # eta_m  20.0    # Distribution Index for Mutation (5-50) must be > 0
                # pCross_bin     0.0     # Probability of Crossover of Binary Variable (0.6-1.0)
                # pMut_real      0.0     # Probability of Mutation of Binary Variables (1/nbits)
                self.iter_max = self.config["globalOptSize"] * self.config["globalOptIterations"]
            elif self.config["globalSolver"] == "ALPSO":
                opt = ALPSO()  # augmented lagrange particle swarm optimization
                if parallel:
                    opt.setOption("parallelType", "SPM")
                opt.setOption("stopCriteria", 0)  # stop at max iters
                opt.setOption("dynInnerIter", 1)  # dynamic inner iter number
                opt.setOption("maxInnerIter", 5)
                opt.setOption("maxOuterIter", self.config["globalOptIterations"])
                opt.setOption("printInnerIters", 1)
                opt.setOption("printOuterIters", 1)
                opt.setOption("SwarmSize", self.config["globalOptSize"])
                opt.setOption("xinit", 1)
                opt.setOption("seed", sr.randint(1, 2**31))
                # opt.setOption('vcrazy', 1e-2)
                # TODO: how to properly limit max number of function calls?
                # no. func calls = (SwarmSize * inner) * outer + SwarmSize
                self.iter_max = opt.getOption("SwarmSize") * opt.getOption("maxInnerIter") * opt.getOption(
                    "maxOuterIter"
                ) + opt.getOption("SwarmSize")
                self.iter_max = self.iter_max // self.mpi_size
            else:
                print("Solver {} not defined".format(self.config["globalSolver"]))
                sys.exit(1)

            # run global optimization

            if self.config["verbose"]:
                print("Running global optimization with {}".format(self.config["globalSolver"]))
            self.is_global = True
            sol = opt(opt_prob, storeHistory=False)

            if self.mpi_rank == 0:
                print(sol)

            self.gather_solutions()

        ### local optimization
        if self.config["useLocalOptimization"]:
            print("Runnning local gradient based solver")

            # TODO: run local optimization for e.g. the three last best results (global solutions
            # could be more or less optimal within their local minima)

            # after using global optimization, refine solution with gradient based method init
            # optimizer (more or less local)
            if self.config["localSolver"] == "SLSQP":
                opt2 = SLSQP()  # sequential least squares
                opt2.setOption("MAXIT", self.config["localOptIterations"])
                if self.config["verbose"]:
                    opt2.setOption("IPRINT", 0)
            elif self.config["localSolver"] == "IPOPT":
                opt2 = IPOPT()
                opt2.setOption("linear_solver", "ma57")  # mumps or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
                opt2.setOption("max_iter", self.config["localOptIterations"])
                if self.config["verbose"]:
                    opt2.setOption("print_level", 4)  # 0 none ... 5 max
                else:
                    opt2.setOption("print_level", 0)  # 0 none ... 5 max
            elif self.config["localSolver"] == "PSQP":
                opt2 = PSQP()
                opt2.setOption("MIT", self.config["localOptIterations"])  # max iterations
                # opt2.setOption('MFV', ??)  # max function evaluations

            self.iter_max = self.local_iter_max

            # use best constrained solution from last run
            if len(self.last_best_sol) > 0:
                dvs = {name: self.last_best_sol[i] for i, name in enumerate(self._var_names)}
                opt_prob.setDVs(dvs)

            if self.config["verbose"]:
                print("Runing local optimization with {}".format(self.config["localSolver"]))
            self.is_global = False
            sol = opt2(opt_prob, sens="FD", sensStep=0.1, storeHistory=False)

            self.gather_solutions()

        if self.mpi_rank == 0:
            print(sol)
            # sol_vec = np.array([sol.getVar(x).value for x in range(0,len(sol.getVarSet()))])

            if len(self.last_best_sol) > 0:
                print("using last best constrained solution instead of given solver solution.")

                print("testing final solution")
                # self.iter_cnt = 0
                self.objectiveFunc(self.last_best_sol, test=True)
                print("\n")
                return self.last_best_sol
            else:
                print("No feasible solution found!")
                sys.exit(-1)
        else:
            # parallel sub-processes, close
            sys.exit(0)
