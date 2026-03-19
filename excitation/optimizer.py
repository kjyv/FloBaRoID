from __future__ import annotations

import os
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

import fcl
from colorama import Fore
from idyntree import bindings as iDynTree

from excitation.capsule import Capsule, capsule_distance, capsule_distance_and_gradient, fit_capsules_from_urdf
from identification.helpers import eulerAnglesToRotationMatrix


def _optuna_worker(
    study_name: str,
    storage_url: str,
    config: dict,
    var_info: list[tuple[str, float, float, float]],
    n_worker_trials: int,
    sampler_name: str = "TPE",
) -> None:
    """Worker process for parallel Optuna optimization.

    Each worker creates its own Model and optimizer instance to avoid
    sharing non-picklable iDynTree objects across processes.
    """
    # prevent matplotlib from opening windows in worker processes
    import matplotlib

    matplotlib.use("Agg")

    import optuna as _optuna

    from excitation.trajectoryGenerator import simulateTrajectory
    from excitation.trajectoryOptimizer import TrajectoryOptimizer
    from identification.model import Model
    from identifier import Identification

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    from idyntree import bindings as iDynTree

    config["jointNames"] = iDynTree.StringVector([])
    iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"])
    config["num_dofs"] = len(config["jointNames"])
    config["showOptimizationGraph"] = 0
    config["showModelVisualization"] = 0
    config["verbose"] = 0

    model = Model(config, config["urdf"])
    idf = Identification(config, config["urdf"], None, None, None, None)
    worker_opt = TrajectoryOptimizer(config, idf, model, simulateTrajectory)
    worker_opt.is_global = True

    def worker_objective(trial: _optuna.Trial) -> float:
        """Objective for a single Optuna trial in a worker process."""
        x = [trial.suggest_float(name, lo, hi) for name, lo, hi, _ in var_info]
        f, g, _fail = worker_opt.objectiveFunc(np.array(x))
        # store constraints as trial user attributes for the constraints_func
        c_s = worker_opt.num_constraints - worker_opt.num_coll_constraints
        violations = [float(g[i]) for i in range(c_s)]
        violations += [float(-g[i]) for i in range(c_s, worker_opt.num_constraints)]
        trial.set_user_attr("constraints", violations)
        return f

    # create a constraint-aware sampler with constant_liar for cross-process coordination.
    # constant_liar makes TPE assume running trials (from other workers) return a pessimistic
    # value, so each worker explores different regions of the search space.
    n_con = worker_opt.num_constraints

    def worker_constraints_func(trial: _optuna.trial.FrozenTrial) -> list[float]:
        return trial.user_attrs.get("constraints", [10.0] * n_con)

    if sampler_name == "NSGA2":
        worker_sampler: Any = _optuna.samplers.NSGAIISampler(constraints_func=worker_constraints_func)
    else:
        worker_sampler = _optuna.samplers.TPESampler(
            constraints_func=worker_constraints_func,
            constant_liar=True,
        )

    w_study = _optuna.load_study(study_name=study_name, storage=storage_url, sampler=worker_sampler)

    # each worker keeps running until the study has enough total trials
    def _stop_callback(study: _optuna.Study, trial: _optuna.trial.FrozenTrial) -> None:
        if len(study.trials) >= n_worker_trials:
            study.stop()

    w_study.optimize(worker_objective, callbacks=[_stop_callback], catch=(ValueError,))


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
        self.last_g: np.ndarray | list[float] | None = None  # last constraint values
        self.is_global = False
        self.local_iter_max = 0

        # attributes declared here for type checking; set by subclasses
        self.trajectory: Any = None
        self.world_gravity: Any = None
        self.num_postures: int = 0
        self._var_names: list[str] = []

        # trajectory optimizer attributes (set by TrajectoryOptimizer/PostureOptimizer)
        self.wf_min: float = 0.0
        self.wf_max: float = 1.0
        self.wf_init: float = 0.5
        self.qmin: list[float] | np.ndarray = []
        self.qmax: list[float] | np.ndarray = []
        self.qinit: list[float] | np.ndarray = []
        self.nf: list[int] = []
        self.amin: float = -1.0
        self.amax: float = 1.0
        self.bmin: float = -1.0
        self.bmax: float = 1.0
        self.ainit: list[np.ndarray] = []
        self.binit: list[np.ndarray] = []
        self.num_constraints: int = 0
        self.num_coll_constraints: int = 0

        self.mpi_rank = 0  # legacy, kept for compatibility with subclasses

        # init plotting progress
        if self.config["showOptimizationGraph"]:
            self.initGraph()

        # init link data
        self.link_cuboid_hulls: dict[str, list] = {}
        for i in range(self.model.num_links):
            link_name = self.model.linkNames[i]
            # skip links without any visual geometry (e.g. sensor frames)
            if not idf.urdfHelpers.hasVisualGeometry(self.model.urdf_file, link_name):
                continue
            box, pos, rot = idf.urdfHelpers.getBoundingBox(
                input_urdf=self.model.urdf_file,
                old_com=self.model.xStdModel[i * 10 + 1 : i * 10 + 4] / self.model.xStdModel[i * 10]
                if self.model.xStdModel[i * 10] != 0
                else [0.0, 0.0, 0.0],
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

        # fit capsule collision primitives (for collision optimization and/or visualization)
        self._capsules: dict[str, Capsule] = {}
        if self.config.get("useCapsuleCollision", False) or self.config.get("showModelVisualization", False):
            self._capsules = fit_capsules_from_urdf(
                self.model.urdf_file,
                self.model.linkNames,
                idf.urdfHelpers,
                radius_scale=self.config.get("scaleCapsuleRadius", 1.0),
            )
            if self.config.get("verbose", 0) and self._capsules:
                print(f"Capsule collision: fitted {len(self._capsules)} capsules")

            # warn if capsules collide at zero pose for non-neighbor pairs
            if self.config.get("useCapsuleCollision", False) and self._capsules:
                from excitation.capsule import find_colliding_links_capsule

                zero_colliding = find_colliding_links_capsule(
                    self._capsules,
                    self.model.kinDyn,
                    self.model.linkNames,
                    ignore_links=set(self.config.get("ignoreLinksForCollision", [])),
                    ignore_pairs=self.config.get("ignoreLinkPairsForCollision", []),
                    neighbors=idf.urdfHelpers.getNeighbors(self.model.idyn_model),
                    max_kin_distance=self.config.get("collisionMaxKinematicDistance", 0),
                )
                if zero_colliding:
                    print(
                        Fore.YELLOW
                        + f"Warning: capsule collision infeasible at zero pose for: {zero_colliding}. "
                        + "Capsule approximation may be too conservative for this robot. "
                        + "Consider disabling useCapsuleCollision or increasing collisionMaxKinematicDistance."
                        + Fore.RESET
                    )

    def testBounds(self, x: np.ndarray) -> bool:
        raise NotImplementedError

    def testConstraints(self, g: np.ndarray) -> bool:
        raise NotImplementedError

    def _getLinkCollisionGeometry(self, link_name: str) -> tuple[Any, np.ndarray]:
        """Get or cache the FCL collision geometry and center offset for a link.

        Tries to load the collision mesh as a triangle BVH for accurate collision
        checking. Falls back to an axis-aligned bounding box if no mesh is available.
        """
        if not hasattr(self, "_geom_cache"):
            self._geom_cache: dict[str, tuple[Any, np.ndarray]] = {}
        if link_name not in self._geom_cache:
            # try loading collision mesh (unless config says to use bounding boxes)
            use_meshes = self.config.get("useCollisionMeshes", 1)
            mesh_path = None
            if use_meshes:
                mesh_path = self.idf.urdfHelpers.getCollisionMeshPath(self.model.urdf_file, link_name)
                if mesh_path is None or not os.path.exists(mesh_path):
                    mesh_path = self.idf.urdfHelpers.getMeshPath(self.model.urdf_file, link_name)

            if mesh_path is not None and os.path.exists(mesh_path):
                import trimesh

                mesh = trimesh.load_mesh(mesh_path)
                # apply per-axis scaling from URDF
                scale_parts = self.idf.urdfHelpers.mesh_scaling.split()
                scale = np.array([float(s) for s in scale_parts])
                verts = np.array(mesh.vertices) * scale
                faces = np.array(mesh.faces)

                # fix face winding if any scale axis is negative (mirrored mesh)
                if np.prod(scale) < 0:
                    faces = faces[:, ::-1]

                p = np.array(self.link_cuboid_hulls[link_name][1])

                if self.config.get("useConvexHullCollision", False):
                    # convex hull: GJK convex-convex is faster than BVH mesh distance,
                    # and is a conservative approximation
                    hull = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
                    hull_verts = np.array(hull.vertices, dtype=np.float64)
                    hull_faces = np.array(hull.faces, dtype=np.int32)

                    # fcl.Convex expects flat face array with vertex count prefix per face
                    flat_faces = np.empty(len(hull_faces) * 4, dtype=np.int32)
                    for fi in range(len(hull_faces)):
                        flat_faces[fi * 4] = 3  # triangle
                        flat_faces[fi * 4 + 1 : fi * 4 + 4] = hull_faces[fi]

                    convex = fcl.Convex(hull_verts, len(hull_faces), flat_faces)
                    self._geom_cache[link_name] = (convex, p)
                else:
                    # use the mesh directly (BVH triangle mesh) — tighter fit
                    bvh = fcl.BVHModel()
                    bvh.beginModel(len(faces), len(verts))
                    bvh.addSubModel(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32))
                    bvh.endModel()
                    self._geom_cache[link_name] = (bvh, p)
            else:
                # fallback to bounding box
                s = self.config["scaleCollisionHull"] if link_name in self.model.linkNames else 1
                b = np.array(self.link_cuboid_hulls[link_name][0]) * s
                p = np.array(self.link_cuboid_hulls[link_name][1])
                center = 0.5 * (b[0] + b[1]) + p
                box = fcl.Box(*(b[1] - b[0]))
                self._geom_cache[link_name] = (box, center)
        return self._geom_cache[link_name]

    def _getLinkTransform(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get rotation and position for a link. For robot links, uses current kinDyn state
        (set via setCollisionRobotState). For world links, returns static transform.
        Results are cached per sample via _transform_cache (cleared each sample)."""
        if link_name in self._transform_cache:
            return self._transform_cache[link_name]
        if link_name in self.model.linkNames:
            t = self.model.kinDyn.getWorldTransform(link_name)
            rot = t.getRotation().toNumPy()
            pos = t.getPosition().toNumPy()
        else:
            pos = self.link_cuboid_hulls[link_name][1]
            rot = eulerAnglesToRotationMatrix(self.link_cuboid_hulls[link_name][2])
        self._transform_cache[link_name] = (rot, pos)
        return rot, pos

    def setCollisionRobotState(self, joint_q: np.ndarray) -> None:
        """Set iDynTree robot state for collision checking (call once per sample, not per pair)."""
        if not hasattr(self, "_coll_s"):
            self._coll_s = iDynTree.JointPosDoubleArray(self.num_dofs)
            self._coll_ds = iDynTree.JointDOFsDoubleArray(self.num_dofs)
        for j in range(self.num_dofs):
            self._coll_s.setVal(j, float(joint_q[j]))
        self.model.kinDyn.setRobotState(self._coll_s, self._coll_ds, self.model.gravity_vec)
        # clear transform cache for new configuration
        self._transform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def getLinkDistance(self, l0_name: str, l1_name: str, joint_q: np.ndarray) -> float:
        """get shortest distance from link with id l0 to link with id l1 for posture joint_q"""

        rot0, pos0 = self._getLinkTransform(l0_name)
        rot1, pos1 = self._getLinkTransform(l1_name)
        geom0, offset0 = self._getLinkCollisionGeometry(l0_name)
        geom1, offset1 = self._getLinkCollisionGeometry(l1_name)

        # place geometry at link world position + mesh/box origin offset
        o0 = fcl.CollisionObject(geom0, fcl.Transform(rot0, pos0 + offset0))
        o1 = fcl.CollisionObject(geom1, fcl.Transform(rot1, pos1 + offset1))

        # fcl.distance returns negative for penetration with Box and Convex geometry.
        # (BVH would return 0, but we use Convex hulls instead.)
        distance = fcl.distance(o0, o1, fcl.DistanceRequest(True), fcl.DistanceResult())

        return distance

    def getCapsuleDistance(self, l0_name: str, l1_name: str) -> float:
        """Get capsule-based distance between two links (state must be set via setCollisionRobotState)."""
        rot0, pos0 = self._getLinkTransform(l0_name)
        rot1, pos1 = self._getLinkTransform(l1_name)
        dist, _, _, _, _, _, _ = capsule_distance(
            self._capsules[l0_name], self._capsules[l1_name], rot0, pos0, rot1, pos1
        )
        return dist

    def getCapsuleDistanceAndGradient(self, l0_name: str, l1_name: str) -> tuple[float, np.ndarray]:
        """Get capsule distance and analytical d(distance)/d(q) for two links.

        Returns (distance, gradient) where gradient has shape (num_dofs,).
        State must be set via setCollisionRobotState beforehand.
        """
        return capsule_distance_and_gradient(
            self._capsules[l0_name],
            self._capsules[l1_name],
            self.model.kinDyn,
            self.num_dofs,
            is_floating=bool(self.config.get("floatingBase", False)),
        )

    def initVisualizer(self):
        if self.config["showModelVisualization"]:
            from visualizer import Visualizer

            self.visualizer = Visualizer(self.config)
            self.visualizer.loadMeshes(
                self.model.urdf_file,
                self.model.linkNames,
                self.idf.urdfHelpers,
                use_convex_hull=self.config.get("useConvexHullCollision", False),
            )
            if self._capsules:
                self.visualizer.loadCapsules(self._capsules)

            # set draw method for visualizer. This taps into local variables here, a bit unclean...
            def draw_model():
                if self.trajectory and self.visualizer.trajectory:
                    # get data of trajectory
                    self.visualizer.trajectory.setTime(self.visualizer.display_index / self.visualizer.playback_rate)
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
        if self.config["showModelVisualization"]:  # and c:
            self.visualizer.display_max = self.num_postures
            self.visualizer.angles = x
            if self.visualizer.event_callback:
                self.visualizer.event_callback()
            self.visualizer.run()

    def showVisualizerTrajectory(self, t):
        """show visualizer for joint trajectory t"""
        if self.config["showModelVisualization"]:  # and c:
            self.visualizer.setModelTrajectory(t)
            self.visualizer.playback_rate = self.config["excitationFrequency"]
            self.visualizer.display_max = t.getPeriodLength() * self.visualizer.playback_rate  # length of trajectory
            self.visualizer.trajectory = t
            self.visualizer.playable = True
            if self.visualizer.event_callback:
                self.visualizer.event_callback()
            self.visualizer.run()

    def objectiveFunc(self, x: np.ndarray, test: bool = False) -> tuple[float, np.ndarray, bool]:
        """calculate objective function and return objective function value f, constraint values g
        and a fail flag"""
        raise NotImplementedError

    def _objFuncWrapper(self, xdict: dict) -> tuple[dict, bool]:
        """Wrap objectiveFunc for pyOptSparse dict-based API."""
        raise NotImplementedError

    def _sensitivityWrapper(self, xdict: dict, funcs: dict) -> tuple[dict, bool]:
        """Compute analytical gradients for pyOptSparse. Overridden in subclasses."""
        raise NotImplementedError

    def addVarsAndConstraints(self, opt_prob: Any, initial_values: np.ndarray | None = None) -> None:
        """Add variables and constraints to the optimization problem."""
        raise NotImplementedError

    def initGraph(self):

        # init graphing of objective function value
        self.fig = plt.figure(0)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        plt.ion()
        self.xar: list[int] = []  # x value, i.e. iteration count
        self.yar: list[float] = []  # y value, i.e. obj func value
        self.x_constr: list[bool] = []  # within constraints or not (feasible)
        # create a single line object that we update (instead of adding new lines)
        (self._graph_line,) = self.ax1.plot([], [], marker=".", markeredgecolor="g", markerfacecolor="g", color="0.75")
        self.ax1.set_xlabel("Function evaluation #")
        self.ax1.set_ylabel("Objective function value")
        self._graph_shown = False

    def updateGraph(self):

        # update the single line with current data
        self._graph_line.set_data(self.xar, self.yar)
        markers = np.where(self.x_constr)[0]
        self._graph_line.set_markevery(list(markers))
        self.ax1.relim()
        self.ax1.autoscale_view()

        if not self._graph_shown:
            plt.show(block=False)
            self._graph_shown = True

        # update without stealing window focus (plt.pause steals focus on macOS)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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

    def gather_solutions(self) -> None:
        """Placeholder for solution gathering (previously used MPI)."""

    def _runOptuna(self) -> None:
        """Run Optuna TPE or NSGA2 global optimization.

        Optuna learns from previous evaluations which parameter regions are
        promising, handles infeasible trials gracefully, and doesn't need
        gradients. Much better than random swarm for constrained problems.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = self.config["globalOptSize"] * self.config["globalOptIterations"]
        sampler_name = self.config.get("optunaSampler", "TPE")

        # build variable bounds for optuna
        var_info: list[tuple[str, float, float, float]] = []  # (name, lower, upper, init)
        var_info.append(("wf", self.wf_min, self.wf_max, self.wf_init))
        for i in range(self.num_dofs):
            var_info.append((f"q_{i}", float(self.qmin[i]), float(self.qmax[i]), float(self.qinit[i])))
        for i in range(self.num_dofs):
            for j in range(self.nf[i]):
                var_info.append((f"a{i}_{j}", self.amin, self.amax, float(self.ainit[i][j])))
        for i in range(self.num_dofs):
            for j in range(self.nf[i]):
                var_info.append((f"b{i}_{j}", self.bmin, self.bmax, float(self.binit[i][j])))

        # store constraint values from each trial for the constraints callback
        trial_constraints: dict[int, list[float]] = {}
        num_constraints = self.num_constraints
        num_coll_constraints = self.num_coll_constraints

        def objective(trial: optuna.Trial) -> float:
            x = []
            for name, lo, hi, init in var_info:
                x.append(trial.suggest_float(name, lo, hi))
            x_arr = np.array(x)
            f, g, fail = self.objectiveFunc(x_arr)

            # store constraint values for this trial (g <= 0 means feasible for
            # joint/velocity/torque constraints, g > 0 means feasible for collision)
            c_s = num_constraints - num_coll_constraints
            # convert all to "violation <= 0 is feasible" convention for optuna
            violations = []
            for i in range(c_s):
                violations.append(float(g[i]))  # already <= 0 for feasible
            for i in range(c_s, num_constraints):
                violations.append(float(-g[i]))  # collision: flip sign (g > 0 → -g < 0 = feasible)
            trial_constraints[trial.number] = violations

            return f

        def constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
            """Return constraint values for a completed trial (<=0 means feasible)."""
            return trial_constraints.get(trial.number, [10.0] * num_constraints)

        # create sampler with constraint awareness
        sampler: Any
        if sampler_name == "NSGA2":
            sampler = optuna.samplers.NSGAIISampler(seed=random.randint(0, 2**31), constraints_func=constraints_func)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=random.randint(0, 2**31),
                constraints_func=constraints_func,
                constant_liar=True,  # parallel: assume running trials return worst value
            )

        # enqueue the initial solution as the first trial
        init_params = {info[0]: info[3] for info in var_info}
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.enqueue_trial(init_params)

        cpu_count = os.cpu_count() or 1
        default_jobs = max(1, cpu_count - 2)
        n_jobs = self.config.get("globalOptJobs", default_jobs)
        print(f"Running Optuna ({sampler_name}) with {n_trials} trials ({n_jobs} jobs)...")

        if n_jobs > 1:
            # multiprocess: use SQLite storage so worker processes share the study.
            # each worker creates its own Model/KinDyn/collision objects.
            import multiprocessing
            import tempfile

            storage_path = tempfile.mktemp(suffix=".db")
            storage = f"sqlite:///{storage_path}"
            study = optuna.create_study(
                study_name="trajectory_opt",
                direction="minimize",
                sampler=sampler,
                storage=storage,
            )
            study.enqueue_trial(init_params)

            # each worker runs until the study has enough total trials
            # (workers check the shared DB and stop when n_trials is reached)
            config_copy = {k: v for k, v in self.config.items() if k != "jointNames"}
            processes = []
            for i in range(n_jobs):
                p = multiprocessing.Process(
                    target=_optuna_worker,
                    args=("trajectory_opt", storage, config_copy, var_info, n_trials, sampler_name),
                    name=f"optuna_worker_{i}",
                )
                p.start()
                print(f"  started worker {i} (pid={p.pid})")
                processes.append(p)
            # poll workers and update graph while they run
            import time

            if self.config["showOptimizationGraph"]:
                plt.show(block=False)

            while any(p.is_alive() for p in processes):
                time.sleep(2)
                if self.config["showOptimizationGraph"]:
                    # read completed trials from shared storage and update graph
                    tmp_study = optuna.load_study(study_name="trajectory_opt", storage=storage)
                    self.xar = []
                    self.yar = []
                    self.x_constr = []
                    for t in tmp_study.trials:
                        if t.value is not None:
                            self.xar.append(t.number)
                            self.yar.append(t.value)
                            constraints = t.user_attrs.get("constraints", None)
                            feasible = constraints is not None and all(v <= 0.01 for v in constraints)
                            self.x_constr.append(feasible)
                    if self.xar:
                        self.iter_cnt = max(self.xar)
                        self.updateGraph()
                n_done = sum(not p.is_alive() for p in processes)
                n_trials_done = len(optuna.load_study(study_name="trajectory_opt", storage=storage).trials)
                print(
                    f"\r  {n_trials_done}/{n_trials} trials, {n_jobs - n_done}/{n_jobs} workers active",
                    end="",
                    flush=True,
                )
            print()

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    print(f"  worker {p.name} exited with code {p.exitcode}")

            # reload study results from shared storage
            study = optuna.load_study(study_name="trajectory_opt", storage=storage)
            os.remove(storage_path)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # extract best feasible solution
        # constraints are stored either in local dict (single process) or trial user_attrs (multiprocess)
        def _get_constraints(t: Any) -> list[float]:
            if t.number in trial_constraints:
                return trial_constraints[t.number]
            return t.user_attrs.get("constraints", [10.0] * num_constraints)

        feasible_trials = [
            t for t in study.trials if t.value is not None and all(v <= 0.01 for v in _get_constraints(t))
        ]
        # sort by objective value (best first) and verify with dense collision check
        feasible_trials.sort(key=lambda t: t.value)  # type: ignore[arg-type,return-value]

        verified = False
        for trial in feasible_trials:
            candidate_x = np.array([trial.params[info[0]] for info in var_info])
            # re-evaluate with dense collision checking (every sample instead of every Nth)
            old_step = self.config.get("collisionCheckStep", 5)
            self.config["collisionCheckStep"] = 1
            f_verify, g_verify, _ = self.objectiveFunc(candidate_x)
            self.config["collisionCheckStep"] = old_step

            c_verify = self.testConstraints(g_verify)
            if c_verify:
                if f_verify < self.last_best_f:
                    self.last_best_f = f_verify
                    self.last_best_sol = candidate_x
                print(
                    f"Optuna best verified: {f_verify:.2f} ({len(feasible_trials)}/{len(study.trials)} passed sparse check)"
                )
                verified = True
                break
            else:
                print(
                    f"  candidate {trial.number} (obj={trial.value:.2f}) failed dense collision check, trying next..."
                )

        if not verified:
            print(f"Optuna: no solution survived dense collision verification ({len(study.trials)} trials)")

    def runOptimizer(self, opt_prob: Any) -> np.ndarray:
        """call global followed by local optimizer, return solution"""

        from pyoptsparse import ALPSO, IPOPT, NSGA2, PSQP, SLSQP, Optimization

        if self.config["useGlobalOptimization"]:
            ### global optimization
            sr = random.SystemRandom()
            if self.config["globalSolver"] == "NSGA2":
                opt = NSGA2()  # genetic algorithm
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
                opt.setOption("stopCriteria", 0)  # stop at max iters
                opt.setOption("dynInnerIter", 1)  # dynamic inner iter number
                opt.setOption("maxInnerIter", 3)
                opt.setOption("maxOuterIter", self.config["globalOptIterations"])
                opt.setOption("printInnerIters", 1)
                opt.setOption("printOuterIters", 1)
                opt.setOption("SwarmSize", self.config["globalOptSize"])
                opt.setOption("xinit", 1)
                opt.setOption("seed", sr.randint(1, 2**31))
                # slightly more explorative than defaults — local solver handles convergence
                opt.setOption("vmax", 3.0)  # higher max velocity for broader search (default 2.0)
                opt.setOption("c2", 0.7)  # weaker social pull, less premature convergence (default 1.0)
                opt.setOption("w2", 0.65)  # higher final inertia to keep searching longer (default 0.55)
                # TODO: how to properly limit max number of function calls?
                # no. func calls = (SwarmSize * inner) * outer + SwarmSize
                self.iter_max = opt.getOption("SwarmSize") * opt.getOption("maxInnerIter") * opt.getOption(
                    "maxOuterIter"
                ) + opt.getOption("SwarmSize")
            elif self.config["globalSolver"] == "Optuna":
                opt = None  # Optuna runs separately below
            else:
                print("Solver {} not defined".format(self.config["globalSolver"]))
                sys.exit(1)

            # run global optimization
            if self.config["verbose"]:
                print("Running global optimization with {}".format(self.config["globalSolver"]))
            self.is_global = True

            try:
                if self.config["globalSolver"] == "Optuna":
                    self._runOptuna()
                else:
                    sol = opt(opt_prob, storeHistory=False)
                    print(sol)
            except KeyboardInterrupt:
                print("\n\nGlobal optimization interrupted by user.")

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
                # IPRINT: -1=silent, 0=iter+final, 1=detailed
                opt2.setOption("IPRINT", -1 if not self.config["verbose"] else 0)
            elif self.config["localSolver"] == "IPOPT":
                opt2 = IPOPT()
                opt2.setOption(
                    "linear_solver", "mumps"
                )  # mumps (bundled) or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
                opt2.setOption("max_iter", self.config["localOptIterations"])
                if self.config["verbose"]:
                    opt2.setOption("print_level", 4)  # 0 none ... 5 max
                else:
                    opt2.setOption("print_level", 0)  # 0 none ... 5 max
            elif self.config["localSolver"] == "PSQP":
                opt2 = PSQP()
                opt2.setOption("MIT", self.config["localOptIterations"])  # max iterations
                # opt2.setOption('MFV', ??)  # max function evaluations

            self.iter_max = 0  # not meaningful for local opt (most FD probes fail silently)

            # Create a fresh opt_prob so ALPSO's NaN state doesn't contaminate
            # the starting point. Pass the ALPSO best as initial values directly
            # to addVar (setDVs is unreliable after ALPSO's NaN solution).
            init_vals = np.array(self.last_best_sol) if len(self.last_best_sol) > 0 else None
            opt_prob_local = Optimization("Trajectory optimization", self._objFuncWrapper)
            opt_prob_local.addObj("f")
            self.opt_prob = opt_prob_local
            self.addVarsAndConstraints(opt_prob_local, initial_values=init_vals)

            if self.config["verbose"]:
                print("Runing local optimization with {}".format(self.config["localSolver"]))
            self.is_global = False
            sens_step = self.config.get("localOptSensStep", 0.01)
            try:
                if self.config.get("useAnalyticalGradients", False):
                    print("Using analytical gradients for local optimization")
                    sol = opt2(opt_prob_local, sens=self._sensitivityWrapper, storeHistory=False)
                else:
                    sol = opt2(opt_prob_local, sens="FD", sensStep=sens_step, storeHistory=False)
            except KeyboardInterrupt:
                print("\n\nOptimization interrupted by user.")

            self.gather_solutions()

        if len(self.last_best_sol) > 0:
            print("using best constrained solution found during optimization.")
            print("verifying final solution (dense collision check, every sample)...")
            # verify with collision check on every sample
            old_step = self.config.get("collisionCheckStep", 5)
            self.config["collisionCheckStep"] = 1
            f_final, g_final, _ = self.objectiveFunc(self.last_best_sol, test=True)
            self.config["collisionCheckStep"] = old_step

            if self.testConstraints(g_final):
                print("Final solution is feasible (dense check passed).")
            else:
                print("WARNING: final solution has constraint violations in dense check!")
            print("\n")
            return self.last_best_sol
        else:
            # no feasible solution found at all — print raw solver output for debugging
            print(sol)
            print("No feasible solution found!")
            sys.exit(-1)
