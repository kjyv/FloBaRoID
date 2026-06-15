from collections.abc import Callable
from typing import Any

import fcl
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from idyntree import bindings as iDynTree
from pyoptsparse import Optimization

from excitation.optimizer import Optimizer, plotter
from excitation.trajectoryGenerator import (
    PulsedTrajectory,
)
from identification.helpers import URDFHelpers


def _collision_worker(
    transforms_chunk: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    configs_chunk: list[tuple[int, np.ndarray]],
    collision_pairs: list[tuple[str, str]],
    geom_data: dict[str, tuple[str, np.ndarray, np.ndarray | None, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FCL distances for a chunk of samples.

    Each worker builds its own FCL geometry objects from the pre-extracted geometry data
    (vertices, faces, offsets) to avoid pickling FCL objects.
    """
    n_pairs = len(collision_pairs)
    min_dists = np.full(n_pairs, 1e10)
    argmin_samples = np.full(n_pairs, -1, dtype=np.int64)

    # Build FCL geometry objects (once per worker)
    geom_cache: dict[str, tuple[Any, np.ndarray]] = {}
    for link_name, (gtype, data, faces, offset) in geom_data.items():
        if gtype == "box":
            geom_cache[link_name] = (fcl.Box(*data), offset)
        elif gtype == "convex" and faces is not None:
            convex = fcl.Convex(data, len(faces), faces)
            geom_cache[link_name] = (convex, offset)

    req = fcl.DistanceRequest(True)
    for transforms, (sample_idx, _) in zip(transforms_chunk, configs_chunk):
        for g_cnt, (l0, l1) in enumerate(collision_pairs):
            rot0, pos0 = transforms[l0]
            rot1, pos1 = transforms[l1]
            geom0, off0 = geom_cache[l0]
            geom1, off1 = geom_cache[l1]
            o0 = fcl.CollisionObject(geom0, fcl.Transform(rot0, pos0 + off0))
            o1 = fcl.CollisionObject(geom1, fcl.Transform(rot1, pos1 + off1))
            d = fcl.distance(o0, o1, req, fcl.DistanceResult())
            if d < min_dists[g_cnt]:
                min_dists[g_cnt] = d
                argmin_samples[g_cnt] = sample_idx

    return min_dists, argmin_samples


class TrajectoryOptimizer(Optimizer):
    def __init__(self, config, idf, model, simulation_func, world=None):
        super().__init__(config, idf, model, simulation_func, world=world)
        self.sim_func: Callable = simulation_func  # narrow: always non-None in this subclass

        # init some classes
        self.limits = URDFHelpers.getJointLimits(config["urdf"], use_deg=False)  # will always be compared to rad
        self.trajectory = PulsedTrajectory(self.num_dofs, use_deg=config["useDeg"])

        # bounded trajectory mode: positions are mapped through tanh to guarantee joint limits
        jn = self.model.jointNames
        if self.config.get("trajectoryBounded", False):
            self._joint_limits: list[tuple[float, float]] | None = [
                (self.limits[jn[i]]["lower"], self.limits[jn[i]]["upper"]) for i in range(self.num_dofs)
            ]
        else:
            self._joint_limits = None

        ## bounds for parameters
        # number of fourier partial sums per joint (more harmonics = higher frequency content)
        nf_config = self.config.get("trajectoryNf", None)
        if isinstance(nf_config, dict):
            # {joint_name: nf} — all joints must be specified
            missing = [name for name in self.model.jointNames if name not in nf_config]
            if missing:
                raise ValueError(f"trajectoryNf missing joints: {missing}")
            self.nf = [int(nf_config[name]) for name in self.model.jointNames]
        elif nf_config is None:
            self.nf = [4] * self.num_dofs
        else:
            raise ValueError("trajectoryNf must be a dict {joint_name: nf} with all joints listed")
        self.total_ab = sum(self.nf)  # total number of a (or b) coefficients across all joints

        # pulsation
        self.wf_min = self.config["trajectoryPulseMin"]
        self.wf_max = self.config["trajectoryPulseMax"]
        self.wf_init = self.config["trajectoryPulseInit"]

        # oscillation center offsets (degrees, added to URDF joint midpoint)
        # The bounded trajectory code clamps positions to URDF limits regardless of offset,
        # so these just control the preferred operating point for each joint.
        center_freedom = self.config.get("trajectoryCenterFreedom", 15.0)
        osc_centers = self.config.get("trajectoryOscillationCenters", {})
        if not isinstance(osc_centers, dict):
            raise ValueError("trajectoryOscillationCenters must be a dict {joint_name: center_deg}")
        # unlisted joints default to 0 (URDF midpoint)
        self.qinit: list[float] | np.ndarray = [float(osc_centers.get(name, 0.0)) for name in self.model.jointNames]
        self.qmin: list[float] | np.ndarray = [c - center_freedom for c in self.qinit]
        self.qmax: list[float] | np.ndarray = [c + center_freedom for c in self.qinit]

        if not self.config["useDeg"]:
            self.qmin = np.deg2rad(self.qmin)
            self.qmax = np.deg2rad(self.qmax)
            self.qinit = np.deg2rad(self.qinit)
        # sin/cos coefficients
        self.amin = self.bmin = self.config["trajectoryCoeffMin"]
        self.amax = self.bmax = self.config["trajectoryCoeffMax"]
        # per-joint init arrays (ragged — each joint may have different nf)
        # scale as coeff_init/k for harmonic k: low harmonics drive position/torque,
        # high harmonics taper off to avoid velocity limit violations
        coeff_init = self.config["trajectoryCoeffInit"]
        self.ainit: list[np.ndarray] = [
            np.array([coeff_init / (j + 1) for j in range(self.nf[i])]) for i in range(self.num_dofs)
        ]
        self.binit: list[np.ndarray] = [
            np.array([coeff_init / (j + 1) for j in range(self.nf[i])]) for i in range(self.num_dofs)
        ]

        self.last_best_f_f1 = 0.0

        self.num_constraints = self.num_dofs * 4  # angle, velocity, torque limits
        if self.config["minVelocityConstraint"]:
            self.num_constraints += self.num_dofs

        # hard constraint: each joint must reach at least minTorqueUtilization of its torque limit
        # (default is conservative — distal joints of serial chains have very small achievable torques)
        self.min_torque_utilization = self.config.get("minTorqueUtilization", 0.02)
        self.num_constraints += self.num_dofs

        # collision constraints

        loader = iDynTree.ModelLoader()
        loader.loadModelFromFile(self.config["urdf"])
        self.idyn_model = loader.model()
        self.neighbors = URDFHelpers.getNeighbors(self.idyn_model)

        # amount of collision checks to be done (exclude links without visual geometry)
        self.no_geometry_links = set(self.model.linkNames) - set(self.link_cuboid_hulls.keys())
        eff_links = (
            self.model.num_links
            - len(self.no_geometry_links)
            - len(self.config["ignoreLinksForCollision"])
            + len(self.world_links)
        )
        self.num_samples = int(self.config["excitationFrequency"] * self.trajectory.getPeriodLength())
        self.num_coll_constraints = eff_links * (eff_links - 1) // 2

        # ignore neighbors
        nb_pairs: list[tuple] = []
        for link in self.neighbors:
            if link in self.config["ignoreLinksForCollision"]:
                continue
            if link not in self.model.linkNames:
                continue
            if link not in self.link_cuboid_hulls:
                continue
            nb_real = (
                set(self.neighbors[link]["links"])
                .difference(self.config["ignoreLinksForCollision"])
                .difference(self.no_geometry_links)
                .intersection(self.model.linkNames)
            )
            for l in nb_real:
                if (link, l) not in nb_pairs and (l, link) not in nb_pairs:
                    nb_pairs.append((link, l))
        # only count ignore pairs where both links are in the effective set
        # (not already removed via ignoreLinksForCollision)
        ignored = set(self.config["ignoreLinksForCollision"]) | self.no_geometry_links
        all_links = set(self.model.linkNames + self.world_links)
        effective_ignore_pairs = [
            p
            for p in self.config["ignoreLinkPairsForCollision"]
            if p[0] in all_links and p[1] in all_links and p[0] not in ignored and p[1] not in ignored
        ]
        self.num_coll_constraints -= (
            len(nb_pairs)  # neighbors
            + len(effective_ignore_pairs)
        )  # custom combinations
        self.num_constraints += self.num_coll_constraints

        self.initVisualizer()

    def vecToParams(self, x):
        """Convert flat solution vector to separate parameter variables.

        Returns (wf, q, a, b) where a and b are lists of per-joint arrays (ragged).
        """
        wf = x[0]
        q = x[1 : self.num_dofs + 1]
        offset = self.num_dofs + 1
        a: list[np.ndarray] = []
        for i in range(self.num_dofs):
            a.append(np.array(x[offset : offset + self.nf[i]]))
            offset += self.nf[i]
        b: list[np.ndarray] = []
        for i in range(self.num_dofs):
            b.append(np.array(x[offset : offset + self.nf[i]]))
            offset += self.nf[i]
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

        x0 = np.asarray(x, dtype=float)
        f0 = f(*((x0,) + args))
        jac = np.zeros((x0.size, f0.size))
        dx = np.zeros(x0.size)
        for i in range(x0.size):
            dx[i] = epsilon
            jac[i] = (f(*((x0 + dx,) + args)) - f0) / epsilon
            dx[i] = 0.0
        return jac.transpose()

    def objectiveFunc(self, x, test=False):
        self.iter_cnt += 1

        # for global optimization, reject clearly invalid inputs early
        if self.is_global:
            if np.any(np.isnan(x)) or not self.testBounds(x):
                print(f"call #{self.iter_cnt} (invalid input, skipped)")
                return 1000.0, [10.0] * self.num_constraints, 1.0

        # for local optimization, replace NaN inputs with zeros (FD probe gone wrong)
        if np.any(np.isnan(x)):
            x = np.nan_to_num(x, nan=0.0)

        print(f"call #{self.iter_cnt}")

        wf, q, a, b = self.vecToParams(x)

        if self.config["verbose"]:
            print(f"wf {wf}")
            print(f"a {[np.round(ai, 5).tolist() for ai in a]}")
            print(f"b {[np.round(bi, 5).tolist() for bi in b]}")
            print(f"q {np.round(q, 5).tolist()}")

        self.trajectory.initWithParams(a, b, q, self.nf, wf, joint_limits=self._joint_limits)

        old_verbose = self.config["verbose"]
        self.config["verbose"] = 0
        # old_floatingBase = self.config['floatingBase']
        # self.config['floatingBase'] = 0
        trajectory_data, data = self.sim_func(self.config, self.trajectory, model=self.model)

        self.config["verbose"] = old_verbose
        # self.config['floatingBase'] = old_floatingBase

        self.last_trajectory_data = trajectory_data
        if self.config["showOptimizationTrajs"]:
            plotter(self.config, data=trajectory_data)

        # Regularized D-optimality: minimize -log(det(Y^T Y + δI))
        # The regularization δ ensures the gradient is numerically stable even when
        # the regressor is extremely ill-conditioned. It caps the effective condition
        # number at (λ_max + δ)/δ, preventing the gradient from losing precision.
        # δ is set relative to the largest eigenvalue so it's scale-invariant.
        YtY = self.model.YBase.T @ self.model.YBase
        eigvals = np.linalg.eigvalsh(YtY)
        lambda_max = float(eigvals[-1])
        # δ = fraction of λ_max; controls the conditioning of the gradient.
        # 1e-4 caps effective cond at ~10000, giving ~12 digits of gradient precision.
        dopt_regularization = self.config.get("doptRegularization", 1e-4)
        delta = dopt_regularization * max(lambda_max, 1e-30)
        eigvals_reg = eigvals + delta
        neg_log_det = -np.sum(np.log(np.maximum(eigvals_reg, 1e-300)))

        # Number of well-observable base parameters (eigenvalue above regularization threshold)
        n_observable = int(np.sum(eigvals > delta))

        if not hasattr(self, "_dopt_scale"):
            # target: D-optimality contribution ~10 for the initial trajectory
            self._dopt_scale = 10.0 / max(abs(neg_log_det), 1.0)
        f = neg_log_det * self._dopt_scale

        # If the simulation produced NaN or inf, clamp to a high but finite value
        self._eval_failed = False
        if not np.isfinite(f):
            f = 100.0
            self._eval_failed = True

        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        # add constraints  (later tested for all: g(n) <= 0)
        g = np.full(self.num_constraints, 1e10)
        jn = self.model.jointNames
        pos = trajectory_data["positions"]
        vel = trajectory_data["velocities"]
        torques = data.samples["torques"]

        # vectorized min/max across all samples for each joint
        fb = 6 if self.config["floatingBase"] else 0
        pos_min = np.min(pos, axis=0)
        pos_max = np.max(pos, axis=0)
        vel_absmax = np.max(np.abs(vel), axis=0)
        # skip base wrench columns (first 6) for floating-base torques
        torque_absmax = np.nanmax(np.abs(torques[:, fb:]), axis=0)

        for n in range(self.num_dofs):
            # check for joint limits
            # joint pos lower
            ovr = self.config.get("ovrPosLimit", {})
            # {joint_name: [lower_deg, upper_deg]}, unlisted joints use URDF limits
            ovr_pair = ovr.get(jn[n]) if isinstance(ovr, dict) else None
            if ovr_pair:
                g[n] = np.deg2rad(ovr_pair[0]) - pos_min[n]
            else:
                g[n] = self.limits[jn[n]]["lower"] - pos_min[n]
            # joint pos upper
            if ovr_pair:
                g[self.num_dofs + n] = pos_max[n] - np.deg2rad(ovr_pair[1])
            else:
                g[self.num_dofs + n] = pos_max[n] - self.limits[jn[n]]["upper"]
            # max joint vel
            g[2 * self.num_dofs + n] = vel_absmax[n] - self.limits[jn[n]]["velocity"]
            # max torques
            g[3 * self.num_dofs + n] = torque_absmax[n] - self.limits[jn[n]]["torque"]

            if self.config["minVelocityConstraint"]:
                # max joint vel of trajectory should at least be 10% of joint limit
                g[4 * self.num_dofs + n] = (
                    self.limits[jn[n]]["velocity"] * self.config["minVelocityPercentage"] - vel_absmax[n]
                )

            # hard constraint: each joint must reach min_torque_utilization of its torque limit
            # (g <= 0 means feasible, so: required - actual <= 0 when actual >= required)
            min_vel_offset = self.num_dofs if self.config["minVelocityConstraint"] else 0
            g[4 * self.num_dofs + min_vel_offset + n] = (
                self.limits[jn[n]]["torque"] * self.min_torque_utilization - torque_absmax[n]
            )

        # check collision constraints
        # (for whole trajectory but only get closest distance as constraint value)
        c_s = self.num_constraints - self.num_coll_constraints  # start where collision constraints start
        if self.config["verbose"] > 1:
            print("checking collisions")

        # pre-compute the list of link pairs to check (only once, not per sample)
        if not hasattr(self, "_collision_pairs"):
            all_links = self.model.linkNames + self.world_links
            ignore_links = set(self.config["ignoreLinksForCollision"]) | self.no_geometry_links
            ignore_pairs = {(a, b) for a, b in self.config["ignoreLinkPairsForCollision"]} | {
                (b, a) for a, b in self.config["ignoreLinkPairsForCollision"]
            }

            # optionally limit collision checks to links within a maximum kinematic distance
            # (dramatically reduces pairs for humanoids where e.g. left foot can't reach right hand)
            max_kin_dist = self.config.get("collisionMaxKinematicDistance", 0)

            def _kin_distance(start: str, target: str) -> int:
                """BFS shortest path in the kinematic tree."""
                visited = {start}
                queue = [(start, 0)]
                while queue:
                    current, dist = queue.pop(0)
                    if current == target:
                        return dist
                    for nb in self.neighbors.get(current, {}).get("links", []):
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, dist + 1))
                return 999

            # Build set of group-excluded pairs (links in different groups can't collide)
            group_ignore: set[tuple[str, str]] = set()
            for group_pair in self.config.get("ignoreCollisionBetweenGroups", []):
                if len(group_pair) == 2:
                    for a in group_pair[0]:
                        for b in group_pair[1]:
                            group_ignore.add((a, b))
                            group_ignore.add((b, a))

            self._collision_pairs: list[tuple[str, str]] = []
            for l0 in range(len(all_links)):
                for l1 in range(l0 + 1, len(all_links)):
                    l0_name = all_links[l0]
                    l1_name = all_links[l1]
                    if l0_name in ignore_links or l1_name in ignore_links:
                        continue
                    if (l0_name, l1_name) in ignore_pairs:
                        continue
                    if (l0_name, l1_name) in group_ignore:
                        continue
                    # neighbors can't collide with a proper joint range
                    if l0 < self.model.num_links and l1 < self.model.num_links:
                        if l0_name in self.neighbors[l1_name]["links"] or l1_name in self.neighbors[l0_name]["links"]:
                            continue
                    # skip pairs too far apart in the kinematic tree
                    if max_kin_dist > 0 and _kin_distance(l0_name, l1_name) > max_kin_dist:
                        continue
                    self._collision_pairs.append((l0_name, l1_name))
            if self.config.get("verbose", 0):
                print(f"Collision pairs: {len(self._collision_pairs)}")

        collision_step = self.config.get("collisionCheckStep", 3)
        use_ag = self.config.get("useAnalyticalGradients", False)
        if use_ag:
            coll_argmin: dict[int, tuple[int, np.ndarray]] = {}

        # Collect all configurations to check (main trajectory + transition)
        collision_configs: list[tuple[int, np.ndarray]] = []
        for p in range(0, pos.shape[0], collision_step):
            collision_configs.append((p, pos[p]))

        # also check the minimum-jerk transitions from/to zero position
        # (the trajectory is periodic so both paths are nearly identical,
        # but check both since pos[0] and pos[-1] may differ slightly)
        transition_duration = self.config.get("transitionDuration", 3.0)
        if transition_duration > 0:
            transition_samples = self.config.get("transitionCollisionSamples", 10)
            zero_pos = np.zeros(self.num_dofs)
            for q_boundary in [pos[0], pos[-1]]:
                for ti in range(transition_samples):
                    tau = (ti + 1) / (transition_samples + 1)
                    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                    q_trans = zero_pos + s * (q_boundary - zero_pos)
                    collision_configs.append((-1 - ti, q_trans))  # negative index = transition sample

        use_capsule = self.config.get("collisionMode", "convex") == "capsule" and self._capsules
        collision_link_names = sorted(set(l for pair in self._collision_pairs for l in pair))

        # Precompute and cache FCL collision objects per link (reuse across samples)
        # This avoids recreating CollisionObject wrappers every iteration.
        if not hasattr(self, "_fcl_objects"):
            self._fcl_objects: dict[str, tuple[Any, np.ndarray]] = {}
            for link_name in collision_link_names:
                geom, offset = self._getLinkCollisionGeometry(link_name)
                self._fcl_objects[link_name] = (fcl.CollisionObject(geom), offset)
        fcl_req = fcl.DistanceRequest(True)

        for p_idx, q_check in collision_configs:
            self.setCollisionRobotState(q_check)
            for g_cnt, (l0_name, l1_name) in enumerate(self._collision_pairs):
                if use_capsule and l0_name in self._capsules and l1_name in self._capsules:
                    d = self.getCapsuleDistance(l0_name, l1_name)
                else:
                    # Update transform on cached FCL objects (avoids recreating them)
                    rot0, pos0 = self._getLinkTransform(l0_name)
                    rot1, pos1 = self._getLinkTransform(l1_name)
                    obj0, off0 = self._fcl_objects[l0_name]
                    obj1, off1 = self._fcl_objects[l1_name]
                    obj0.setTransform(fcl.Transform(rot0, pos0 + off0))
                    obj1.setTransform(fcl.Transform(rot1, pos1 + off1))
                    d = fcl.distance(obj0, obj1, fcl_req, fcl.DistanceResult())
                if d < g[c_s + g_cnt]:
                    g[c_s + g_cnt] = d
                    if use_ag and p_idx >= 0:
                        coll_argmin[g_cnt] = (p_idx, q_check.copy())

        if use_ag:
            self._ag_collision_cache = coll_argmin

        self.last_g = g

        # Replace NaN constraint values with large violations (rather than returning
        # a penalty objective that corrupts the gradient)
        if np.any(np.isnan(g)):
            np.nan_to_num(g, copy=False, nan=10.0)  # treat NaN as violated

        # soft costs for trajectory quality (all scaled to ~0-10 range)
        torque_limits_arr = np.array([self.limits[jn[n]]["torque"] for n in range(self.num_dofs)])
        utilization = torque_absmax / torque_limits_arr

        # 1. torque balance: penalize uneven utilization across joints (CoV)
        util_mean = np.mean(utilization)
        if util_mean > 0:
            f1 = np.std(utilization) / util_mean  # coefficient of variation (0 = perfectly balanced)
        else:
            f1 = 1.0
        f += f1 * 10.0

        # 2. torque magnitude: penalize low overall torque utilization
        # (higher torques = better signal-to-noise for identification)
        target_utilization = self.config.get("trajectoryTargetTorqueUtil", 0.25)
        f3 = max(0.0, 1.0 - util_mean / target_utilization)  # 0 when mean >= target, up to 1 when no torque
        f += f3 * 10.0

        # 3. position range: penalize low joint range utilization
        pos_range_used = pos_max - pos_min
        pos_range_available = np.array(
            [self.limits[jn[n]]["upper"] - self.limits[jn[n]]["lower"] for n in range(self.num_dofs)]
        )
        pos_utilization = pos_range_used / pos_range_available
        pos_util_mean = np.mean(pos_utilization)
        f2 = (1.0 - pos_util_mean) * 10.0  # 0 when using full range, 10 when no motion
        f += f2

        # 4. velocity magnitude: penalize joints whose peak velocity stays below the
        # target. Friction (the viscous term in particular) is unidentifiable for
        # joints that barely move — D-optimality of the inertial regressor does not
        # reward velocity by itself, so without this term slow joints stay slow.
        vel_target = float(self.config.get("trajectoryTargetVelocity", 0.0))
        f4 = 0.0
        if vel_target > 0:
            vel_utilization = vel_absmax / vel_target
            f4 = float(np.mean(np.maximum(0.0, 1.0 - vel_utilization)))
            f += f4 * 10.0

        if self.config["verbose"] or f1 > 0.3:
            print(
                f"torque: {np.round(utilization, 3).tolist()} (bal={f1:.2f}, mag={util_mean:.2f}/{target_utilization})"
            )

        c = self.testConstraints(g)
        if c:
            print(Fore.GREEN, end=" ")
        else:
            print(Fore.YELLOW, end=" ")

        dopt_part = neg_log_det * self._dopt_scale
        print(
            f"obj: {f:.1f} (dopt: {dopt_part:.1f} [obs={n_observable}/{self.model.YBase.shape[1]}] + torq_bal: {f1 * 10.0:.1f} + torq_mag: {f3 * 10.0:.1f} + pos_range: {f2:.1f} + vel_mag: {f4 * 10.0:.1f}, best: {self.last_best_f:.1f})",
            end=" ",
        )
        print(Fore.RESET)

        is_gradient = getattr(getattr(self, "opt_prob", None), "is_gradient", False)
        if self.config["verbose"] and is_gradient:
            print("(Gradient evaluation)")
        if not is_gradient and self.config["showOptimizationGraph"]:
            self.xar.append(self.iter_cnt)
            self.yar.append(f)
            self.x_constr.append(c)
            self.updateGraph()

        self.showVisualizerTrajectory(self.trajectory)

        # keep last best solution (some solvers don't keep it)
        if c and f < self.last_best_f:
            self.last_best_f = f
            self.last_best_f_f1 = f1
            self.last_best_sol = x

        print("\n\n")

        # Cache data for analytical gradient computation
        if self.config.get("useAnalyticalGradients", False) and not self._eval_failed:
            self._ag_cache = {
                "YBase": self.model.YBase.copy(),
                "positions": pos.copy(),
                "velocities": vel.copy(),
                "accelerations": trajectory_data["accelerations"].copy(),
                "times": trajectory_data["times"].copy(),
                "torques": torques.copy(),
                "n_observable": n_observable,
                "dopt_scale": self._dopt_scale,
                "torque_absmax_idx": np.argmax(np.abs(torques[:, fb:]), axis=0),
                "pos_min_idx": np.argmin(pos, axis=0),
                "pos_max_idx": np.argmax(pos, axis=0),
                "vel_absmax_idx": np.argmax(np.abs(vel), axis=0),
                "vel_absmax": vel_absmax.copy(),
                "utilization": utilization.copy(),
                "util_mean": float(util_mean),
                "util_std": float(np.std(utilization)),
                "f1": float(f1),
                "f3": float(f3),
                "pos_range_available": pos_range_available.copy(),
            }

        # IPOPT respects the fail flag (discards the evaluation), SLSQP doesn't
        # (it treats the returned values as real, so fail=1 corrupts its gradient)
        fail = 1.0 if (self._eval_failed and self.config.get("localSolver") == "IPOPT") else 0.0
        return f, g, fail

    def testBounds(self, x):
        # test variable bounds
        wf, q, a, b = self.vecToParams(x)
        wf_t = wf >= self.wf_min and wf <= self.wf_max
        q_t = np.all(q <= self.qmax) and np.all(q >= self.qmin)
        a_t = all(np.all(a[i] <= self.amax) and np.all(a[i] >= self.amin) for i in range(len(a)))
        b_t = all(np.all(b[i] <= self.bmax) and np.all(b[i] >= self.bmin) for i in range(len(b)))
        res = wf_t and q_t and a_t and b_t

        if not res:
            print("bounds violated")

        return res

    def testConstraints(self, g):
        g = np.array(g)
        c_s = self.num_constraints - self.num_coll_constraints  # start where collision constraints start
        res = np.all(g[:c_s] <= self.config["minTolConstr"])
        res_c = np.all(g[c_s:] > 0)
        if not res:
            print("constraints violated:")
            if True in np.isin(
                list(range(1, 2 * self.num_dofs)),
                np.where(g >= self.config["minTolConstr"]),
            ):
                print("- angle limits")
                print(np.array(g)[list(range(1, 2 * self.num_dofs))])
            if True in np.isin(
                list(range(2 * self.num_dofs, 3 * self.num_dofs)),
                np.where(g >= self.config["minTolConstr"]),
            ):
                print("- max velocity limits")
                # print np.array(g)[range(2*self.num_dofs,3*self.num_dofs)]
            if True in np.isin(
                list(range(3 * self.num_dofs, 4 * self.num_dofs)),
                np.where(g >= self.config["minTolConstr"]),
            ):
                print("- max torque limits")

            if self.config["minVelocityConstraint"]:
                if True in np.isin(
                    list(range(4 * self.num_dofs, 5 * self.num_dofs)),
                    np.where(g >= self.config["minTolConstr"]),
                ):
                    print("- min velocity limits")

            # min torque utilization constraints
            min_vel_offset = self.num_dofs if self.config["minVelocityConstraint"] else 0
            min_torq_start = 4 * self.num_dofs + min_vel_offset
            min_torq_end = min_torq_start + self.num_dofs
            if True in np.isin(
                list(range(min_torq_start, min_torq_end)),
                np.where(g >= self.config["minTolConstr"]),
            ):
                print("- min torque utilization limits")

            if not res_c:
                print("- collision constraints")
        elif not res_c:
            print("constraints violated:")
            print("- collision constraints")
            # show the worst collision violations
            coll_g = g[c_s:]
            worst = np.argsort(coll_g)[:5]
            for idx in worst:
                if coll_g[idx] <= 0:
                    pair = self._collision_pairs[idx] if hasattr(self, "_collision_pairs") else f"pair {idx}"
                    print(f"  {pair}: distance={coll_g[idx]:.4f}")
        return res and res_c

    def testParams(self, **kwargs):
        x = kwargs["x_new"]
        return self.testBounds(x) and self.testConstraints(self.last_g)

    def _objFuncWrapper(self, xdict: dict) -> tuple[dict, bool]:
        """Wrapper to convert pyOptSparse dict-based API to flat array."""
        x = np.array([xdict[name] for name in self._var_names]).flatten()
        f, g, fail = self.objectiveFunc(x)
        funcs = {"f": f, "g": np.array(g)}
        return funcs, bool(fail)

    def _sensitivityWrapper(self, xdict: dict, funcs: dict) -> tuple[dict, bool]:
        """Compute analytical gradients for pyOptSparse.

        Called by pyOptSparse when sens= is set to this method. Uses cached
        data from the last objectiveFunc call to compute exact gradients via
        the chain rule through the regressor SVD.
        """
        from excitation.analyticalGradient import compute_analytical_gradient

        if not hasattr(self, "_ag_cache"):
            print("Warning: no cached data for analytical gradient, returning zeros")
            func_sens: dict = {"f": {}, "g": {}}
            for name in self._var_names:
                func_sens["f"][name] = np.array([0.0])
                func_sens["g"][name] = np.zeros((self.num_constraints, 1))
            return func_sens, False

        obj_grad, con_grad = compute_analytical_gradient(self)

        func_sens = {"f": {}, "g": {}}
        for k, name in enumerate(self._var_names):
            func_sens["f"][name] = np.array([obj_grad[k]])
            func_sens["g"][name] = con_grad[:, k].reshape(-1, 1)

        return func_sens, False

    def addVarsAndConstraints(self, opt_prob: Any, initial_values: np.ndarray | None = None) -> None:
        """Add variables, define bounds.

        Args:
            opt_prob: pyOptSparse Optimization object
            initial_values: optional flat array of initial variable values (e.g. from a
                previous global optimization). If None, uses default init values.
        """

        self._var_names: list[str] = []
        idx = 0

        # w_f - pulsation
        self._var_names.append("wf")
        wf_val = float(initial_values[idx]) if initial_values is not None else self.wf_init
        opt_prob.addVar("wf", value=wf_val, lower=self.wf_min, upper=self.wf_max)
        idx += 1

        # q - offsets
        for i in range(self.num_dofs):
            name = "q_%d" % i
            self._var_names.append(name)
            q_val = float(initial_values[idx]) if initial_values is not None else self.qinit[i]
            opt_prob.addVar(name, value=q_val, lower=self.qmin[i], upper=self.qmax[i])
            idx += 1
        # a, b - sin/cos params (per-joint nf, so each joint may have different count)
        for i in range(self.num_dofs):
            for j in range(self.nf[i]):
                name = f"a{i}_{j}"
                self._var_names.append(name)
                a_val = float(initial_values[idx]) if initial_values is not None else self.ainit[i][j]
                opt_prob.addVar(name, value=a_val, lower=self.amin, upper=self.amax)
                idx += 1
        for i in range(self.num_dofs):
            for j in range(self.nf[i]):
                name = f"b{i}_{j}"
                self._var_names.append(name)
                b_val = float(initial_values[idx]) if initial_values is not None else self.binit[i][j]
                opt_prob.addVar(name, value=b_val, lower=self.bmin, upper=self.bmax)
                idx += 1

        # add constraint vars (constraint functions are in obfunc).
        # Joint/velocity/torque constraints use g <= 0 for feasible (e.g.
        # g = limit_lower - min_position < 0 when joint is above lower limit).
        # Collision constraints use g >= 0 for feasible (positive = no collision).
        c_s = self.num_constraints - self.num_coll_constraints
        lower = np.concatenate([-np.inf * np.ones(c_s), np.zeros(self.num_constraints - c_s)])
        upper = np.concatenate([np.zeros(c_s), np.inf * np.ones(self.num_constraints - c_s)])
        opt_prob.addConGroup("g", self.num_constraints, lower=lower, upper=upper)

    def optimizeTrajectory(self) -> "PulsedTrajectory":
        """Find trajectory parameters by optimization.

        Catches KeyboardInterrupt so that the best solution found so far
        is returned even if the user aborts early with Ctrl-C.
        """

        # Instanciate Optimization Problem
        opt_prob = Optimization("Trajectory optimization", self._objFuncWrapper)
        opt_prob.addObj("f")
        self.opt_prob = opt_prob

        self.addVarsAndConstraints(opt_prob)
        try:
            sol_vec = self.runOptimizer(opt_prob)
        except KeyboardInterrupt:
            print("\n\nTrajectory optimization interrupted by user.")
            if len(self.last_best_sol) > 0:
                print(f"Returning best feasible solution found (obj={self.last_best_f:.2f}).")
                sol_vec = self.last_best_sol
            else:
                print("WARNING: no feasible solution found before interruption, using last evaluation.")
                # Use whatever the last evaluated point was
                sol_vec = np.array(
                    [self.wf_init]
                    + list(self.qinit)
                    + [c for ai in self.ainit for c in ai]
                    + [c for bi in self.binit for c in bi]
                )

        sol_wf, sol_q, sol_a, sol_b = self.vecToParams(sol_vec)
        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf, joint_limits=self._joint_limits)

        if self.config["showOptimizationGraph"]:
            plt.ioff()

        return self.trajectory
