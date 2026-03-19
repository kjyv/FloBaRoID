from collections.abc import Callable
from typing import Any

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
        if nf_config and len(nf_config) == self.num_dofs:
            self.nf = list(nf_config)
        else:
            self.nf = [4] * self.num_dofs
        self.total_ab = sum(self.nf)  # total number of a (or b) coefficients across all joints

        # pulsation
        self.wf_min = self.config["trajectoryPulseMin"]
        self.wf_max = self.config["trajectoryPulseMax"]
        self.wf_init = self.config["trajectoryPulseInit"]

        # angle offsets
        if self.config["trajectoryAngleRanges"] and self.config["trajectoryAngleRanges"][0] is not None:
            self.qmin: list[float] | np.ndarray = []
            self.qmax: list[float] | np.ndarray = []
            self.qinit: list[float] | np.ndarray = []
            for i in range(0, self.num_dofs):
                low = self.config["trajectoryAngleRanges"][i][0]
                high = self.config["trajectoryAngleRanges"][i][1]
                self.qmin.append(low)
                self.qmax.append(high)
                self.qinit.append((high + low) * 0.5)  # set init to middle of range
        else:
            self.qmin = [self.config["trajectoryAngleMin"]] * self.num_dofs
            self.qmax = [self.config["trajectoryAngleMax"]] * self.num_dofs
            self.qinit = [
                0.5 * self.config["trajectoryAngleMin"] + 0.5 * self.config["trajectoryAngleMax"]
            ] * self.num_dofs

        if not self.config["useDeg"]:
            self.qmin = np.deg2rad(self.qmin)
            self.qmax = np.deg2rad(self.qmax)
            self.qinit = np.deg2rad(self.qinit)
        # sin/cos coefficients
        self.amin = self.bmin = self.config["trajectoryCoeffMin"]
        self.amax = self.bmax = self.config["trajectoryCoeffMax"]
        # per-joint init arrays (ragged — each joint may have different nf)
        coeff_init = self.config["trajectoryCoeffInit"]
        self.ainit: list[np.ndarray] = [np.full(self.nf[i], coeff_init) for i in range(self.num_dofs)]
        self.binit: list[np.ndarray] = [np.full(self.nf[i], coeff_init) for i in range(self.num_dofs)]

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

        cond = np.linalg.cond(self.model.YBase)
        # use log of condition number so the scale is manageable, then normalize
        # so it's comparable to the other penalties (~5-15 range).
        # on the first valid evaluation, record the baseline and use it for scaling.
        # clamp to avoid inf (degenerate regressor from e.g. zero-frequency trajectory)
        log_cond = min(np.log10(max(cond, 1.0)), 100.0)
        if not hasattr(self, "_cond_scale"):
            # target: condition number contribution ~10 for the initial trajectory
            self._cond_scale = 10.0 / max(log_cond, 1.0)
        f = log_cond * self._cond_scale

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
            if len(self.config["ovrPosLimit"]) > n and self.config["ovrPosLimit"][n]:
                g[n] = np.deg2rad(self.config["ovrPosLimit"][n][0]) - pos_min[n]
            else:
                g[n] = self.limits[jn[n]]["lower"] - pos_min[n]
            # joint pos upper
            if len(self.config["ovrPosLimit"]) > n and self.config["ovrPosLimit"][n]:
                g[self.num_dofs + n] = pos_max[n] - np.deg2rad(self.config["ovrPosLimit"][n][1])
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
            self._collision_pairs: list[tuple[str, str]] = []
            for l0 in range(len(all_links)):
                for l1 in range(l0 + 1, len(all_links)):
                    l0_name = all_links[l0]
                    l1_name = all_links[l1]
                    if l0_name in ignore_links or l1_name in ignore_links:
                        continue
                    if (l0_name, l1_name) in ignore_pairs:
                        continue
                    # neighbors can't collide with a proper joint range
                    if l0 < self.model.num_links and l1 < self.model.num_links:
                        if l0_name in self.neighbors[l1_name]["links"] or l1_name in self.neighbors[l0_name]["links"]:
                            continue
                    self._collision_pairs.append((l0_name, l1_name))

        for p in range(0, pos.shape[0], 10):
            if self.config["verbose"] > 1:
                print(f"Sample {p}")
            q = pos[p]
            # set robot state once per sample (not per link pair)
            self.setCollisionRobotState(q)
            for g_cnt, (l0_name, l1_name) in enumerate(self._collision_pairs):
                d = self.getLinkDistance(l0_name, l1_name, q)
                if d < g[c_s + g_cnt]:
                    g[c_s + g_cnt] = d

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

        if self.config["verbose"] or f1 > 0.3:
            print(
                f"torque: {np.round(utilization, 3).tolist()} (bal={f1:.2f}, mag={util_mean:.2f}/{target_utilization})"
            )

        c = self.testConstraints(g)
        if c:
            print(Fore.GREEN, end=" ")
        else:
            print(Fore.YELLOW, end=" ")

        cond_part = log_cond * self._cond_scale
        print(
            f"obj: {f:.1f} (cond: {cond_part:.1f} + torq_bal: {f1 * 10.0:.1f} + torq_mag: {f3 * 10.0:.1f} + pos_range: {f2:.1f}, best: {self.last_best_f:.1f})",
            end=" ",
        )
        print(Fore.RESET)

        if self.config["verbose"]:
            if hasattr(self.opt_prob, "is_gradient") and self.opt_prob.is_gradient:
                print("(Gradient evaluation)")

        if (
            self.mpi_rank == 0
            and not getattr(self.opt_prob, "is_gradient", False)
            and self.config["showOptimizationGraph"]
        ):
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
        # use non-linear optimization to find parameters for minimal
        # condition number trajectory

        # Instanciate Optimization Problem
        opt_prob = Optimization("Trajectory optimization", self._objFuncWrapper)
        opt_prob.addObj("f")
        self.opt_prob = opt_prob

        self.addVarsAndConstraints(opt_prob)
        sol_vec = self.runOptimizer(opt_prob)

        sol_wf, sol_q, sol_a, sol_b = self.vecToParams(sol_vec)
        self.trajectory.initWithParams(sol_a, sol_b, sol_q, self.nf, sol_wf)

        if self.config["showOptimizationGraph"]:
            plt.ioff()

        return self.trajectory
