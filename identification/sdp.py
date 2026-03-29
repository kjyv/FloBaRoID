"""SDP-based physically consistent parameter identification using cvxpy.

Supports multiple solvers (CLARABEL, SCS, MOSEK, etc.) through cvxpy's backend
abstraction.

Based on Sousa, 2014: "Physical feasibility of robot base inertial parameter
identification: A linear matrix inequality approach" but uses a direct cvxopt
formulation instead of author's sympy/lmi_sdp code.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from identifier import Identification

import cvxpy as cp
import numpy as np
import numpy.linalg as la
from colorama import Fore

from identification import sdp_helpers
from identification.helpers import Timer
from identification.sdp_helpers import sdp_scalar as _s


class SDP:
    """SDP-based identification using cvxpy."""

    def __init__(self, idf: Identification) -> None:
        self.idf = idf

        # set solver config
        self.solver_name = idf.opt.get("sdpSolver", "clarabel")
        self.solver_opts: dict[str, Any] = idf.opt.get("sdpSolverOptions", {})

        # collect constraint flags for display
        self.constr_per_param: dict[int, list[str]] = {}
        for i in self.idf.model.identified_params:
            self.constr_per_param[i] = []

    def checkFeasibility(self, prime: np.ndarray) -> bool:
        """Check if a parameter vector satisfies all physical consistency constraints."""
        print("Checking feasibility of a priori parameters...")
        idable_params = sorted(list(set(self.idf.model.identified_params).difference(self.delete_cols)))

        # temporarily set the variable values to check constraints, then restore
        old_value = self.x.value
        self.x.value = prime[idable_params]

        feasible = True
        for i, c in enumerate(self.constraints):
            viol = c.violation()
            if isinstance(viol, np.ndarray):
                max_viol = float(np.max(viol))
            else:
                max_viol = float(viol)
            if max_viol > 1e-6:
                print(f"Constraint {i} violated (max violation: {max_viol:.6f})")
                feasible = False

        # restore previous value so checkFeasibility doesn't affect solver initialization
        self.x.value = old_value
        return feasible

    def initSDP_LMIs(self, idf: Identification, remove_nonid: bool = True) -> None:
        """Initialize constraints for physical consistency using cvxpy."""
        with Timer() as t:
            if idf.opt["verbose"]:
                print("Initializing cvxpy constraints...")

            zero = np.zeros((1, 1))

            # don't include equations for 0'th link (in case it's fixed)
            if idf.opt["floatingBase"] == 0 and idf.opt["deleteFixedBase"]:
                if idf.opt["identifyGravityParamsOnly"]:
                    self.delete_cols = [0, 1, 2, 3]
                else:
                    self.delete_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                if set(self.delete_cols).issubset(idf.model.non_id):
                    start_link = 1
                else:
                    start_link = 0
                    self.delete_cols = []
            else:
                start_link = 0
                self.delete_cols = []

            # create the cvxpy decision variable vector
            idable_params = sorted(list(set(idf.model.identified_params).difference(self.delete_cols)))
            self.x = cp.Variable(len(idable_params), name="x_std")
            self.param_index_map = {p: i for i, p in enumerate(idable_params)}
            self.epsilon_safemargin = float(idf.opt.get("sdpSafeMargin", 1e-6))

            self.constraints: list[Any] = []

            def idx(p: int) -> int:
                return self.param_index_map[p]

            # links with all 10 params pinned don't need PSD constraints
            # (their a priori values may be physically inconsistent, e.g. zero-mass virtual links)
            pinned_params = set(idf.opt.get("dontChangeParams", []))
            pinned_links = set()
            for i in range(idf.model.num_links):
                link_params = set(range(i * 10, i * 10 + 10))
                if link_params.issubset(pinned_params) or link_params.issubset(self.delete_cols):
                    pinned_links.add(i)

            if idf.opt["identifyGravityParamsOnly"]:
                # only constrain mass > 0
                for i in range(start_link, idf.model.num_links):
                    if i * 10 not in self.delete_cols and i not in pinned_links:
                        p = idf.model.mass_params[i]
                        self.constraints.append(self.x[idx(p)] >= self.epsilon_safemargin)
            else:
                # pseudo-inertia PSD constraint for each link
                for i in range(start_link, idf.model.num_links):
                    if i in pinned_links:
                        continue
                    m = self.x[idx(i * 10)]
                    mx = self.x[idx(i * 10 + 1)]
                    my = self.x[idx(i * 10 + 2)]
                    mz = self.x[idx(i * 10 + 3)]
                    Ixx = self.x[idx(i * 10 + 4)]
                    Ixy = self.x[idx(i * 10 + 5)]
                    Ixz = self.x[idx(i * 10 + 6)]
                    Iyy = self.x[idx(i * 10 + 7)]
                    Iyz = self.x[idx(i * 10 + 8)]
                    Izz = self.x[idx(i * 10 + 9)]

                    # D_i = [[L, S(l)^T], [S(l), m*I_3]]
                    Di = cp.bmat(
                        [
                            [_s(Ixx), _s(Ixy), _s(Ixz), zero, _s(mz), _s(-my)],
                            [_s(Ixy), _s(Iyy), _s(Iyz), _s(-mz), zero, _s(mx)],
                            [_s(Ixz), _s(Iyz), _s(Izz), _s(my), _s(-mx), zero],
                            [zero, _s(-mz), _s(my), _s(m), zero, zero],
                            [_s(mz), zero, _s(-mx), zero, _s(m), zero],
                            [_s(-my), _s(mx), zero, zero, zero, _s(m)],
                        ]
                    )
                    self.constraints.append(Di >> self.epsilon_safemargin * np.eye(6))

            # params to skip (noChange + dontChangeParams)
            params_to_skip: list[int] = []
            if idf.opt["noChange"]:
                linkConds = idf.model.getSubregressorsConditionNumbers()
            robotmass_apriori = 0.0
            for i in range(0, idf.model.num_links):
                robotmass_apriori += idf.model.xStdModel[i * 10]
                if idf.opt["noChange"] and linkConds[i] > idf.opt["noChangeThresh"]:
                    print(Fore.YELLOW + f"not changing parameters of link {i} ({idf.model.linkNames[i]})!" + Fore.RESET)
                    params_to_skip.extend(range(i * 10, i * 10 + 10))

            for p in idf.opt["dontChangeParams"]:
                params_to_skip.append(p)

            for p in set(params_to_skip):
                if p in self.delete_cols:
                    continue
                if (idf.opt["identifyGravityParamsOnly"] and p not in idf.model.inertia_params) or not idf.opt[
                    "identifyGravityParamsOnly"
                ]:
                    if p not in idf.opt["dontConstrain"]:
                        # pin to a priori: upper and lower bound
                        self.constraints.append(self.x[idx(p)] <= idf.model.xStdModel[p])
                        self.constraints.append(self.x[idx(p)] >= idf.model.xStdModel[p])
                        self.constr_per_param[p].append("cad")

            # constrain overall mass
            if idf.opt["limitOverallMass"]:
                if idf.opt["limitMassVal"]:
                    robotmaxmass = idf.opt["limitMassVal"] - sum(idf.model.xStdModel[0 : start_link * 10 : 10])
                else:
                    robotmaxmass = robotmass_apriori
                robotmaxmass_ub = robotmaxmass + idf.opt["limitMassRange"]
                robotmaxmass_lb = robotmaxmass - idf.opt["limitMassRange"]

                mass_sum = sum(self.x[idx(idf.model.mass_params[i])] for i in range(start_link, idf.model.num_links))
                self.constraints.append(mass_sum <= robotmaxmass_ub)
                self.constraints.append(mass_sum >= robotmaxmass_lb)

            # constrain masses per link
            if idf.opt["limitMassToApriori"]:
                if idf.opt["noChange"]:
                    linkConds = idf.model.getSubregressorsConditionNumbers()
                for i in range(start_link, idf.model.num_links):
                    if i in pinned_links:
                        continue
                    if not (idf.opt["noChange"] and linkConds[i] > idf.opt["noChangeThresh"]):
                        p = i * 10
                        if p not in idf.opt["dontConstrain"]:
                            bound = np.abs(idf.model.xStdModel[p]) * idf.opt["limitMassAprioriBoundary"]
                            self.constraints.append(self.x[idx(p)] <= idf.model.xStdModel[p] + bound)
                            self.constraints.append(self.x[idx(p)] >= idf.model.xStdModel[p] - bound)
                            self.constr_per_param[p].append("mA")

            # constrain COM to a priori
            if idf.opt["limitCOMToApriori"]:
                if idf.opt["noChange"]:
                    linkConds = idf.model.getSubregressorsConditionNumbers()
                for i in range(start_link, idf.model.num_links):
                    if i in pinned_links:
                        continue
                    if not (idf.opt["noChange"] and linkConds[i] > idf.opt["noChangeThresh"]):
                        for p in range(i * 10 + 1, i * 10 + 4):
                            if p not in idf.opt["dontConstrain"]:
                                bound = np.abs(idf.model.xStdModel[p]) * idf.opt["limitCOMAprioriBoundary"]
                                if np.abs(idf.model.xStdModel[p]) < 0.01:
                                    bound += 0.01
                                self.constraints.append(self.x[idx(p)] <= idf.model.xStdModel[p] + bound)
                                self.constraints.append(self.x[idx(p)] >= idf.model.xStdModel[p] - bound)
                                self.constr_per_param[p].append("cA")

            # constrain COM to bounding hull
            if idf.opt["restrictCOMtoHull"]:
                link_cuboid_hulls: dict[str, tuple] = {}
                if idf.opt["noChange"]:
                    linkConds = idf.model.getSubregressorsConditionNumbers()
                for i in range(start_link, idf.model.num_links):
                    if i in pinned_links:
                        continue
                    if not (idf.opt["noChange"] and linkConds[i] > idf.opt["noChangeThresh"]):
                        link_name = idf.model.linkNames[i]
                        link_mass = idf.model.xStdModel[i * 10]
                        if abs(link_mass) > 1e-10:
                            old_com = idf.model.xStdModel[i * 10 + 1 : i * 10 + 4] / link_mass
                        else:
                            old_com = np.zeros(3)
                        box, pos, rot = idf.urdfHelpers.getBoundingBox(
                            input_urdf=idf.model.urdf_file,
                            old_com=old_com,
                            link_name=link_name,
                        )
                        link_cuboid_hulls[link_name] = (box, pos, rot)
                        m_var = self.x[idx(i * 10)]

                        for j in range(3):
                            p = i * 10 + 1 + j
                            if p not in self.delete_cols and p not in idf.opt["dontConstrain"]:
                                l_var = self.x[idx(p)]
                                self.constraints.append(l_var >= m_var * box[0][j])
                                self.constraints.append(l_var <= m_var * box[1][j])
                                self.constr_per_param[p].append("hull")

            elif not idf.opt["limitCOMToApriori"] and idf.opt["identifyGravityParamsOnly"]:
                print(Fore.RED + "COM parameters are not constrained,", end=" ")
                print("might result in rank deficiency when solving SDP problem!" + Fore.RESET)

            # symmetry constraints
            if idf.opt["useSymmetryConstraints"] and idf.opt["symmetryConstraints"]:
                for a, b, sign in idf.opt["symmetryConstraints"]:
                    if (
                        idf.opt["identifyGravityParamsOnly"]
                        and a not in idf.model.inertia_params
                        and b not in idf.model.inertia_params
                    ) or not idf.opt["identifyGravityParamsOnly"]:
                        eps = idf.opt["symmetryTolerance"]
                        diff = self.x[idx(a)] - sign * self.x[idx(b)]
                        # Schur complement: (a - sign*b)^2 <= eps
                        M = cp.bmat([[np.array([[eps]]), _s(diff)], [_s(diff), np.ones((1, 1))]])
                        self.constraints.append(M >> 0)
                        self.constr_per_param[a].append("sym")
                        self.constr_per_param[b].append("sym")

            # friction constraints
            if idf.opt["identifyFriction"]:
                if not idf.opt["identifyGravityParamsOnly"]:
                    for i in range(idf.model.num_dofs):
                        # Fv > 0
                        p_fv = idf.model.num_model_params + idf.model.num_dofs + i
                        self.constraints.append(self.x[idx(p_fv)] >= self.epsilon_safemargin)
                        self.constr_per_param[p_fv].append(">0")
                        if not idf.opt["identifySymmetricVelFriction"]:
                            p_fv2 = idf.model.num_model_params + idf.model.num_dofs * 2 + i
                            self.constraints.append(self.x[idx(p_fv2)] >= self.epsilon_safemargin)
                            self.constr_per_param[p_fv2].append(">0")

        if idf.opt["showTiming"]:
            print(f"Initializing cvxpy constraints took {t.interval:.3f} sec.")

    def identifyFeasibleStandardParameters(self, idf: Identification) -> None:
        """Find physically consistent standard parameters minimizing torque error."""
        with Timer() as t:
            if idf.opt["verbose"]:
                print("Preparing cvxpy SDP...")

            idable_params = sorted(list(set(idf.model.identified_params).difference(self.delete_cols)))

            YBase = idf.model.YBase
            tau = idf.model.torques_stack

            # get projection matrix
            if idf.opt["useBasisProjection"]:
                K = idf.model.Binv
            else:
                K = idf.model.K

            # remove delete_cols from K
            K = np.delete(K, self.delete_cols, axis=1)

            Q, R = la.qr(YBase)
            Q1 = Q[:, 0 : idf.model.num_base_params]
            R1 = np.array(R[: idf.model.num_base_params, : idf.model.num_base_params])
            rho1 = Q1.T.dot(tau)

            contactForces = Q.T.dot(idf.model.contactForcesSum)

            if idf.opt["useRegressorRegularization"]:
                p_nid = idf.model.non_id
                p_nid = list(
                    set(p_nid).difference(set(self.delete_cols)).intersection(set(idf.model.identified_params))
                )
                contactForces = np.concatenate((contactForces, np.zeros(len(p_nid))))

            if idf.opt["verbose"] > 1:
                print("Building cost matrix...", time.ctime())

            # residual: e = rho1 - contactForces - R1*K*x
            # (must subtract contactForcesSum consistently for Schur complement)
            rho2_norm_sqr = (
                la.norm(idf.model.torques_stack - idf.model.contactForcesSum - idf.model.YBase.dot(idf.model.xBase))
                ** 2
            )

            if idf.opt["useRegressorRegularization"] and len(p_nid):
                l_reg = (float(idf.base_error) / len(p_nid)) * idf.opt["regularizationFactor"]
                # augmented system: [R1*K; l*I_nid] @ x ≈ [rho1; l*xStdModel_nid]
                R1_K = R1 @ K
                Y_bot = np.zeros((len(p_nid), len(idable_params)))
                for i, p in enumerate(p_nid):
                    Y_bot[i, self.param_index_map[p]] = l_reg
                Y_combined = np.vstack([R1_K, Y_bot])  # numpy stack, then one cvxpy multiply

                rho1_hat = np.concatenate((rho1, l_reg * idf.model.xStdModel[p_nid]))
                e_rho1 = rho1_hat - contactForces - Y_combined @ self.x
            else:
                R1_K = R1 @ K  # pure numpy multiply
                e_rho1 = rho1 - contactForces - R1_K @ self.x

            if idf.opt["verbose"] > 1:
                print("Building Schur complement...", time.ctime())

            # Schur complement: [[u - rho2, e^T], [e, I]] >> 0
            u = cp.Variable(name="u")
            n = e_rho1.shape[0]
            U_rho = cp.bmat(
                [
                    [_s(u - rho2_norm_sqr), cp.reshape(e_rho1, (1, n), order="C")],
                    [cp.reshape(e_rho1, (n, 1), order="C"), np.eye(n)],
                ]
            )

            all_constraints = [U_rho >> 0] + self.constraints
            prob = cp.Problem(cp.Minimize(u), all_constraints)

            if idf.opt["verbose"]:
                print(f"Solving SDP with {self.solver_name} ({len(all_constraints)} constraints)...")

            if idf.opt["checkAPrioriFeasibility"]:
                self.checkFeasibility(idf.model.xStdModel)

            # warm start from a priori values — helps CLARABEL find an initial
            # factorization for ill-conditioned problems
            self.x.value = idf.model.xStdModel[idable_params]

            status = sdp_helpers.solve_sdp(
                prob,
                solver_name=self.solver_name,
                solver_opts=self.solver_opts,
                verbose=idf.opt.get("verbose", 0) > 1,
            )

            if status in ("optimal", "optimal_inaccurate"):
                print(f"SDP found std solution with {u.value:.2f} squared residual error")
                idf.model.xStd = np.array(self.x.value).flatten()
            else:
                print(Fore.YELLOW + f"SDP solver failed ({status}), keeping a priori parameters" + Fore.RESET)
                idf.model.xStd = idf.model.xStdModel.copy()

            # prepend apriori values for 0'th link non-identifiable variables
            for c in self.delete_cols:
                idf.model.xStd = np.insert(idf.model.xStd, c, 0)
            idf.model.xStd[self.delete_cols] = idf.model.xStdModel[self.delete_cols]

        if idf.opt["showTiming"]:
            print(f"Constrained SDP optimization took {t.interval:.3f} sec.")

    def identifyFeasibleStandardParametersDirect(self, idf: Identification) -> None:
        """Find physically consistent std params using YStd directly (no base projection)."""
        with Timer() as t:
            if idf.opt["verbose"]:
                print("Preparing cvxpy SDP (direct)...")

            YStd = idf.model.YStd
            tau = idf.model.torques_stack

            p_nid = idf.model.non_id
            if idf.opt["useRegressorRegularization"] and len(p_nid):
                l_reg = [(float(idf.base_error) / len(p_nid)) * 1.5] * len(p_nid)
                YStd = np.vstack(
                    (
                        YStd,
                        (np.array(l_reg) * np.identity(idf.model.num_identified_params)[p_nid].T).T,
                    )
                )
                tau = np.concatenate((tau, l_reg * idf.model.xStdModel[p_nid]))

            YStd = np.delete(YStd, self.delete_cols, axis=1)

            Q, R = la.qr(YStd)
            Q1 = Q[:, 0 : idf.model.num_identified_params]
            rho1 = Q1.T.dot(tau)
            R1 = np.array(R[: idf.model.num_identified_params, : idf.model.num_identified_params])

            if idf.opt["useRegressorRegularization"]:
                contactForcesSum = np.concatenate((idf.model.contactForcesSum, np.zeros(len(p_nid))))
            else:
                contactForcesSum = idf.model.contactForcesSum
            contactForces = Q.T.dot(contactForcesSum)

            e_rho1 = rho1 - contactForces - R1 @ self.x

            rho2_norm_sqr = (
                la.norm(idf.model.torques_stack - idf.model.contactForcesSum - idf.model.YBase.dot(idf.model.xBase))
                ** 2
            )

            u = cp.Variable(name="u")
            n = idf.model.num_identified_params
            U_rho = cp.bmat(
                [
                    [_s(u - rho2_norm_sqr), cp.reshape(e_rho1, (1, n), order="C")],
                    [cp.reshape(e_rho1, (n, 1), order="C"), np.eye(n)],
                ]
            )

            all_constraints = [U_rho >> 0] + self.constraints
            prob = cp.Problem(cp.Minimize(u), all_constraints)

            if idf.opt["verbose"]:
                print(f"Solving SDP (direct) with {self.solver_name}...")

            status = sdp_helpers.solve_sdp(
                prob,
                solver_name=self.solver_name,
                solver_opts=self.solver_opts,
            )

            if status in ("optimal", "optimal_inaccurate"):
                print(f"SDP found std solution with {u.value:.2f} squared residual error")
                idf.model.xStd = np.array(self.x.value).flatten()
            else:
                print(Fore.YELLOW + f"SDP solver failed ({status}), keeping a priori parameters" + Fore.RESET)
                idf.model.xStd = idf.model.xStdModel.copy()

            for c in self.delete_cols:
                idf.model.xStd = np.insert(idf.model.xStd, c, 0)
            idf.model.xStd[self.delete_cols] = idf.model.xStdModel[self.delete_cols]

        if idf.opt["showTiming"]:
            print(f"Constrained SDP optimization took {t.interval:.3f} sec.")

    def identifyFeasibleBaseParameters(self, idf: Identification) -> None:
        """Find physically feasible base parameters via SDP.

        Not yet implemented in cvxpy backend — falls back to a priori.
        """
        print(Fore.YELLOW + "identifyFeasibleBaseParameters not yet implemented in cvxpy backend" + Fore.RESET)

    def findFeasibleStdFromFeasibleBase(self, idf: Identification, xBase: np.ndarray) -> None:
        """Find std params closest to CAD consistent with given base params."""
        with Timer() as t:
            idable_params = sorted(list(set(idf.model.identified_params).difference(self.delete_cols)))

            base_constraints = list(self.constraints)  # copy, don't mutate

            # add explicit constraints for each base param = xBase[i]
            # base_deps gives the base params as sympy expressions of std params.
            # evaluate numerically: beta_i = K_row_i @ x_std
            if idf.opt["useBasisProjection"]:
                K = idf.model.Binv
            else:
                K = idf.model.K
            K = np.delete(K, self.delete_cols, axis=1)

            # constrain base params to match step 1 solution (relaxed tolerance to avoid
            # numerical infeasibility — the base params determine torque prediction, so
            # small deviations here have minimal effect on accuracy)
            base_param_tol = float(idf.opt.get("sdpBaseParamTol", 1e-3))
            for i in range(idf.model.num_base_params):
                base_expr = K[i, :] @ self.x
                base_constraints.append(base_expr <= xBase[i] + base_param_tol)
                base_constraints.append(base_expr >= xBase[i] - base_param_tol)

            # minimize distance to CAD
            u = cp.Variable(name="u")
            diff = idf.model.xStdModel[idable_params] - self.x
            n = len(idable_params)
            U_rho = cp.bmat(
                [
                    [_s(u), cp.reshape(diff, (1, n), order="C")],
                    [cp.reshape(diff, (n, 1), order="C"), np.eye(n)],
                ]
            )

            all_constraints = [U_rho >> 0] + base_constraints
            prob = cp.Problem(cp.Minimize(u), all_constraints)

            xStd_before = np.delete(idf.model.xStd, self.delete_cols)
            old_dist = la.norm(idf.model.xStdModel[idable_params] - xStd_before) ** 2

            if idf.opt["checkAPrioriFeasibility"]:
                self.checkFeasibility(idf.model.xStd)

            status = sdp_helpers.solve_sdp(
                prob,
                solver_name=self.solver_name,
                solver_opts=self.solver_opts,
            )

            if status in ("optimal", "optimal_inaccurate"):
                print(f"SDP found std solution with distance {u.value:.2f} from CAD (compared to {old_dist:.2f})")
                idf.model.xStd = np.array(self.x.value).flatten()

                for c in self.delete_cols:
                    idf.model.xStd = np.insert(idf.model.xStd, c, 0)
                idf.model.xStd[self.delete_cols] = idf.model.xStdModel[self.delete_cols]
            else:
                print(f"Could not find closer-to-CAD solution (solver: {status}), keeping previous solution")

        if idf.opt["showTiming"]:
            print(f"Constrained SDP optimization took {t.interval:.3f} sec.")

    def findFeasibleStdFromStd(self, idf: Identification, xStd: np.ndarray) -> np.ndarray:
        """Find closest feasible std solution for some std parameters."""
        idable_params = sorted(list(set(idf.model.identified_params).difference(self.delete_cols)))

        u = cp.Variable(name="u")
        n = len(idable_params)
        diff = xStd - self.x
        U_delta = cp.bmat(
            [
                [_s(u), cp.reshape(diff, (1, n), order="C")],
                [cp.reshape(diff, (n, 1), order="C"), np.eye(n)],
            ]
        )

        all_constraints = [U_delta >> 0] + self.constraints
        prob = cp.Problem(cp.Minimize(u), all_constraints)

        status = sdp_helpers.solve_sdp(
            prob,
            solver_name=self.solver_name,
            solver_opts=self.solver_opts,
        )

        if status in ("optimal", "optimal_inaccurate"):
            print(f"SDP found std solution with {u.value:.2f} error increase from previous solution")
            return np.array(self.x.value).flatten()
        else:
            print(Fore.YELLOW + f"SDP solver failed ({status}), returning input" + Fore.RESET)
            return xStd
