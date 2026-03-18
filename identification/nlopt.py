import os
import sys
from typing import Any

import numpy as np
import numpy.linalg as la
from pyoptsparse import ALPSO, IPOPT, NSGA2, PSQP, SLSQP, Optimization

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from excitation.optimizer import Optimizer


class NLOPT(Optimizer):
    def __init__(self, idf):
        idf.opt["num_dofs"] = idf.model.num_dofs
        super().__init__(idf.opt, idf, idf.model, None)

        if not self.idf.opt["floatingBase"] and self.idf.opt["deleteFixedBase"]:
            # ignore first fixed link
            self.nl = self.model.num_links - 1
            self.start_link = 1
        else:
            # identify all links
            self.nl = self.model.num_links
            self.start_link = 0

        # optimize in projected feasible space (Traversaro, 2016)
        # seems to be buggy / not working
        self.idf.opt["optInFeasibleParamSpace"] = 0

        if self.idf.opt["identifyGravityParamsOnly"]:
            self.per_link = 4
        else:
            self.per_link = 10
        self.start_param = self.start_link * self.per_link

        self.identified_params = sorted(list(set(self.idf.model.identified_params).difference(range(self.start_param))))

        # get COM boundaries
        self.link_hulls = {}
        for i in range(self.start_link, idf.model.num_links):
            link_name = idf.model.linkNames[i]
            box, pos, rot = idf.urdfHelpers.getBoundingBox(
                input_urdf=idf.model.urdf_file,
                old_com=idf.model.xStdModel[i * self.per_link + 1 : i * self.per_link + 4]
                / idf.model.xStdModel[i * self.per_link],
                link_name=link_name,
            )
            self.link_hulls[idf.model.linkNames[i - self.start_link]] = (box, pos, rot)

        self.inner_iter = 0
        self.last_best_u = 1e16
        self.last_best_x: np.ndarray | None = None

        self.param_weights = np.zeros(self.model.num_all_params - self.start_link * self.per_link)
        for p in range(len(self.param_weights)):
            if p < (self.model.num_model_params - self.start_param):
                if p % self.per_link == 0:
                    # masses
                    self.param_weights[p] = 1.0
                elif p % self.per_link in [1, 2, 3]:
                    # com
                    self.param_weights[p] = 1.0
                elif p % self.per_link > 3:
                    # inertia
                    self.param_weights[p] = 1.0
            else:
                # friction
                self.param_weights[p] = 1.0

        # minimize error or param distance
        self.min_est_error = False

    def skew(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def mapStdToConsistent(self, params):
        # map to fully physically consistent parametrization space (Traversaro, 2016)
        # expecting link frame params (without friction)

        S = self.skew

        out = np.zeros(self.nl * 16)
        for l in range(self.nl):
            # mass m is the same
            m = params[l * 10]
            out[l * 16] = m

            # com c is the same, R^3
            c = params[l * 10 + 1 : l * 10 + 3 + 1] / m
            out[l * 16 + 1] = c[0]
            out[l * 16 + 2] = c[1]
            out[l * 16 + 3] = c[2]

            I = self.idf.paramHelpers.invvech(params[l * 10 + 4 : l * 10 + 9 + 1]) + m * S(c).dot(S(c))

            # get rotation matrix from eigenvectors of I
            Q, J, Qt = la.svd(I)  # la.eig(I)

            # J_xx = L_yy + L_zz
            # J_yy = L_xx + L_zz
            # J_zz = L_xx + L_yy
            # solve for L_xx, L_yy, L_zz to get L
            P = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            L = la.inv(P).dot(J)

            # rotation matrix Q R^(3x3) (SO(3)) between body frame and frame of principal
            # axes at COM
            out[l * 16 + 4] = Q[0, 0]
            out[l * 16 + 5] = Q[0, 1]
            out[l * 16 + 6] = Q[0, 2]
            out[l * 16 + 7] = Q[1, 0]
            out[l * 16 + 8] = Q[1, 1]
            out[l * 16 + 9] = Q[1, 2]
            out[l * 16 + 10] = Q[2, 0]
            out[l * 16 + 11] = Q[2, 1]
            out[l * 16 + 12] = Q[2, 2]

            # central second moment of mass along principal axes, R>=^3
            out[l * 16 + 13] = L[0]
            out[l * 16 + 14] = L[1]
            out[l * 16 + 15] = L[2]

        return out

    def mapConsistentToStd(self, params):
        # map from fully physically consistent parametrization space to std (at link frame) (Traversaro, 2016)
        # expecting consistent space params without friction (16*links)
        out = np.zeros(self.nl * 10)

        for l in range(self.nl):
            # mass
            m = params[l * 16]
            out[l * 10] = m

            # mass*COM
            c = params[l * 16 + 1 : l * 16 + 3 + 1]
            out[l * 10 + 1] = m * c[0]
            out[l * 10 + 2] = m * c[1]
            out[l * 10 + 3] = m * c[2]

            # get inertia matrix at frame origin
            S = self.skew
            vech = self.idf.paramHelpers.vech
            Q = params[l * 16 + 4 : l * 16 + 12 + 1].reshape(3, 3)
            L = params[l * 16 + 13 : l * 16 + 15 + 1]
            P = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            Ib = vech(Q.dot(np.diag(P.dot(L))).dot(Q.T) - m * S(c).dot(S(c)))
            out[l * 10 + 4 : l * 10 + 9 + 1] = Ib

        return out

    def minimizeSolToCADStd(self, x):
        """use parameters in std space"""
        fail = 0
        if False in np.isfinite(x):
            fail = 1
            return (0, [], fail)

        # minimize estimation error
        if self.min_est_error:
            tau = self.model.torques_stack
            u = (
                la.norm((tau - self.model.contactForcesSum) - self.model.YStd[:, self.start_param :].dot(x)) ** 2
            )  # / self.idf.data.num_used_samples
        else:
            # minimize distance to CAD
            apriori = self.model.xStdModel[self.identified_params]
            # distance to CAD, weighted/normalized params
            diff = x - apriori  # *self.param_weights
            u = np.square(la.norm(diff))

        cons: list[float] = []
        cons_base: list[float] = [0.0]
        if not self.min_est_error:
            # base equations == xBase as constraints
            if self.idf.opt["useBasisProjection"]:
                cons_base = list(self.model.Binv[:, self.start_param :].dot(x) - self.xBase_feas)
            else:
                cons_base = list(self.model.K[:, self.start_param :].dot(x) - self.xBase_feas)
            cons += cons_base

        cons_inertia = [0.0] * self.nl * 3
        cons_tri = [0.0] * self.nl * 3
        if not self.idf.opt["identifyGravityParamsOnly"]:
            # test for positive definiteness (get params expressed at COM, I_c has to be positive definite)
            x_bary = self.idf.paramHelpers.paramsLink2Bary(x)

            tensors = self.idf.paramHelpers.inertiaTensorFromParams(x_bary)
            min_tol = 1e-10  # allow also slightly negative values to be considered positive
            for l in range(self.nl):
                eigvals = la.eigvals(tensors[l])
                # inertia tensor needs to be positive (semi-)definite
                cons_inertia[l * 3 + 0] = eigvals[0] + min_tol
                cons_inertia[l * 3 + 1] = eigvals[1] + min_tol
                cons_inertia[l * 3 + 2] = eigvals[2] + min_tol

                # triangle inequality of principal axes
                cons_tri[l * 3 + 0] = (eigvals[0] + eigvals[1]) - eigvals[2]
                cons_tri[l * 3 + 1] = (eigvals[0] + eigvals[2]) - eigvals[1]
                cons_tri[l * 3 + 2] = (eigvals[1] + eigvals[2]) - eigvals[0]
            cons += cons_inertia

            if self.use_tri_ineq:
                cons += cons_tri

        # constrain overall mass
        cons_mass_sum: list[float] = [0.0]
        if self.idf.opt["limitOverallMass"]:
            est_mass = np.sum(x[0 : self.model.num_model_params - self.start_link : self.per_link])
            cons_mass_sum[0] = est_mass
            cons += cons_mass_sum

        # constrain com
        if self.idf.opt["restrictCOMtoHull"]:
            cons_com = [0.0] * (self.nl * 6)
            for l in range(self.nl):
                ln = self.idf.model.linkNames[l]
                com = x[l * self.per_link + 1 : l * self.per_link + 4] / x[l * self.per_link]
                cons_com[l * 6 + 0] = com[0] - self.link_hulls[ln][0][0][0]  # lower bound
                cons_com[l * 6 + 1] = com[1] - self.link_hulls[ln][0][0][1]
                cons_com[l * 6 + 2] = com[2] - self.link_hulls[ln][0][0][2]
                cons_com[l * 6 + 3] = self.link_hulls[ln][0][1][0] - com[0]  # upper bound
                cons_com[l * 6 + 4] = self.link_hulls[ln][0][1][1] - com[1]  # upper bound
                cons_com[l * 6 + 5] = self.link_hulls[ln][0][1][2] - com[2]  # upper bound
            cons += cons_com

        c = self.testConstraints(cons)

        if self.idf.opt["showOptimizationGraph"] and self.mpi_rank == 0:
            if c or not getattr(self.opt_prob, "is_gradient", False):
                self.xar.append(self.inner_iter)
                self.yar.append(float(u))
                self.x_constr.append(c)
                self.updateGraph()

        # print some call stats
        if self.idf.opt["verbose"] and self.inner_iter % 100 == 0:
            print(
                f"call #{self.inner_iter}, u={u}, base dist:{np.sum(np.array(cons_base))}, inertia pd:{np.all(np.array(cons_inertia) >= 0)}, tri ineq:{np.all(np.array(cons_tri) >= 0)}, mass sum:{cons_mass_sum}"
            )
        self.inner_iter += 1

        # keep best solution manually
        if u < self.last_best_u and c:
            self.last_best_x = x
            self.last_best_u = float(u)
            if self.idf.opt["verbose"]:
                print("keeping new best solution")

        return (u, cons, fail)

    def minimizeSolToCADFeasible(self, x):
        """minimize in feasible parametrization space"""
        if False in np.isfinite(x):
            fail = 1
            return (0, [], fail)

        fail = 0

        x_std = self.mapConsistentToStd(x)
        x_std = np.concatenate((x_std, x[16 * self.nl :]))  # add friction again

        # minimize estimation error
        if self.min_est_error:
            tau = self.model.torques_stack
            u = (
                la.norm((tau - self.model.contactForcesSum) - self.model.YStd[:, self.start_param :].dot(x_std)) ** 2
            )  # / self.idf.data.num_used_samples
        else:
            # minimize distance to CAD
            apriori = self.model.xStdModel[self.identified_params]

            # distance to CAD, weighted/normalized params
            diff = x_std - apriori
            u = np.square(la.norm(diff))

        cons: list[float] = []
        cons_base: list[float] = [0.0]
        if not self.min_est_error:
            # base equations == xBase as constraints
            if self.idf.opt["useBasisProjection"]:
                cons_base = list(self.model.Binv[:, self.start_param :].dot(x_std) - self.xBase_feas)
            else:
                cons_base = list(self.model.K[:, self.start_param :].dot(x_std) - self.xBase_feas)
            cons += cons_base

        # constrain norm(Q) = 1 (quaternion corresponding to rotation matrix in SO(3))
        cons_det_q = [0.0] * (self.nl)
        cons_ident_q = [0.0] * (self.nl)
        for l in range(self.nl):
            Q = x[l * 16 + 4 : l * 16 + 12 + 1].reshape(3, 3)
            cons_det_q[l] = la.det(Q)
            cons_ident_q[l] = np.sum(Q.T.dot(Q) - np.identity(3))

        cons += cons_det_q
        cons += cons_ident_q
        # constrain overall mass
        cons_mass_sum: list[float] = [0.0]
        if self.idf.opt["limitOverallMass"]:
            est_mass = np.sum(x[0 : self.model.num_model_params - self.start_link : self.per_link + 1])
            cons_mass_sum[0] = est_mass
            cons += cons_mass_sum

        # constrain com
        if self.idf.opt["restrictCOMtoHull"]:
            cons_com = [0.0] * (self.nl * 6)
            for l in range(self.nl):
                ln = self.idf.model.linkNames[l]
                com = x[l * self.per_link + 1 + 1 : l * self.per_link + 1 + 4]
                cons_com[l * 6 + 0] = com[0] - self.link_hulls[ln][0][0][0]  # lower bound
                cons_com[l * 6 + 1] = com[1] - self.link_hulls[ln][0][0][1]
                cons_com[l * 6 + 2] = com[2] - self.link_hulls[ln][0][0][2]
                cons_com[l * 6 + 3] = self.link_hulls[ln][0][1][0] - com[0]  # upper bound
                cons_com[l * 6 + 4] = self.link_hulls[ln][0][1][1] - com[1]  # upper bound
                cons_com[l * 6 + 5] = self.link_hulls[ln][0][1][2] - com[2]  # upper bound
            cons += cons_com

        c = self.testConstraints(cons)

        if self.idf.opt["showOptimizationGraph"] and self.mpi_rank == 0:
            if c or not getattr(self.opt_prob, "is_gradient", False):
                self.xar.append(self.inner_iter)
                self.yar.append(float(u))
                self.x_constr.append(c)
                self.updateGraph()

        # print some iter stats
        if self.idf.opt["verbose"] and self.inner_iter % 100 == 0:
            print(
                f"inner iter {self.inner_iter}, u={u}, base dist:{np.sum(np.array(cons_base))}, det Q:{np.all(np.array(cons_det_q) >= 0)}, ident Q:{np.all(np.abs(cons_ident_q) <= 0.001)}, mass sum:{cons_mass_sum}"
            )
        self.inner_iter += 1

        # keep best solution manually (whatever these solvers are doing...)
        if u < self.last_best_u and c:
            self.last_best_x = x
            self.last_best_u = float(u)
            if self.idf.opt["verbose"]:
                print("keeping new best solution")

        return (u, cons, fail)

    def testConstraints(self, g):
        result = False
        cons_base = g[0 : len(self.xBase_feas)]
        if self.idf.opt["optInFeasibleParamSpace"]:
            iq_start = len(self.xBase_feas) + self.nl
            iq_end = len(self.xBase_feas) + self.nl * 2
            cons_ident_q = g[iq_start:iq_end]
            cons_det_q = g[0 : self.nl]
            cons_mass = g[iq_end]
            mass_con = self.opt_prob.getCon(iq_end)
            cons_com = g[iq_end + 1 :]
            if (
                np.all(np.array(cons_det_q) >= 0)
                and (np.abs(np.sum(np.array(cons_base))) <= 1e-6)
                and np.all(np.abs(cons_ident_q) <= 0.001)
                and np.all(cons_com >= 0.0)
                and cons_mass < mass_con.upper
                and cons_mass > mass_con.lower
            ):
                result = True
        else:
            cons_mass = g[-self.nl * 6 - 1]
            mass_con = self.opt_prob.getCon(len(g) - self.nl * 6 - 1)
            if (
                np.all(np.array(g[len(self.xBase_feas) :]) >= 0)
                and (np.abs(np.sum(np.array(cons_base))) <= 1e-6)
                and cons_mass < mass_con.upper
                and cons_mass > mass_con.lower
            ):
                result = True

        return result

    def addVarsAndConstraints(self, opt, initial_values=None):
        """Add variables and constraints for the optimization problem."""
        if not self.idf.opt["identifyGravityParamsOnly"]:
            # only constrain triangle inequality if it holds true for starting point
            self.use_tri_ineq = False not in self.idf.paramHelpers.checkPhysicalConsistency(self.model.xStd).values()
        else:
            self.use_tri_ineq = False

        if self.use_tri_ineq:
            print("Constraining to triangle inequality")
        else:
            print("Not constraining to triangle inequality")

        self._var_names: list[str] = []

        ## add variables for standard params
        for i in self.identified_params:
            p = self.model.param_syms[i]
            if i % self.per_link == 0 and i < self.model.num_model_params:  # mass
                self._var_names.append(str(p))
                opt.addVar(str(p), lower=0.1, upper=np.inf)
            else:
                if not self.idf.opt["identifyGravityParamsOnly"] and i >= self.model.num_model_params:  # friction
                    # TODO: remove friction from optimization (they shouldn't change and are not dependent in base params)
                    self._var_names.append(str(p))
                    opt.addVar(str(p), lower=0.0, upper=np.inf)
                else:
                    if i % self.per_link not in [1, 2, 3] and self.idf.opt["optInFeasibleParamSpace"]:
                        pass
                    else:
                        # inertia: unbounded variable
                        self._var_names.append(str(p))
                        opt.addVar(str(p), lower=-np.inf, upper=np.inf)

            # add variables for feasible parametrization space
            if self.idf.opt["optInFeasibleParamSpace"] and not self.idf.opt["identifyGravityParamsOnly"]:
                if i % self.per_link == 4 and i < self.model.num_model_params:
                    l = i // 10
                    # Q as rotation matrix in R^3x3
                    for qi in range(9):
                        name = f"q{qi}_{l}"
                        self._var_names.append(name)
                        opt.addVar(name, lower=-10, upper=10)
                    # L ln R^3+
                    for li in range(3):
                        name = f"l{li}_{l}"
                        self._var_names.append(name)
                        opt.addVar(name, lower=0, upper=np.inf)

        ## constraints

        # add constraints for projection to feasible base parameters (equality constraints before
        # inequality)
        if not self.min_est_error:
            opt.addConGroup("xbase", len(self.xBase_feas), lower=0.0, upper=0.0)

        if not self.idf.opt["optInFeasibleParamSpace"]:
            # add naive consistency constraints (inertia positive definite, triangel inequality)
            if not self.idf.opt["identifyGravityParamsOnly"]:
                opt.addConGroup("inertia", 3 * self.nl, lower=0.0, upper=100)
                if self.use_tri_ineq:
                    opt.addConGroup("tri_ineq", 3 * self.nl, lower=0.0, upper=100)
        else:
            opt.addConGroup("det R > 0", self.nl, lower=0.0, upper=np.inf)
            opt.addConGroup("R.T*R = I", self.nl, lower=-0.001, upper=0.001)

        # constraints for overall mass
        if self.idf.opt["limitOverallMass"]:
            lower = upper = self.idf.opt["limitMassVal"]
            if not self.idf.opt["floatingBase"] and self.idf.opt["deleteFixedBase"]:
                lower = upper = self.idf.opt["limitMassVal"] - self.idf.model.xStd[0]

            opt.addConGroup(
                "mass sum",
                1,
                lower=lower * (1 - self.idf.opt["limitMassRange"]),
                upper=upper * (1 + self.idf.opt["limitMassRange"]),
            )

        # constraints for COM in enclosing hull
        if self.idf.opt["restrictCOMtoHull"]:
            opt.addConGroup("com", 6 * self.nl, lower=0.0, upper=100)

    def _objFuncWrapperStd(self, xdict: dict) -> tuple[dict, bool]:
        """Wrapper for minimizeSolToCADStd to convert pyOptSparse dict API."""
        x = np.array([xdict[name] for name in self._var_names]).flatten()
        u, cons, fail = self.minimizeSolToCADStd(x)
        funcs: dict[str, Any] = {"u": u}
        # distribute constraints into their named groups
        idx = 0
        if not self.min_est_error:
            n = len(self.xBase_feas)
            funcs["xbase"] = np.array(cons[idx : idx + n])
            idx += n
        if not self.idf.opt["optInFeasibleParamSpace"]:
            if not self.idf.opt["identifyGravityParamsOnly"]:
                n = 3 * self.nl
                funcs["inertia"] = np.array(cons[idx : idx + n])
                idx += n
                if self.use_tri_ineq:
                    funcs["tri_ineq"] = np.array(cons[idx : idx + n])
                    idx += n
        if self.idf.opt["limitOverallMass"]:
            funcs["mass sum"] = np.array(cons[idx : idx + 1])
            idx += 1
        if self.idf.opt["restrictCOMtoHull"]:
            n = 6 * self.nl
            funcs["com"] = np.array(cons[idx : idx + n])
            idx += n
        return funcs, bool(fail)

    def _objFuncWrapperFeasible(self, xdict: dict) -> tuple[dict, bool]:
        """Wrapper for minimizeSolToCADFeasible to convert pyOptSparse dict API."""
        x = np.array([xdict[name] for name in self._var_names]).flatten()
        u, cons, fail = self.minimizeSolToCADFeasible(x)
        funcs: dict[str, Any] = {"u": u}
        # distribute constraints into their named groups
        idx = 0
        if not self.min_est_error:
            n = len(self.xBase_feas)
            funcs["xbase"] = np.array(cons[idx : idx + n])
            idx += n
        funcs["det R > 0"] = np.array(cons[idx : idx + self.nl])
        idx += self.nl
        funcs["R.T*R = I"] = np.array(cons[idx : idx + self.nl])
        idx += self.nl
        if self.idf.opt["limitOverallMass"]:
            funcs["mass sum"] = np.array(cons[idx : idx + 1])
            idx += 1
        if self.idf.opt["restrictCOMtoHull"]:
            n = 6 * self.nl
            funcs["com"] = np.array(cons[idx : idx + n])
            idx += n
        return funcs, bool(fail)

    def identifyFeasibleStdFromFeasibleBase(self, xBase):
        self.xBase_feas = xBase

        # formulate problem as objective function
        if self.idf.opt["optInFeasibleParamSpace"]:
            opt = Optimization("Constrained OLS", self._objFuncWrapperFeasible)
        else:
            opt = Optimization("Constrained OLS", self._objFuncWrapperStd)
        opt.addObj("u")

        """
        x_cons = self.mapStdToConsistent(self.idf.model.xStd[self.start_param:self.idf.model.num_model_params])
        test = self.mapConsistentToStd(x_cons)
        print(test - self.idf.model.xStd[self.start_param:self.idf.model.num_model_params])
        """

        self.addVarsAndConstraints(opt)

        # set previous sol as starting point (as primal should be already within constraints for
        # most solvers to perform well)
        if self.idf.opt["optInFeasibleParamSpace"]:
            x_cons = self.mapStdToConsistent(self.idf.model.xStd[self.start_param : self.idf.model.num_model_params])
            dvs = {}
            for i, name in enumerate(self._var_names):
                if i < len(x_cons):
                    dvs[name] = x_cons[i]
                else:
                    j = i - len(x_cons)
                    dvs[name] = self.model.xStd[self.idf.model.num_model_params + j]
            opt.setDVs(dvs)
        else:
            dvs = {}
            for i, name in enumerate(self._var_names):
                dvs[name] = self.model.xStd[i + self.start_link * self.per_link]
            opt.setDVs(dvs)

        if self.idf.opt["verbose"]:
            print(opt)

        if self.idf.opt["nlOptSolver"] == "IPOPT":
            # not necessarily deterministic
            if self.idf.opt["verbose"]:
                print("Using IPOPT")
            solver = IPOPT()
            # solver.setOption('linear_solver', 'ma97')  #mumps or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
            # for details, see http://www.gams.com/latest/docs/solvers/ipopt/index.html#IPOPTlinear_solver
            solver.setOption("max_iter", self.idf.opt["nlOptMaxIterations"])
            solver.setOption("print_level", 3)  # 0 none ... 5 max

            # don't start too far away from inital values (boundaries push even if starting inside feasible set)
            solver.setOption("bound_push", 0.0000001)
            solver.setOption("bound_frac", 0.0000001)
            # don't relax bounds
            solver.setOption("bound_relax_factor", 0.0)  # 1e-16)

        elif self.idf.opt["nlOptSolver"] == "SLSQP":
            # solve optimization problem
            if self.idf.opt["verbose"]:
                print("Using SLSQP")
            solver = SLSQP()
            solver.setOption("MAXIT", self.idf.opt["nlOptMaxIterations"])
            if self.idf.opt["verbose"]:
                solver.setOption("IPRINT", 0)

        elif self.idf.opt["nlOptSolver"] == "PSQP":
            # solve optimization problem
            if self.idf.opt["verbose"]:
                print("Using PSQP")
            solver = PSQP()
            solver.setOption("MIT", self.idf.opt["nlOptMaxIterations"])
            if self.idf.opt["verbose"]:
                solver.setOption("IPRINT", 0)

        elif self.idf.opt["nlOptSolver"] == "ALPSO":
            if self.idf.opt["verbose"]:
                print("Using ALPSO")
            solver = ALPSO()
            solver.setOption("stopCriteria", 0)
            solver.setOption("dynInnerIter", 1)  # dynamic inner iter number
            solver.setOption("maxInnerIter", 5)
            solver.setOption("maxOuterIter", self.idf.opt["nlOptMaxIterations"])
            solver.setOption("printInnerIters", 0)
            solver.setOption("printOuterIters", 0)
            solver.setOption("SwarmSize", 100)
            solver.setOption("xinit", 1)

        elif self.idf.opt["nlOptSolver"] == "NSGA2":
            if self.idf.opt["verbose"]:
                print("Using NSGA2")
            solver = NSGA2()
            solver.setOption("PopSize", 100)  # Population Size (a Multiple of 4)
            solver.setOption("maxGen", self.config["nlOptMaxIterations"])  # Maximum Number of Generations
            solver.setOption("PrintOut", 0)  # Flag to Turn On Output to files (0-None, 1-Subset, 2-All)
            solver.setOption("xinit", 1)  # Use Initial Solution Flag (0 - random population, 1 - use given solution)
            # solver.serion('seed', sr.random())   # Random Number Seed 0..1 (0 - Auto based on time clock)
            # pCross_real    0.6     Probability of Crossover of Real Variable (0.6-1.0)
            solver.setOption("pMut_real", 0.5)  # Probablity of Mutation of Real Variables (1/nreal)
            # eta_c  10.0    # Distribution Index for Crossover (5-20) must be > 0
            # eta_m  20.0    # Distribution Index for Mutation (5-50) must be > 0
            # pCross_bin     0.0     # Probability of Crossover of Binary Variable (0.6-1.0)
            # pMut_real      0.0     # Probability of Mutation of Binary Variables (1/nbits)
        else:
            raise RuntimeError(f"Unknown nlOptSolver: {self.idf.opt['nlOptSolver']!r}")

        self.opt_prob = opt
        # run optimizer (use FD for gradient-based solvers)
        if self.idf.opt["nlOptSolver"] in ["IPOPT", "SLSQP", "PSQP"]:
            solver(opt, sens="FD")
        else:
            solver(opt)

        # use manually tracked best solution (solver may not keep the best feasible one)
        if self.last_best_x is None:
            self.last_best_x = self.model.xStd[self.start_param :]

        if self.idf.opt["verbose"]:
            print(opt)

        if self.idf.opt["optInFeasibleParamSpace"] and len(self.last_best_x) > len(self.model.xStd[self.start_param :]):
            # we get consistent parameterized params as solution
            x_std = self.mapConsistentToStd(self.last_best_x)
            self.model.xStd[self.start_param : self.idf.model.num_model_params] = x_std
        else:
            # we get std vars as solution
            self.model.xStd[self.start_param :] = self.last_best_x

    def identifyFeasibleStandardParameters(self):
        self.min_est_error = True
        self.identifyFeasibleStdFromFeasibleBase(xBase=[])
