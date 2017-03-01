from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range

import numpy as np
import numpy.linalg as la
import pyOpt

from .quaternion import Quaternion

from IPython import embed

class NLOPT(object):
    def __init__(self, idf):
        self.idf = idf
        self.model = idf.model

        if not self.idf.opt['floatingBase'] and self.idf.opt['deleteFixedBase']:
            # ignore first fixed link
            self.nl = self.model.num_links - 1
            self.start_link = 1
        else:
            # identify all links
            self.nl = self.model.num_links
            self.start_link = 0

        self.idf.opt['optInFeasibleParamSpace'] = 1

        if self.idf.opt['identifyGravityParamsOnly']:
            self.per_link = 4
        else:
            self.per_link = 10
        self.start_param = self.start_link * self.per_link

        self.identified_params = sorted(list(set(self.idf.model.identified_params).difference(range(self.start_param))))

        #get COM boundaries
        self.link_hulls = np.zeros((self.nl, 3, 2))
        for i in range(self.start_link, idf.model.num_links):
            self.link_hulls[i - self.start_link] = np.array(
                                idf.urdfHelpers.getBoundingBox(
                                    input_urdf = idf.model.urdf_file,
                                    old_com = idf.model.xStdModel[i*self.per_link+1:i*self.per_link+4] / idf.model.xStdModel[i*self.per_link],
                                    link_nr = i
                                )
                            )
        self.inner_iter = 0
        self.last_best_u = 1e16
        self.last_best_x = None

        self.param_weights = np.zeros(self.model.num_all_params - self.start_link*self.per_link)
        for p in range(len(self.param_weights)):
            if p < (self.model.num_model_params-self.start_param):
                if p % self.per_link == 0:
                    #masses
                    self.param_weights[p] = 1.0
                elif p % self.per_link in [1,2,3]:
                    #com
                    self.param_weights[p] = 1.0
                elif p % self.per_link > 3:
                    #inertia
                    self.param_weights[p] = 1.0
            else:
                #friction
                self.param_weights[p] = 1.0

    def skew(self, v):
        return np.array([ [     0, -v[2],  v[1] ],
                          [  v[2],     0, -v[0] ],
                          [ -v[1],  v[0],   0 ] ])

    def mapStdToConsistent(self, params):
        # map to fully physically consistent parametrization space (Traversaro, 2016)
        # expecting link frame params (without friction)

        S = self.skew

        out = np.zeros(self.nl*16)
        for l in range(self.nl):
            # mass m is the same
            m = params[l*10]
            out[l*16] = m

            # com c is the same, R^3
            c = params[l*10+1:l*10+3+1] / m
            out[l*16+1] = c[0]
            out[l*16+2] = c[1]
            out[l*16+3] = c[2]

            I = self.idf.paramHelpers.invvech(params[l*10+4:l*10+9+1]) + m*S(c).dot(S(c))

            # get rotation matrix from eigenvectors of I
            Q, J, Qt = la.svd(I) #la.eig(I)

            #J_xx = L_yy + L_zz
            #J_yy = L_xx + L_zz
            #J_zz = L_xx + L_yy
            #solve for L_xx, L_yy, L_zz to get L
            P = np.array([[0,1,1],[1,0,1],[1,1,0]])
            L = la.inv(P).dot(J)

            # rotation matrix Q R^(3x3) (SO(3)) between body frame and frame of principal
            # axes at COM
            out[l*16+4] = Q[0, 0]
            out[l*16+5] = Q[0, 1]
            out[l*16+6] = Q[0, 2]
            out[l*16+7] = Q[1, 0]
            out[l*16+8] = Q[1, 1]
            out[l*16+9] = Q[1, 2]
            out[l*16+10] = Q[2, 0]
            out[l*16+11] = Q[2, 1]
            out[l*16+12] = Q[2, 2]

            # central second moment of mass along principal axes, R>=^3
            out[l*16+13] = L[0]
            out[l*16+14] = L[1]
            out[l*16+15] = L[2]

        return out

    def mapConsistentToStd(self, params):
        # map from fully physically consistent parametrization space to std (at link frame) (Traversaro, 2016)
        # expecting consistent space params without friction (16*links)
        out = np.zeros(self.nl*10)

        for l in range(self.nl):
            #mass
            m = params[l*16]
            out[l*10] = m

            #mass*COM
            c = params[l*16+1:l*16+3+1]
            out[l*10+1] = m*c[0]
            out[l*10+2] = m*c[1]
            out[l*10+3] = m*c[2]

            #get inertia matrix at frame origin
            S = self.skew
            vech = self.idf.paramHelpers.vech
            Q = params[l*16+4:l*16+12+1].reshape(3,3)
            L = params[l*16+13:l*16+15+1]
            P = np.array([[0,1,1],[1,0,1],[1,1,0]])
            Ib = vech(Q.dot(np.diag( P.dot(L) )).dot(Q.T) - m*S(c).dot(S(c)))
            out[l*10+4:l*10+9+1] = Ib

        return out

    def minimizeSolToCADStd(self, x):
        """ use parameters in std space """
        fail = 0
        if False in np.isfinite(x):
            fail = 1
            return (0, [], fail)

        #minimize estimation error
        #tau = self.model.torques_stack
        #u = la.norm( (tau + self.model.contactForcesSum) - self.model.YStd[:, self.start_param:].dot(x_std))**2 #/ self.idf.data.num_used_samples

        #minimize distance to CAD
        apriori = self.model.xStdModel[self.identified_params]
        #distance to CAD, weighted/normalized params
        diff = (x - apriori) #*self.param_weights
        u = np.square(la.norm(diff))

        cons = []
        # base equations == xBase as constraints
        if self.idf.opt['useBasisProjection']:
            cons_base = list((self.model.Binv[:, self.start_param:].dot(x) - self.xBase_feas))
        else:
            cons_base = list((self.model.K[:, self.start_param:].dot(x) - self.xBase_feas))
        cons += cons_base

        cons_inertia = [0.0]*self.nl*3
        cons_tri = [0.0]*self.nl*3
        if not self.idf.opt['identifyGravityParamsOnly']:
            #test for positive definiteness (get params expressed at COM, I_c has to be positive definite)
            x_bary = self.idf.paramHelpers.paramsLink2Bary(x)

            tensors = self.idf.paramHelpers.inertiaTensorFromParams(x_bary)
            min_tol = 1e-10   # allow also slightly negative values to be considered positive
            for l in range(self.nl):
                eigvals = la.eigvals(tensors[l])
                # inertia tensor needs to be positive (semi-)definite
                cons_inertia[l*3+0] = eigvals[0] + min_tol
                cons_inertia[l*3+1] = eigvals[1] + min_tol
                cons_inertia[l*3+2] = eigvals[2] + min_tol

                # triangle inequality of principal axes
                cons_tri[l*3+0] = (eigvals[0] + eigvals[1]) - eigvals[2]
                cons_tri[l*3+1] = (eigvals[0] + eigvals[2]) - eigvals[1]
                cons_tri[l*3+2] = (eigvals[1] + eigvals[2]) - eigvals[0]
            cons += cons_inertia

            if self.use_tri_ineq:
                cons += cons_tri

        # constrain overall mass
        if self.idf.opt['limitOverallMass']:
            cons_mass_sum = [0.0]*1
            est_mass = np.sum(x[0:self.model.num_model_params-self.start_link:self.per_link])
            cons_mass_sum[0] = est_mass
            cons += cons_mass_sum

        # constrain com
        if self.idf.opt['restrictCOMtoHull']:
            cons_com = [0.0]*(self.nl*6)
            for l in range(self.nl):
                com = x[l*self.per_link+1:l*self.per_link+4]/x[l*self.per_link]
                cons_com[l*6+0] = com[0] - self.link_hulls[l][0][0]  #lower bound
                cons_com[l*6+1] = com[1] - self.link_hulls[l][1][0]
                cons_com[l*6+2] = com[2] - self.link_hulls[l][2][0]
                cons_com[l*6+0] = self.link_hulls[l][0][1] - com[0]  #upper bound
                cons_com[l*6+1] = self.link_hulls[l][1][1] - com[1]  #upper bound
                cons_com[l*6+2] = self.link_hulls[l][2][1] - com[2]  #upper bound
            cons += cons_com

        # print some iter stats
        if self.idf.opt['verbose'] and self.inner_iter % 100 == 0:
            print("inner iter {}, u={}, base dist:{}, inertia pd:{}, tri ineq:{}".format(self.inner_iter, u,
                np.sum(np.array(cons_base)), np.all(np.array(cons_inertia) >= 0), np.all(np.array(cons_tri) >= 0)))
        self.inner_iter += 1

        #keep best solution manually (whatever these solvers are doing...)
        #TODO: check properly if all constraints are met (create test function)
        if u < self.last_best_u and (np.abs(np.sum(np.array(cons_base))) - 1e-6) <= 0 and \
                np.all(np.array(cons_inertia) >= 0):
            if (self.use_tri_ineq and np.all(np.array(cons_tri) >= 0)) or not self.use_tri_ineq:
                self.last_best_x = x
                self.last_best_u = u
                if self.idf.opt['verbose']:
                    print("keeping new best solution")

        return (u, cons, fail)

    def minimizeSolToCADFeasible(self, x):
        """ minimize in feasible parametrization space """
        if False in np.isfinite(x):
            fail = 1
            return (0, [], fail)

        fail = 0

        #minimize distance to CAD
        apriori = self.model.xStdModel[self.identified_params]

        #distance to CAD, weighted/normalized params
        x_std = self.mapConsistentToStd(x)
        x_std = np.concatenate((x_std, x[16*self.nl:]))  #add friction again
        diff = (x_std - apriori)
        u = np.square(la.norm(diff))

        #minimize estimation error
        #tau = self.model.torques_stack
        #u = la.norm( (tau + self.model.contactForcesSum) - self.model.YStd[:, self.start_param:].dot(x_std))**2 #/ self.idf.data.num_used_samples

        cons = []
        # base equations == xBase as constraints
        if self.idf.opt['useBasisProjection']:
            cons_base = list((self.model.Binv[:, self.start_param:].dot(x_std) - self.xBase_feas))
        else:
            cons_base = list((self.model.K[:, self.start_param:].dot(x_std) - self.xBase_feas))
        cons += cons_base

        # constrain norm(Q) = 1 (quaternion corresponding to rotation matrix in SO(3))
        cons_det_q = [0.0]*(self.nl)
        cons_ident_q = [0.0]*(self.nl)
        for l in range(self.nl):
            Q = x[l*16+4:l*16+12+1].reshape(3,3)
            cons_det_q[l] = la.det(Q)
            cons_ident_q[l] = np.sum(Q.T.dot(Q) - np.identity(3))

        cons += cons_det_q
        cons += cons_ident_q
        # constrain overall mass
        if self.idf.opt['limitOverallMass']:
            cons_mass_sum = [0.0]*1
            est_mass = np.sum(x[0:self.model.num_model_params-self.start_link:self.per_link+1])
            cons_mass_sum[0] = est_mass
            cons += cons_mass_sum

        # constrain com
        if self.idf.opt['restrictCOMtoHull']:
            cons_com = [0.0]*(self.nl*6)
            for l in range(self.nl):
                com = x[l*self.per_link+1+1:l*self.per_link+1+4]
                cons_com[l*6+0] = com[0] - self.link_hulls[l][0][0]  #lower bound
                cons_com[l*6+1] = com[1] - self.link_hulls[l][1][0]
                cons_com[l*6+2] = com[2] - self.link_hulls[l][2][0]
                cons_com[l*6+0] = self.link_hulls[l][0][1] - com[0]  #upper bound
                cons_com[l*6+1] = self.link_hulls[l][1][1] - com[1]  #upper bound
                cons_com[l*6+2] = self.link_hulls[l][2][1] - com[2]  #upper bound
            cons += cons_com

        # print some iter stats
        if self.idf.opt['verbose'] and self.inner_iter % 100 == 0:
            print("inner iter {}, u={}, base dist:{}, det Q:{}, ident Q:{}".format(self.inner_iter, u,
                np.sum(np.array(cons_base)), np.all(np.array(cons_det_q) >= 0), np.all(np.abs(cons_ident_q) <= 0.001)))
        self.inner_iter += 1

        #keep best solution manually (whatever these solvers are doing...)
        if u < self.last_best_u and (np.abs(np.sum(np.array(cons_base))) - 1e-6) <= 0 and \
            np.all(np.array(cons_det_q) >= 0) and np.all(np.abs(cons_ident_q) <= 0.001):
                self.last_best_x = x
                self.last_best_u = u
                if self.idf.opt['verbose']:
                    print("keeping new best solution")

        return (u, cons, fail)

    def addVarsAndConstraints(self, opt):
        if not self.idf.opt['identifyGravityParamsOnly']:
            # only constrain triangle inequality if it holds true for starting point
            self.use_tri_ineq = not (False in self.idf.paramHelpers.checkPhysicalConsistency(self.model.xStd).values())
        else:
            self.use_tri_ineq = False

        ## add variables for standard params
        for i in self.identified_params:
            p = self.model.param_syms[i]
            if i % self.per_link == 0 and i < self.model.num_model_params:   #mass
                opt.addVar(str(p), type="c", lower=0.1, upper=np.inf)
            else:
                if not self.idf.opt['identifyGravityParamsOnly'] and i >= self.model.num_model_params:   #friction
                    #TODO: remove friction from optimization (they shouldn't change and are not dependent in base params)
                    opt.addVar(str(p), type="c", lower=0.0, upper=np.inf)
                else:
                    if i % self.per_link not in [1,2,3] and self.idf.opt['optInFeasibleParamSpace']:
                        pass
                    else:
                        #inertia: unbounded variable
                        opt.addVar(str(p), type="c", lower=-np.inf, upper=np.inf)

            # add variables for feasible parametrization space
            if self.idf.opt['optInFeasibleParamSpace'] and not self.idf.opt['identifyGravityParamsOnly']:
                if i % self.per_link == 4 and i < self.model.num_model_params:
                    l = i // 10
                    # Q as rotation matrix in R^3x3
                    opt.addVar('q0_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q1_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q2_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q3_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q4_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q5_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q6_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q7_{}'.format(l), type="c", lower=-10, upper=10)
                    opt.addVar('q8_{}'.format(l), type="c", lower=-10, upper=10)
                    # L ln R^3+
                    opt.addVar('l0_{}'.format(l), type="c", lower=0, upper=np.inf)
                    opt.addVar('l1_{}'.format(l), type="c", lower=0, upper=np.inf)
                    opt.addVar('l2_{}'.format(l), type="c", lower=0, upper=np.inf)

        ## constraints

        # add constraints for projection to feasible base parameters (equality constraints before
        # inequality)
        opt.addConGroup('xbase', len(self.xBase_feas), 'e', equal=0.0)

        if not self.idf.opt['optInFeasibleParamSpace']:
            # add naive consistency constraints (inertia positive definite, triangel inequality)
            if not self.idf.opt['identifyGravityParamsOnly']:
                opt.addConGroup('inertia', 3*self.nl, 'i', lower=0.0, upper=100)
                if self.use_tri_ineq:
                    opt.addConGroup('tri_ineq', 3*self.nl, 'i', lower=0.0, upper=100)
        else:
            opt.addConGroup('det R > 0', self.nl, 'i', lower=0.0, upper=np.inf)
            opt.addConGroup('R.T*R = I', self.nl, 'i', lower=-0.001, upper=0.001) #equal=0.0)

        # constraints for overall mass
        if self.idf.opt['limitOverallMass']:
            opt.addConGroup('mass sum', 1, 'i',
                            lower=self.idf.opt['limitMassVal']*(1-self.idf.opt['limitMassRange']),
                            upper=self.idf.opt['limitMassVal']*(1+self.idf.opt['limitMassRange']))

        # constraints for COM in enclosing hull
        if self.idf.opt['restrictCOMtoHull']:
            opt.addConGroup('com', 6*self.nl, 'i', lower=0.0, upper=100)


    def identifyFeasibleStdFromFeasibleBase(self, xBase):
        self.xBase_feas = xBase

        # formulate problem as objective function
        if self.idf.opt['optInFeasibleParamSpace']:
            opt = pyOpt.Optimization('Constrained OLS', self.minimizeSolToCADFeasible)
        else:
            opt = pyOpt.Optimization('Constrained OLS', self.minimizeSolToCADStd)
        opt.addObj('u')

        '''
        x_cons = self.mapStdToConsistent(self.idf.model.xStd[self.start_param:self.idf.model.num_model_params])
        test = self.mapConsistentToStd(x_cons)
        print(test - self.idf.model.xStd[self.start_param:self.idf.model.num_model_params])
        '''

        self.addVarsAndConstraints(opt)

        # set previous sol as starting point (as primal should be already within constraints for
        # most solvers to perform well)
        if self.idf.opt['optInFeasibleParamSpace']:
            x_cons = self.mapStdToConsistent(self.idf.model.xStd[self.start_param:self.idf.model.num_model_params])
            for i in range(len(opt.getVarSet())):
                #atm, we have 16*no_link + n_dof vars
                if i < len(x_cons):
                    opt.getVar(i).value = x_cons[i]
                else:
                    j = i - len(x_cons)
                    opt.getVar(i).value = self.model.xStd[self.idf.model.num_model_params + j]
        else:
            for i in range(len(opt.getVarSet())):
                opt.getVar(i).value = self.model.xStd[i+self.start_link*self.per_link]

        if self.idf.opt['verbose']:
            print(opt)

        if self.idf.opt['useIPOPTforNL']:
            # not necessarily deterministic
            if self.idf.opt['verbose']:
                print('Using IPOPT to get closer to a priori')
            solver = pyOpt.IPOPT()
            solver.setOption('linear_solver', 'ma97')  #mumps or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
            #for details, see http://www.gams.com/latest/docs/solvers/ipopt/index.html#IPOPTlinear_solver
            solver.setOption('max_iter', self.idf.opt['nlOptMaxIterations'])
            solver.setOption('print_level', 3)  #0 none ... 5 max

            #don't start too far away from inital values (boundaries push even if starting inside feasible set)
            solver.setOption('bound_push', 0.0000001)
            solver.setOption('bound_frac', 0.0000001)
            #don't relax bounds
            solver.setOption('bound_relax_factor', 0.0) #1e-16)
        else:
            # solve optimization problem
            '''
            if self.idf.opt['verbose']:
                print('Using PSQP to get closer to a priori')
            solver = pyOpt.PSQP(disp_opts=True)
            solver.setOption('MIT', self.idf.opt['nlOptMaxIterations'])
            solver.setOption('TOLC', 1e-16)
            if self.idf.opt['verbose']:
                solver.setOption('IPRINT', 1)
            '''
            if self.idf.opt['verbose']:
                print('Using ALPSO to get closer to a priori')
            solver = pyOpt.ALPSO(disp_opts=True)
            solver.setOption('stopCriteria', 0)
            solver.setOption('maxInnerIter', 3)
            solver.setOption('maxOuterIter', self.idf.opt['nlOptMaxIterations'])
            solver.setOption('printInnerIters', 1)
            solver.setOption('printOuterIters', 1)
            solver.setOption('SwarmSize', 30)
            solver.setOption('xinit', 1)

        solver(opt)         #run optimization

        # set best solution again (is often different than final solver solution)
        if self.last_best_x is not None:
            for i in range(len(opt.getVarSet())):
                opt.getVar(i).value = self.last_best_x[i]
        else:
            self.last_best_x = self.model.xStd[self.start_param:]

        sol = opt.solution(0)
        if self.idf.opt['verbose']:
            print(sol)

        if self.idf.opt['optInFeasibleParamSpace'] and len(self.last_best_x) > len(self.model.xStd[self.start_param:]):
            # we get consistent parameterized params as solution
            x_std = self.mapConsistentToStd(self.last_best_x)
            self.model.xStd[self.start_param:self.idf.model.num_model_params] = x_std
        else:
            # we get std vars as solution
            self.model.xStd[self.start_param:] = self.last_best_x

