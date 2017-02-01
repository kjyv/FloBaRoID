from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range

import sys
import numpy as np
import numpy.linalg as la
import scipy
import scipy.linalg as sla

import pyOpt

from IPython import embed

class NLOPT(object):
    def __init__(self, idf):
        self.idf = idf
        self.model = idf.model

        #get COM boundaries
        self.link_hulls = np.zeros((idf.model.N_LINKS, 3, 2))
        for i in range(idf.model.N_LINKS):
            self.link_hulls[i] = np.array(
                                idf.urdfHelpers.getBoundingBox(
                                    input_urdf = idf.model.urdf_file,
                                    old_com = idf.model.xStdModel[i*10+1:i*10+4],
                                    link_nr = i
                                )
                            )
        self.inner_iter = 0
        self.last_best_u = 1e16

    def minimizeBaseOLSError(self, x):
        fail = 0

        # estimation error
        #tau = self.model.torques_stack
        #u = la.norm(tau - self.model.YStd.dot(x))**2 / self.idf.data.num_used_samples

        #distance to CAD
        u = la.norm(x - self.model.xStdModel)**2

        cons = []
        # base equations == xBase as constraints
        cons_base = list((self.model.Binv.dot(x) - self.xBase_feas)*10)
        cons += cons_base

        x_bary = self.idf.paramHelpers.paramsLink2Bary(x)

        #test for positive definiteness (too naive?)
        tensors = self.idf.paramHelpers.inertiaTensorFromParams(x_bary)
        cons_inertia = [0.0]*self.model.N_LINKS*3
        cons_tri = [0.0]*self.model.N_LINKS*3
        is_pd = [False]*self.model.N_LINKS
        for l in range(self.model.N_LINKS):
            eigvals = la.eigvals(tensors[l])
            # inertia tensor needs to be positive (semi-)definite
            cons_inertia[l*3+0] = eigvals[0]
            cons_inertia[l*3+1] = eigvals[1]
            cons_inertia[l*3+2] = eigvals[2]

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
            est_mass = np.sum(x[0:self.model.num_model_params:10])
            cons_mass_sum[0] = est_mass
            cons += cons_mass_sum

        # constrain com
        if self.idf.opt['restrictCOMtoHull']:
            cons_com = [0.0]*self.model.N_LINKS*6
            for l in range(self.model.N_LINKS):
                cons_com[l*6+0] = x_bary[l*10+1] - self.link_hulls[l][0][0]  #lower bound
                cons_com[l*6+1] = x_bary[l*10+2] - self.link_hulls[l][1][0]
                cons_com[l*6+2] = x_bary[l*10+3] - self.link_hulls[l][2][0]
                cons_com[l*6+3] = self.link_hulls[l][0][1] - x_bary[l*10+1]   #upper bound
                cons_com[l*6+4] = self.link_hulls[l][1][1] - x_bary[l*10+2]
                cons_com[l*6+5] = self.link_hulls[l][2][1] - x_bary[l*10+3]
            cons += cons_com

        # print some iter stats
        if self.inner_iter % 100 == 0:
            print("inner iter {}, u={}, base dist:{}, inertia pd:{}, tri ineq:{}".format(self.inner_iter, u,
                np.sum(np.array(cons_base)), np.all(np.array(cons_inertia) >= 0), np.all(np.array(cons_tri) >= 0)))
        self.inner_iter += 1

        #keep best solution manually (whatever these solvers are doing...)
        #TODO: check that all constraints are met (create test function)
        if u < self.last_best_u and (np.abs(np.sum(np.array(cons_base))) - 1e9) <= 0 and \
                np.all(np.array(cons_inertia) >= 0):
            if (self.use_tri_ineq and np.all(np.array(cons_tri) >= 0)) or not self.use_tri_ineq:
                self.last_best_x = x
                self.last_best_u = u
                print("keeping new best solution")

        return (u, cons, fail)

    def addVarsAndConstraints(self, opt):
        #add all std vars and some constraints
        for i in range(len(self.model.param_syms)):
            p = self.model.param_syms[i]
            if i % 10 == 0 and i < self.model.num_model_params:   #mass
                opt.addVar(str(p), type="c", lower=0.1, upper=20.0)
                #self.idf.sdp.constr_per_param[i].append('m>0')
            elif i >= self.model.num_model_params:   #friction
                opt.addVar(str(p), type="c", lower=0.0, upper=1000)
                #self.idf.sdp.constr_per_param[i].append('f>0')
            #elif i % 10 in [4,7,9]:     #inertia elements on principal axes
            #    opt.addVar(str(p), type="c", lower=0.0)
            #    self.idf.sdp.constr_per_param[i].append('i>0')
            else:
                #(for now) unconstrained variable (could be a bit narrow)
                opt.addVar(str(p), type="c", lower=-100, upper=100)

    def identifyFeasibleStdFromFeasibleBase(self, xBase):
        self.xBase_feas = xBase

        # only constraint triangle inequality if it is true for starting point
        self.use_tri_ineq = not (False in self.idf.paramHelpers.checkPhysicalConsistency(self.model.xStd).values())

        # formulate problem as objective function
        opt = pyOpt.Optimization('Constrained OLS', self.minimizeBaseOLSError)
        opt.addObj('u')

        self.addVarsAndConstraints(opt)

        nl = self.model.N_LINKS
        opt.addConGroup('xbase', len(xBase), 'e', equal=0.0)   #=xBase)  # lower=-0.1, upper=0.1)
        opt.addConGroup('inertia', 3*nl, 'i', lower=0.0, upper=100)
        if self.use_tri_ineq:
            opt.addConGroup('tri_ineq', 3*nl, 'i', lower=0.0, upper=100)
        if self.idf.opt['limitOverallMass']:
            opt.addConGroup('mass sum', 1, 'i',
                            lower=self.idf.opt['limitMassVal']*0.99,
                            upper=self.idf.opt['limitMassVal']*1.01)
        if self.idf.opt['restrictCOMtoHull']:
            opt.addConGroup('com', 6*nl, 'i', lower=0.0, upper=100)

        # set CAD/previous sol as starting point (should be already within constraints for some
        # solvers?)
        # atm, either can be eqal in projection to feasible base or consistent and equal to CAD (and
        # not equal to base), hm
        for i in range(len(opt._variables)):
            opt._variables[i].value = self.model.xStd[i]   #xStdModel[i]

        print(opt)

        if self.idf.opt['useIPOPTforNL']:
            # not deterministic, takes a long time?
            solver = pyOpt.IPOPT()
            solver.setOption('linear_solver', 'ma57')  #mumps or hsl: ma27, ma57, ma77, ma86, ma97 or mkl: pardiso
            solver.setOption('max_iter', self.idf.opt['nlOptIterations'])
            #solver.setOption('max_cpu_time', 120)
            solver.setOption('print_level', 3)  #0 none ... 5 max
            #solver.setOption('expect_infeasible_problem', 'yes')

            #don't start too far away from inital values (boundaries push even if starting inside feasible set)
            solver.setOption('bound_push', 0.0000001)
            solver.setOption('bound_frac', 0.0000001)
            #don't relax bounds
            solver.setOption('bound_relax_factor', 0.0) #1e-16)
        else:
            # solve optimization problem
            solver = pyOpt.PSQP(disp_opts=True)
            solver.setOption('MIT', self.idf.opt['nlOptIterations'])
            solver.setOption('TOLC', 0.0)
            solver.setOption('XMAX', 1e8)
            if self.idf.opt['verbose']:
                solver.setOption('IPRINT', 1)

            '''
            #SLSQP seems to violate constraints, no options to configure boundary handling
            solver = pyOpt.SLSQP(disp_opts=True)
            solver.setOption('MAXIT', self.idf.opt['nlOptIterations'])
            if self.idf.opt['verbose']:
                solver.setOption('IPRINT', -1)
            '''

        solver(opt)         #run optimization

        sol = opt.solution(0)
        #print(sol)
        # get solution vector
        #sol_vec = np.array([sol.getVar(x).value for x in range(0, len(sol._variables))])

        sol_vec = self.last_best_x

        self.model.xStd = sol_vec

