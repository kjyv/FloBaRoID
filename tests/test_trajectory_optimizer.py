"""Tests for the trajectory optimizer's cyipopt driver and collision-constraint bookkeeping."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: avoid opening windows during optimizer import/construction

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class _StubOptimizer:
    """Minimal stand-in providing the two attributes IpoptProblem needs from an optimizer.

    Avoids constructing a full TrajectoryOptimizer (model, meshes, ...) for the
    isolated cyipopt-driver test.
    """

    is_gradient_eval = False

    def approx_jacobian(self, f, x, epsilon, *args):
        """Forward finite-difference jacobian of the vector function f at x."""
        x = np.asarray(x, dtype=float)
        f0 = np.asarray(f(x), dtype=float)
        jac = np.zeros((f0.size, x.size))
        for i in range(x.size):
            xp = x.copy()
            xp[i] += epsilon
            jac[:, i] = (np.asarray(f(xp), dtype=float) - f0) / epsilon
        return jac


def test_ipopt_problem_solves_constrained_qp():
    """The cyipopt IpoptProblem adapter solves a trivial constrained QP via the same
    class-based API the optimizer uses, guarding the pyOptSparse->cyipopt migration.

    minimize x0^2 + x1^2  s.t.  x0 + x1 >= 1  ->  optimum (0.5, 0.5).
    """
    from typing import cast

    import cyipopt

    from excitation.optimizer import IpoptProblem, Optimizer

    def eval_func(x):
        """Return (objective, constraint values, fail flag) for the QP."""
        f = float(x[0] ** 2 + x[1] ** 2)
        g = np.array([x[0] + x[1]])
        return f, g, 0.0

    # the stub implements only the subset of Optimizer that IpoptProblem uses
    problem = IpoptProblem(cast(Optimizer, _StubOptimizer()), eval_func, num_constraints=1, sens_step=1e-7)
    nlp = cyipopt.Problem(
        n=2,
        m=1,
        problem_obj=problem,
        lb=np.array([-10.0, -10.0]),
        ub=np.array([10.0, 10.0]),
        cl=np.array([1.0]),
        cu=np.array([1e19]),
    )
    nlp.add_option("linear_solver", "mumps")
    nlp.add_option("print_level", 0)
    nlp.add_option("sb", "yes")
    nlp.add_option("hessian_approximation", "limited-memory")

    x_opt, info = nlp.solve(np.array([2.0, 2.0]))
    assert info["status"] == 0  # solved to optimality
    np.testing.assert_allclose(x_opt, [0.5, 0.5], atol=1e-4)
    assert x_opt[0] + x_opt[1] >= 1.0 - 1e-6  # constraint satisfied


def _build_kuka_optimizer(monkeypatch):
    """Construct a TrajectoryOptimizer for the kuka with its world (setup only, no solve)."""
    monkeypatch.chdir(PROJECT_ROOT)
    import yaml
    from idyntree import bindings as iDynTree

    from excitation.trajectoryGenerator import computeTrajectoryDynamics
    from excitation.trajectoryOptimizer import TrajectoryOptimizer
    from identification.model import Model
    from identifier import Identification

    with open("configs/kuka_lwr4.yaml") as f:
        config = yaml.safe_load(f)
    config["urdf"] = "model/kuka_lwr4.urdf"
    config["urdf_real"] = None
    config["jointNames"] = iDynTree.StringVector([])
    assert iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"])
    config["num_dofs"] = len(config["jointNames"])
    config["verbose"] = 0
    config["showOptimizationGraph"] = 0
    config["showModelVisualization"] = 0

    model = Model(config, config["urdf"])
    idf = Identification(config, config["urdf"], None, None, None, None)
    return TrajectoryOptimizer(config, idf, model, computeTrajectoryDynamics, world="model/world_kuka.urdf")


def test_collision_constraint_count_matches_pairs(monkeypatch):
    """num_coll_constraints must equal the number of actually-checked collision pairs.

    Regression for a bug where an approximate formula over- or under-counted the pairs,
    so the per-pair collision residuals overran the constraint vector (IndexError) once
    the pairing configuration changed.
    """
    topt = _build_kuka_optimizer(monkeypatch)
    assert topt.num_coll_constraints == len(topt._collision_pairs)
    # the constraint-bound vectors must match the total constraint count, so the
    # collision residuals written at index c_s + g_cnt always stay in range
    con_lower, con_upper = topt.buildConstraintBounds()
    assert len(con_lower) == topt.num_constraints == len(con_upper)
