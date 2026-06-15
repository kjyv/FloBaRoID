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


def _var_info(topt):
    """Reconstruct the Optuna var_info list (name, lo, hi, init) for an optimizer."""
    vi = [("wf", topt.wf_min, topt.wf_max, topt.wf_init)]
    for i in range(topt.num_dofs):
        vi.append((f"q_{i}", float(topt.qmin[i]), float(topt.qmax[i]), float(topt.qinit[i])))
    for i in range(topt.num_dofs):
        for j in range(topt.nf[i]):
            vi.append((f"a{i}_{j}", topt.amin, topt.amax, float(topt.ainit[i][j])))
    for i in range(topt.num_dofs):
        for j in range(topt.nf[i]):
            vi.append((f"b{i}_{j}", topt.bmin, topt.bmax, float(topt.binit[i][j])))
    return vi


def test_scale_amplitudes_only_scales_coefficients(monkeypatch):
    """scaleAmplitudes scales the Fourier a/b coefficients but leaves wf and q untouched."""
    topt = _build_kuka_optimizer(monkeypatch)
    x0, _, _ = topt.buildVariableBounds()
    scaled = topt.scaleAmplitudes(x0, 0.5)
    wf0, q0, a0, b0 = topt.vecToParams(x0)
    wf1, q1, a1, b1 = topt.vecToParams(scaled)
    assert wf1 == wf0
    np.testing.assert_allclose(q1, q0)
    for i in range(topt.num_dofs):
        np.testing.assert_allclose(a1[i], 0.5 * a0[i])
        np.testing.assert_allclose(b1[i], 0.5 * b0[i])


def test_build_seed_trial_params(monkeypatch):
    """buildSeedTrialParams loads a structure-matching trajectory file and skips a mismatched one."""
    import tempfile

    topt = _build_kuka_optimizer(monkeypatch)
    vi = _var_info(topt)
    tmp = tempfile.mkdtemp()

    # matching file: same joint count and per-joint harmonics as the current config
    good = os.path.join(tmp, "good_traj.npz")
    a_good = np.array([np.full(topt.nf[i], 0.05) for i in range(topt.num_dofs)], dtype=object)
    np.savez(
        good,
        a=a_good,
        b=a_good,
        q=np.zeros(topt.num_dofs),
        nf=np.array([topt.nf[i] for i in range(topt.num_dofs)]),
        wf=np.float64(0.1),
    )
    topt.config["trajectorySeedSolutions"] = [good]
    seeds = topt.buildSeedTrialParams(vi)
    assert len(seeds) == 1
    assert set(seeds[0].keys()) == {name for name, *_ in vi}

    # mismatched structure (different per-joint harmonics) must be skipped
    bad = os.path.join(tmp, "bad_traj.npz")
    a_bad = np.array([np.zeros(topt.nf[i] + 1) for i in range(topt.num_dofs)], dtype=object)
    np.savez(
        bad,
        a=a_bad,
        b=a_bad,
        q=np.zeros(topt.num_dofs),
        nf=np.array([topt.nf[i] + 1 for i in range(topt.num_dofs)]),
        wf=np.float64(0.1),
    )
    topt.config["trajectorySeedSolutions"] = [bad]
    assert topt.buildSeedTrialParams(vi) == []


def test_analytical_gradient_matches_finite_differences(monkeypatch):
    """The analytical objective gradient agrees with finite differences of objectiveFunc.

    Guards the intricate gradient assembly (D-optimality + velocity/torque/position soft
    costs) against regressions; the velocity-magnitude cost is enabled so its term is
    exercised too.
    """
    topt = _build_kuka_optimizer(monkeypatch)
    topt.config["useAnalyticalGradients"] = 1
    topt.config["trajectoryTargetVelocity"] = 0.5  # exercise the velocity-magnitude term
    from excitation.analyticalGradient import compute_analytical_gradient

    x0, _, _ = topt.buildVariableBounds()
    f0, _g0, _ = topt.objectiveFunc(x0)
    obj_grad, _con_grad = compute_analytical_gradient(topt)

    rng = np.random.default_rng(0)
    idxs = sorted(rng.choice(len(x0), size=8, replace=False).tolist())
    eps = 1e-6
    for i in idxs:
        xp = x0.copy()
        xp[i] += eps
        xm = x0.copy()
        xm[i] -= eps
        fp, _, _ = topt.objectiveFunc(xp)
        fm, _, _ = topt.objectiveFunc(xm)
        fd = (fp - fm) / (2 * eps)
        # generous tolerance: catches sign flips / missing terms / factor errors. The
        # analytical D-optimality part itself uses an inner FD, so exact agreement is
        # not expected; near-zero entries are dominated by FD noise (atol).
        assert abs(obj_grad[i] - fd) <= 0.08 * abs(fd) + 0.05, f"grad[{i}]: analytical {obj_grad[i]} vs FD {fd}"
