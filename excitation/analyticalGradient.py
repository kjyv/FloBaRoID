"""Analytical gradient computation for trajectory optimization.

Implements the gradient chain:

    d(objective)/d(alpha) = d(obj)/d(cond) * d(cond)/d(YBase) : d(YBase)/d(x) * d(x)/d(alpha)

where x = (q, dq, ddq) are joint states, alpha are Fourier parameters,
and ':' is the Frobenius inner product (tensor contraction).

The condition number gradient uses the SVD decomposition from
Ayusawa et al., "Generating Optimal Excitation Trajectories for
Identification of Large Dimensional Systems", ICRA 2017.

The overall gradient is split into two phases:
  Phase A (expensive): Loop over samples, compute d(cond)/d(state) via numerical FD
    of the iDynTree regressor. Result: sensitivity arrays of shape (n_samples, n_dofs).
  Phase B (cheap): Chain sensitivities with analytical trajectory Jacobians using
    vectorized numpy operations to get d(objective)/d(alpha).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from idyntree import bindings as iDynTree

from excitation.trajectoryGenerator import BoundedOscillationGenerator, PulsedTrajectory

if TYPE_CHECKING:
    from excitation.trajectoryGenerator import OscillationGenerator
    from excitation.trajectoryOptimizer import TrajectoryOptimizer
    from identification.model import Model


def compute_cond_gradient(
    YBase: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute condition number and SVD components for its gradient.

    The gradient of cond(Y) = s_max/s_min w.r.t. Y is (Ayusawa Eq. 17):
        dCond/dY = (1/s_min) * u_1 @ v_1^T - (s_max/s_min^2) * u_n @ v_n^T

    Returns:
        cond, s_max, s_min, u_first, u_last, v_first, v_last
    """
    U, S, Vt = np.linalg.svd(YBase, full_matrices=False)
    s_max = float(S[0])
    s_min = float(S[-1])
    cond = s_max / max(s_min, 1e-30)
    return cond, s_max, s_min, U[:, 0].copy(), U[:, -1].copy(), Vt[0, :].copy(), Vt[-1, :].copy()


def _get_projection_matrix(model: Model) -> np.ndarray:
    """Get the base projection matrix (Pb or B depending on config)."""
    if model.opt.get("useBasisProjection", False):
        return model.B
    return model.Pb


def _compute_single_regressor(
    model: Model,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    q_buf: iDynTree.JointPosDoubleArray,
    dq_buf: iDynTree.JointDOFsDoubleArray,
    ddq_buf: iDynTree.JointDOFsDoubleArray,
    base_acc: iDynTree.Vector6,
    reg_buf: iDynTree.MatrixDynSize,
    set_state: bool = True,
    # pre-allocated floating base objects (None for fixed base)
    fb_identity: iDynTree.Transform | None = None,
    fb_zero_twist: iDynTree.Twist | None = None,
) -> np.ndarray:
    """Compute full standard regressor for one sample, including friction columns.

    When set_state=False, skips setRobotState (use for acceleration-only perturbations
    where q and dq are unchanged from the previous call).
    """
    nd = model.num_dofs
    fb = 6 if model.opt["floatingBase"] else 0

    for i in range(nd):
        q_buf.setVal(i, float(pos[i]))
        dq_buf.setVal(i, float(vel[i]))
        ddq_buf.setVal(i, float(acc[i]))

    if set_state:
        if model.opt["floatingBase"]:
            model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
        else:
            model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)

    model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
    Y = reg_buf.toNumPy().copy()

    if not model.opt["floatingBase"]:
        Y = Y[6:, :]

    if model.opt.get("identifyGravityParamsOnly", False):
        Y = np.delete(Y, model.inertia_params, 1)

    # Append friction columns (must match what computeRegressors does in model.py)
    if model.opt.get("identifyFriction", False):
        sign_vel = np.sign(vel)
        static_diag = np.identity(nd) * sign_vel
        if fb:
            offset_reg = np.vstack((np.zeros((fb, nd)), static_diag))
        else:
            offset_reg = static_diag
        Y = np.concatenate((Y, offset_reg), axis=1)

        if not model.opt.get("identifyGravityParamsOnly", False):
            if model.opt.get("identifySymmetricVelFriction", True):
                vel_diag = np.identity(nd) * vel
                if fb:
                    friction_reg = np.vstack((np.zeros((fb, nd)), vel_diag))
                else:
                    friction_reg = vel_diag
            else:
                dq_p = vel.copy()
                dq_p[dq_p < 0] = 0
                dq_m = vel.copy()
                dq_m[dq_m > 0] = 0
                vel_diag = np.hstack((np.identity(nd) * dq_p, np.identity(nd) * dq_m))
                if fb:
                    friction_reg = np.vstack((np.zeros((fb, 2 * nd)), vel_diag))
                else:
                    friction_reg = vel_diag
            Y = np.concatenate((Y, friction_reg), axis=1)

    return Y


def _compute_wf_derivatives(
    optimizer: TrajectoryOptimizer,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute d(q,dq,ddq)/d(wf) via central differences of the trajectory.

    Returns three arrays of shape (n_samples, n_dofs): dq_dwf, ddq_dwf, dddq_dwf.
    """
    nd = optimizer.num_dofs
    n_samples = len(times)
    wf = optimizer.trajectory.w_f_global
    use_deg = optimizer.config["useDeg"]
    eps_wf = 1e-7

    def _eval_traj(wf_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate trajectory at given wf using vectorized Fourier computation."""
        traj = PulsedTrajectory(nd, use_deg=use_deg)
        traj.initWithParams(
            optimizer.trajectory.a,
            optimizer.trajectory.b,
            optimizer.trajectory.q,
            optimizer.trajectory.nf,
            wf_val,
            joint_limits=optimizer._joint_limits,
        )
        p = np.empty((n_samples, nd))
        v = np.empty((n_samples, nd))
        a = np.empty((n_samples, nd))
        for di in range(nd):
            oi = traj.oscillators[di]
            li = np.arange(1, oi.nf + 1)
            wlti = wf_val * np.outer(times, li)
            si, ci = np.sin(wlti), np.cos(wlti)
            ai, bi = np.array(oi.a), np.array(oi.b)
            if isinstance(oi, BoundedOscillationGenerator):
                rw = ci @ bi + si @ ai
                thi = np.tanh(rw)
                sc2 = 1.0 - thi**2
                wli = wf_val * li
                rd = ci @ (ai * wli) - si @ (bi * wli)
                rdd = -si @ (ai * wli**2) - ci @ (bi * wli**2)
                p[:, di] = oi.q_center + oi.q_range * thi
                v[:, di] = oi.q_range * sc2 * rd
                a[:, di] = oi.q_range * (sc2 * rdd - 2.0 * thi * sc2 * rd**2)
            else:
                ac = ai / (wf_val * li)
                bc = bi / (wf_val * li)
                p[:, di] = si @ ac - ci @ bc + oi.nf * oi.q0
                v[:, di] = ci @ ai + si @ bi
                wli = wf_val * li
                a[:, di] = -si @ (ai * wli) + ci @ (bi * wli)
        if use_deg:
            p, v, a = np.deg2rad(p), np.deg2rad(v), np.deg2rad(a)
        return p, v, a

    pp, vp, ap = _eval_traj(wf + eps_wf)
    pm, vm, am = _eval_traj(wf - eps_wf)
    inv_2eps = 1.0 / (2.0 * eps_wf)
    return (pp - pm) * inv_2eps, (vp - vm) * inv_2eps, (ap - am) * inv_2eps


def _traj_jac_single(
    osc: OscillationGenerator | BoundedOscillationGenerator,
    t_val: float,
    nf_d: int,
    wf: float,
    use_deg: bool,
    n_vars: int,
    a_var_start: int,
    b_var_start: int,
    q0_var_idx: int,
    dq_dwf_t: float,
    ddq_dwf_t: float,
    dddq_dwf_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute trajectory Jacobians for one joint at one time point.

    Returns (dq_row, ddq_row, dddq_row), each shape (n_vars,).
    Only entries for variables affecting this joint (and wf) are nonzero.
    """
    deg_factor = np.pi / 180.0 if use_deg else 1.0
    dq_row = np.zeros(n_vars)
    ddq_row = np.zeros(n_vars)
    dddq_row = np.zeros(n_vars)

    l_arr = np.arange(1, nf_d + 1)
    wl = wf * l_arr
    wlt = wf * l_arr * t_val
    sin_wlt = np.sin(wlt)
    cos_wlt = np.cos(wlt)
    a_arr = np.array(osc.a)
    b_arr = np.array(osc.b)

    is_bounded = isinstance(osc, BoundedOscillationGenerator)

    if is_bounded:
        bounded_osc = cast("BoundedOscillationGenerator", osc)
        raw = np.dot(cos_wlt, b_arr) + np.dot(sin_wlt, a_arr)
        th = np.tanh(raw)
        sc = 1.0 - th**2
        qr = bounded_osc.q_range
        raw_dot = np.dot(cos_wlt, a_arr * wl) - np.dot(sin_wlt, b_arr * wl)
        raw_ddot = -np.dot(sin_wlt, a_arr * wl**2) - np.dot(cos_wlt, b_arr * wl**2)

        for l_idx in range(nf_d):
            # a coefficient
            vi_a = a_var_start + l_idx
            dr_a = sin_wlt[l_idx]
            dr_dot_a = wl[l_idx] * cos_wlt[l_idx]
            dr_ddot_a = -(wl[l_idx] ** 2) * sin_wlt[l_idx]
            dq_row[vi_a] = qr * sc * dr_a
            ddq_row[vi_a] = qr * sc * (-2.0 * th * dr_a * raw_dot + dr_dot_a)
            dsc = -2.0 * th * sc * dr_a
            d_thsc = sc * (sc - 2.0 * th**2) * dr_a
            dddq_row[vi_a] = qr * (
                dsc * raw_ddot + sc * dr_ddot_a - 2.0 * d_thsc * raw_dot**2 - 4.0 * th * sc * raw_dot * dr_dot_a
            )

            # b coefficient
            vi_b = b_var_start + l_idx
            dr_b = cos_wlt[l_idx]
            dr_dot_b = -wl[l_idx] * sin_wlt[l_idx]
            dr_ddot_b = -(wl[l_idx] ** 2) * cos_wlt[l_idx]
            dq_row[vi_b] = qr * sc * dr_b
            ddq_row[vi_b] = qr * sc * (-2.0 * th * dr_b * raw_dot + dr_dot_b)
            dsc_b = -2.0 * th * sc * dr_b
            d_thsc_b = sc * (sc - 2.0 * th**2) * dr_b
            dddq_row[vi_b] = qr * (
                dsc_b * raw_ddot + sc * dr_ddot_b - 2.0 * d_thsc_b * raw_dot**2 - 4.0 * th * sc * raw_dot * dr_dot_b
            )

        # q0: only affects position via q_center
        dq_row[q0_var_idx] = deg_factor
    else:
        inv_wl = 1.0 / wl
        for l_idx in range(nf_d):
            vi_a = a_var_start + l_idx
            dq_row[vi_a] = sin_wlt[l_idx] * inv_wl[l_idx]
            ddq_row[vi_a] = cos_wlt[l_idx]
            dddq_row[vi_a] = -wl[l_idx] * sin_wlt[l_idx]

            vi_b = b_var_start + l_idx
            dq_row[vi_b] = -cos_wlt[l_idx] * inv_wl[l_idx]
            ddq_row[vi_b] = sin_wlt[l_idx]
            dddq_row[vi_b] = wl[l_idx] * cos_wlt[l_idx]

        dq_row[q0_var_idx] = nf_d * deg_factor

    # wf contribution (from precomputed numerical FD)
    dq_row[0] = dq_dwf_t
    ddq_row[0] = ddq_dwf_t
    dddq_row[0] = dddq_dwf_t

    return dq_row, ddq_row, dddq_row


def compute_analytical_gradient(
    optimizer: TrajectoryOptimizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute complete analytical gradient of objective and constraint Jacobian.

    Phase A: Loop over samples, compute regressor sensitivities via numerical FD.
    Phase B: Chain with analytical trajectory Jacobians (vectorized).
    Phase C: Penalty and constraint gradients.

    Returns:
        obj_grad: shape (n_vars,)
        con_grad: shape (n_constraints, n_vars)
    """
    cache = optimizer._ag_cache
    model = optimizer.model
    config = optimizer.config
    nd = optimizer.num_dofs
    fb = 6 if config["floatingBase"] else 0
    n_dofs_out = nd + fb
    n_vars = 1 + nd + 2 * optimizer.total_ab

    YBase = cache["YBase"]
    positions = cache["positions"]
    velocities = cache["velocities"]
    accelerations = cache["accelerations"]
    times = cache["times"]
    n_samples = len(times)

    if config.get("verbose", 0):
        print("Computing analytical gradient...")

    # ---- Phase 1: SVD of YBase ----
    cond_val, s_max, s_min, u1, u_last, v1, v_last = compute_cond_gradient(YBase)
    proj = _get_projection_matrix(model)

    # Projected singular vectors for efficient contraction
    w1 = proj @ v1  # (n_std_params,)
    w2 = proj @ v_last
    c1 = 1.0 / max(s_min, 1e-30)
    c2 = -s_max / max(s_min**2, 1e-60)

    # Parameter vector for torque gradient (matches regressor column order)
    params = model.xStdModel[model.identified_params]

    # ---- Phase 2: wf derivatives (numerical FD of trajectory) ----
    dq_dwf, ddq_dwf, dddq_dwf = _compute_wf_derivatives(optimizer, times)

    # ---- Phase A: Regressor sensitivities via FD (the expensive loop) ----
    epsilon = config.get("analyticalGradientEpsilon", 1e-7)
    inv_2eps = 1.0 / (2.0 * epsilon)

    # Sensitivity arrays: d(cond)/d(state) at each sample
    sens_q = np.zeros((n_samples, nd))
    sens_dq = np.zeros((n_samples, nd))
    sens_ddq = np.zeros((n_samples, nd))

    # Torque sensitivities at argmax samples: d(tau[fb+n])/d(state) for each joint n
    torque_absmax_idx = cache["torque_absmax_idx"]
    torque_grad_samples = set(int(idx) for idx in torque_absmax_idx)
    # dtau_d{state}[n, d] = d(tau[fb+n])/d(state_d) at the argmax sample for joint n
    dtau_dq = np.zeros((nd, nd))
    dtau_ddq_state = np.zeros((nd, nd))
    dtau_dddq = np.zeros((nd, nd))

    # Pre-allocate iDynTree buffers
    q_buf = iDynTree.JointPosDoubleArray(nd)
    dq_buf = iDynTree.JointDOFsDoubleArray(nd)
    ddq_buf = iDynTree.JointDOFsDoubleArray(nd)
    base_acc = iDynTree.Vector6()
    reg_buf = iDynTree.MatrixDynSize()
    fb_identity = iDynTree.Transform.Identity() if fb else None
    fb_zero_twist = iDynTree.Twist.Zero() if fb else None

    # Working buffers for perturbation (avoid allocating per-perturbation)
    pos_buf = np.empty(nd)
    vel_buf = np.empty(nd)
    acc_buf = np.empty(nd)

    subsample = config.get("analyticalGradientSubsample", 1)

    for t in range(0, n_samples, subsample):
        np.copyto(pos_buf, positions[t])
        np.copyto(vel_buf, velocities[t])
        np.copyto(acc_buf, accelerations[t])

        row_start = t * n_dofs_out
        u1_t = u1[row_start : row_start + n_dofs_out]
        u_last_t = u_last[row_start : row_start + n_dofs_out]
        need_torque = t in torque_grad_samples

        # --- Position perturbations ---
        for d in range(nd):
            orig = pos_buf[d]
            pos_buf[d] = orig + epsilon
            Y_plus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=True,
                fb_identity=fb_identity,
                fb_zero_twist=fb_zero_twist,
            )
            pos_buf[d] = orig - epsilon
            Y_minus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=True,
                fb_identity=fb_identity,
                fb_zero_twist=fb_zero_twist,
            )
            pos_buf[d] = orig

            dY = Y_plus - Y_minus
            dY_w1 = dY @ w1 * inv_2eps
            dY_w2 = dY @ w2 * inv_2eps
            sens_q[t, d] = c1 * np.dot(u1_t, dY_w1) + c2 * np.dot(u_last_t, dY_w2)

            if need_torque:
                dtau = dY @ params * inv_2eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        dtau_dq[n, d] = dtau[fb + n]

        # --- Velocity perturbations ---
        for d in range(nd):
            orig = vel_buf[d]
            vel_buf[d] = orig + epsilon
            Y_plus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=True,
                fb_identity=fb_identity,
                fb_zero_twist=fb_zero_twist,
            )
            vel_buf[d] = orig - epsilon
            Y_minus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=True,
                fb_identity=fb_identity,
                fb_zero_twist=fb_zero_twist,
            )
            vel_buf[d] = orig

            dY = Y_plus - Y_minus
            dY_w1 = dY @ w1 * inv_2eps
            dY_w2 = dY @ w2 * inv_2eps
            sens_dq[t, d] = c1 * np.dot(u1_t, dY_w1) + c2 * np.dot(u_last_t, dY_w2)

            if need_torque:
                dtau = dY @ params * inv_2eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        dtau_ddq_state[n, d] = dtau[fb + n]

        # --- Acceleration perturbations ---
        # Set state once at nominal (q, dq), then only perturb ddq
        _compute_single_regressor(
            model,
            pos_buf,
            vel_buf,
            acc_buf,
            q_buf,
            dq_buf,
            ddq_buf,
            base_acc,
            reg_buf,
            set_state=True,
            fb_identity=fb_identity,
            fb_zero_twist=fb_zero_twist,
        )
        for d in range(nd):
            orig = acc_buf[d]
            acc_buf[d] = orig + epsilon
            Y_plus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=False,
            )
            acc_buf[d] = orig - epsilon
            Y_minus = _compute_single_regressor(
                model,
                pos_buf,
                vel_buf,
                acc_buf,
                q_buf,
                dq_buf,
                ddq_buf,
                base_acc,
                reg_buf,
                set_state=False,
            )
            acc_buf[d] = orig

            dY = Y_plus - Y_minus
            dY_w1 = dY @ w1 * inv_2eps
            dY_w2 = dY @ w2 * inv_2eps
            sens_ddq[t, d] = c1 * np.dot(u1_t, dY_w1) + c2 * np.dot(u_last_t, dY_w2)

            if need_torque:
                dtau = dY @ params * inv_2eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        dtau_dddq[n, d] = dtau[fb + n]

    # If subsampling, scale sensitivities to account for missing samples
    if subsample > 1:
        sens_q *= subsample
        sens_dq *= subsample
        sens_ddq *= subsample

    # ---- Phase B: Chain rule with trajectory Jacobians (vectorized) ----
    obj_grad = np.zeros(n_vars)
    wf = optimizer.trajectory.w_f_global
    use_deg = config["useDeg"]
    deg_factor = np.pi / 180.0 if use_deg else 1.0
    cond_scale = cache["cond_scale"]
    cond = cache["cond"]

    # Scale factor: d(log10(cond)*scale)/d(cond) = scale / (cond * ln(10))
    if cond > 1.0:
        log_scale = cond_scale / (cond * np.log(10.0))
    else:
        log_scale = 0.0

    # wf contribution: sum over all samples and joints
    obj_grad[0] += log_scale * (np.sum(sens_q * dq_dwf) + np.sum(sens_dq * ddq_dwf) + np.sum(sens_ddq * dddq_dwf))

    # Per-joint Fourier coefficient contributions
    q0_start = 1
    a_start = 1 + nd
    b_start = a_start + optimizer.total_ab
    a_offset = a_start
    b_offset = b_start

    for d in range(nd):
        osc = optimizer.trajectory.oscillators[d]
        nf_d = optimizer.nf[d]
        l_arr = np.arange(1, nf_d + 1)
        wl = wf * l_arr
        wlt = wf * np.outer(times, l_arr)  # (n_samples, nf_d)
        sin_wlt = np.sin(wlt)
        cos_wlt = np.cos(wlt)
        a_arr = np.array(osc.a)
        b_arr = np.array(osc.b)

        s_q = sens_q[:, d]  # (n_samples,)
        s_dq = sens_dq[:, d]
        s_ddq = sens_ddq[:, d]

        is_bounded = isinstance(osc, BoundedOscillationGenerator)

        if is_bounded:
            # Precompute bounded-mode intermediates for all samples
            raw = cos_wlt @ b_arr + sin_wlt @ a_arr  # (n_samples,)
            th = np.tanh(raw)
            sc = 1.0 - th**2
            qr = osc.q_range
            raw_dot = cos_wlt @ (a_arr * wl) - sin_wlt @ (b_arr * wl)
            raw_ddot = -sin_wlt @ (a_arr * wl**2) - cos_wlt @ (b_arr * wl**2)

            for l_idx in range(nf_d):
                # --- a coefficient ---
                vi_a = a_offset + l_idx
                dr = sin_wlt[:, l_idx]
                dr_dot = wl[l_idx] * cos_wlt[:, l_idx]
                dr_ddot = -(wl[l_idx] ** 2) * sin_wlt[:, l_idx]

                dq_val = qr * sc * dr
                ddq_val = qr * sc * (-2.0 * th * dr * raw_dot + dr_dot)
                dsc = -2.0 * th * sc * dr
                d_thsc = sc * (sc - 2.0 * th**2) * dr
                dddq_val = qr * (
                    dsc * raw_ddot + sc * dr_ddot - 2.0 * d_thsc * raw_dot**2 - 4.0 * th * sc * raw_dot * dr_dot
                )
                obj_grad[vi_a] += log_scale * (np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val))

                # --- b coefficient ---
                vi_b = b_offset + l_idx
                dr = cos_wlt[:, l_idx]
                dr_dot = -wl[l_idx] * sin_wlt[:, l_idx]
                dr_ddot = -(wl[l_idx] ** 2) * cos_wlt[:, l_idx]

                dq_val = qr * sc * dr
                ddq_val = qr * sc * (-2.0 * th * dr * raw_dot + dr_dot)
                dsc = -2.0 * th * sc * dr
                d_thsc = sc * (sc - 2.0 * th**2) * dr
                dddq_val = qr * (
                    dsc * raw_ddot + sc * dr_ddot - 2.0 * d_thsc * raw_dot**2 - 4.0 * th * sc * raw_dot * dr_dot
                )
                obj_grad[vi_b] += log_scale * (np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val))

            # q0: only position, through q_center
            q0_vi = q0_start + d
            obj_grad[q0_vi] += log_scale * np.sum(s_q) * deg_factor

        else:
            # Unbounded (classic Swevers) mode
            inv_wl = 1.0 / wl
            for l_idx in range(nf_d):
                vi_a = a_offset + l_idx
                dq_val = sin_wlt[:, l_idx] * inv_wl[l_idx]
                ddq_val = cos_wlt[:, l_idx]
                dddq_val = -wl[l_idx] * sin_wlt[:, l_idx]
                obj_grad[vi_a] += log_scale * (np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val))

                vi_b = b_offset + l_idx
                dq_val = -cos_wlt[:, l_idx] * inv_wl[l_idx]
                ddq_val = sin_wlt[:, l_idx]
                dddq_val = wl[l_idx] * cos_wlt[:, l_idx]
                obj_grad[vi_b] += log_scale * (np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val))

            q0_vi = q0_start + d
            obj_grad[q0_vi] += log_scale * np.sum(s_q) * nf_d * deg_factor

        a_offset += nf_d
        b_offset += nf_d

    # ---- Phase C: Penalty gradients ----
    jn = model.jointNames
    torque_limits = np.array([optimizer.limits[jn[n]]["torque"] for n in range(nd)])
    utilization = cache["utilization"]
    util_mean = cache["util_mean"]
    util_std = cache["util_std"]
    f1 = cache["f1"]
    f3 = cache["f3"]
    target_util = config.get("trajectoryTargetTorqueUtil", 0.25)

    # Build per-variable index maps for _traj_jac_single calls
    a_offsets = []
    b_offsets = []
    a_off = a_start
    b_off = b_start
    for d in range(nd):
        a_offsets.append(a_off)
        b_offsets.append(b_off)
        a_off += optimizer.nf[d]
        b_off += optimizer.nf[d]

    # Compute d(torque_absmax[n])/d(alpha) for each joint n
    dtorque_absmax_dalpha = np.zeros((nd, n_vars))
    for n in range(nd):
        t_star = int(torque_absmax_idx[n])
        sign_n = np.sign(cache["torques"][t_star, fb + n])
        t_val = times[t_star]

        for d_joint in range(nd):
            osc_d = optimizer.trajectory.oscillators[d_joint]
            dq_row, ddq_row, dddq_row = _traj_jac_single(
                osc_d,
                t_val,
                optimizer.nf[d_joint],
                wf,
                use_deg,
                n_vars,
                a_offsets[d_joint],
                b_offsets[d_joint],
                q0_start + d_joint,
                dq_dwf[t_star, d_joint],
                ddq_dwf[t_star, d_joint],
                dddq_dwf[t_star, d_joint],
            )
            dtorque_absmax_dalpha[n, :] += sign_n * (
                dtau_dq[n, d_joint] * dq_row + dtau_ddq_state[n, d_joint] * ddq_row + dtau_dddq[n, d_joint] * dddq_row
            )

    # f1 = CoV(utilization) * 10
    dutil_dalpha = dtorque_absmax_dalpha / torque_limits[:, np.newaxis]
    if util_mean > 0 and util_std > 0:
        N = nd
        df1_dutil = np.zeros(nd)
        for n in range(nd):
            df1_dutil[n] = (1.0 / (N * util_mean)) * ((utilization[n] - util_mean) / util_std - f1)
        obj_grad += 10.0 * df1_dutil @ dutil_dalpha

    # f3 = max(0, 1 - util_mean/target) * 10
    if f3 > 0:
        dutil_mean_dalpha = np.mean(dutil_dalpha, axis=0)
        obj_grad += 10.0 * (-1.0 / target_util) * dutil_mean_dalpha

    # f2 = (1 - mean(pos_utilization)) * 10
    pos_min_idx = cache["pos_min_idx"]
    pos_max_idx = cache["pos_max_idx"]
    pos_range_available = cache["pos_range_available"]

    dpos_range_dalpha = np.zeros((nd, n_vars))
    for n in range(nd):
        t_max = int(pos_max_idx[n])
        t_min = int(pos_min_idx[n])
        # trajectory Jacobians at these specific samples for joint n
        for d_joint in range(nd):
            osc_d = optimizer.trajectory.oscillators[d_joint]
            # At t_max
            dq_max, _, _ = _traj_jac_single(
                osc_d,
                times[t_max],
                optimizer.nf[d_joint],
                wf,
                use_deg,
                n_vars,
                a_offsets[d_joint],
                b_offsets[d_joint],
                q0_start + d_joint,
                dq_dwf[t_max, d_joint],
                ddq_dwf[t_max, d_joint],
                dddq_dwf[t_max, d_joint],
            )
            # At t_min
            dq_min, _, _ = _traj_jac_single(
                osc_d,
                times[t_min],
                optimizer.nf[d_joint],
                wf,
                use_deg,
                n_vars,
                a_offsets[d_joint],
                b_offsets[d_joint],
                q0_start + d_joint,
                dq_dwf[t_min, d_joint],
                ddq_dwf[t_min, d_joint],
                dddq_dwf[t_min, d_joint],
            )
            # Only joint n's position matters (positions are per-joint)
            if d_joint == n:
                dpos_range_dalpha[n, :] = (dq_max - dq_min) / pos_range_available[n]

    obj_grad += -10.0 * np.mean(dpos_range_dalpha, axis=0)

    # ---- Constraint gradients ----
    con_grad = np.zeros((optimizer.num_constraints, n_vars))

    # Helper to get trajectory Jacobian for joint n at sample t
    def _get_traj_jac(n: int, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        osc_n = optimizer.trajectory.oscillators[n]
        return _traj_jac_single(
            osc_n,
            times[t],
            optimizer.nf[n],
            wf,
            use_deg,
            n_vars,
            a_offsets[n],
            b_offsets[n],
            q0_start + n,
            dq_dwf[t, n],
            ddq_dwf[t, n],
            dddq_dwf[t, n],
        )

    for n in range(nd):
        # Position lower: g[n] = limit_lower - pos_min[n] → dg = -dq at argmin
        dq_min, _, _ = _get_traj_jac(n, int(pos_min_idx[n]))
        con_grad[n, :] = -dq_min

        # Position upper: g[nd+n] = pos_max[n] - limit_upper → dg = dq at argmax
        dq_max, _, _ = _get_traj_jac(n, int(pos_max_idx[n]))
        con_grad[nd + n, :] = dq_max

        # Velocity: g[2*nd+n] = vel_absmax[n] - vel_limit
        t_vel = int(cache["vel_absmax_idx"][n])
        sign_v = np.sign(velocities[t_vel, n])
        _, ddq_vel, _ = _get_traj_jac(n, t_vel)
        con_grad[2 * nd + n, :] = sign_v * ddq_vel

        # Torque: g[3*nd+n] = torque_absmax[n] - torque_limit
        con_grad[3 * nd + n, :] = dtorque_absmax_dalpha[n, :]

    offset = 4 * nd
    if config["minVelocityConstraint"]:
        for n in range(nd):
            t_vel = int(cache["vel_absmax_idx"][n])
            sign_v = np.sign(velocities[t_vel, n])
            _, ddq_vel, _ = _get_traj_jac(n, t_vel)
            con_grad[offset + n, :] = -sign_v * ddq_vel
        offset += nd

    # Min torque utilization: g = limit*min_util - torque_absmax
    for n in range(nd):
        con_grad[offset + n, :] = -dtorque_absmax_dalpha[n, :]
    offset += nd

    # ---- Collision constraint gradients ----
    if optimizer.num_coll_constraints > 0 and hasattr(optimizer, "_ag_collision_cache"):
        coll_cache = optimizer._ag_collision_cache

        # Group pairs by their argmin sample for efficiency
        sample_pairs: dict[int, list[int]] = {}
        for g_cnt in coll_cache:
            t_star = coll_cache[g_cnt][0]
            sample_pairs.setdefault(t_star, []).append(g_cnt)

        for t_star, pair_indices in sample_pairs.items():
            q_star = positions[t_star].copy()

            # Compute d(distance)/dq for all pairs at this sample via FD
            ddist_dq: dict[int, np.ndarray] = {gi: np.zeros(nd) for gi in pair_indices}

            for j in range(nd):
                q_p = q_star.copy()
                q_p[j] += epsilon
                optimizer.setCollisionRobotState(q_p)
                d_plus: dict[int, float] = {}
                for gi in pair_indices:
                    l0, l1 = optimizer._collision_pairs[gi]
                    d_plus[gi] = optimizer.getLinkDistance(l0, l1, q_p)

                q_m = q_star.copy()
                q_m[j] -= epsilon
                optimizer.setCollisionRobotState(q_m)
                for gi in pair_indices:
                    l0, l1 = optimizer._collision_pairs[gi]
                    d_minus = optimizer.getLinkDistance(l0, l1, q_m)
                    ddist_dq[gi][j] = (d_plus[gi] - d_minus) * inv_2eps

            # Chain with trajectory Jacobians: d(distance)/dalpha = ddist_dq @ dq/dalpha at t_star
            for gi in pair_indices:
                c_idx = offset + gi
                # Sum over joints: ddist_dq[j] * dq_j/dalpha
                for d_joint in range(nd):
                    dq_row, _, _ = _get_traj_jac(d_joint, t_star)
                    con_grad[c_idx, :] += ddist_dq[gi][d_joint] * dq_row

    if config.get("verbose", 0):
        print(f"Analytical gradient computed (cond={cond_val:.1f}, |grad|={np.linalg.norm(obj_grad):.3e})")

    return obj_grad, con_grad
