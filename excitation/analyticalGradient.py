"""Analytical gradient computation for trajectory optimization.

Implements the gradient chain:

    d(objective)/d(alpha) = d(obj)/d(Y) : d(Y)/d(x) * d(x)/d(alpha)

where x = (q, dq, ddq) are joint states, alpha are Fourier parameters,
and ':' is the Frobenius inner product (tensor contraction).

The objective is regularized D-optimality: -log(det(Y^T Y + δI)).

The overall gradient is split into two phases:
  Phase A (expensive): Loop over samples, compute d(obj)/d(state) via numerical FD
    of the iDynTree regressor. Result: sensitivity arrays of shape (n_samples, n_dofs).
  Phase B (cheap): Chain sensitivities with analytical trajectory Jacobians using
    vectorized numpy operations to get d(objective)/d(alpha).
"""

from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING, cast

import numpy as np
from idyntree import bindings as iDynTree

from excitation.trajectoryGenerator import BoundedOscillationGenerator, PulsedTrajectory

if TYPE_CHECKING:
    from excitation.trajectoryGenerator import OscillationGenerator
    from excitation.trajectoryOptimizer import TrajectoryOptimizer
    from identification.model import Model

# --- Worker pool for parallel gradient computation ---
# Each worker process has its own iDynTree model instance (stored in module-level dict).
# The pool persists across gradient calls to avoid re-creating models.
_worker_state: dict = {}
_gradient_pool: multiprocessing.pool.Pool | None = None
_gradient_pool_size: int = 0


def _expand_gravity_only_params(w_gravity: np.ndarray, n_links: int) -> np.ndarray:
    """Expand a gravity-only parameter vector (4 per link) to full inertial (10 per link).

    When identifyGravityParamsOnly is True, the regressor only has mass and first-moment
    columns (4 per link). Expands to full 10-per-link by inserting zeros for inertia tensor.
    """
    w_full = np.zeros(n_links * 10)
    for i in range(n_links):
        w_full[10 * i : 10 * i + 4] = w_gravity[4 * i : 4 * (i + 1)]
        # Indices 4-9 (inertia tensor) remain zero
    return w_full


# --- Analytical RNEA derivatives (spatial algebra, no Pinocchio) ---


def _skew3(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from 3-vector."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _batch_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch cross product: (n,3) x (n,3) -> (n,3). Broadcasts single vectors."""
    return np.cross(a, b)


def _motion_cross_matrix(v: np.ndarray) -> np.ndarray:
    """6×6 spatial motion cross product matrix [v]× for constant v.

    [v]× = [[ω×, 0], [v₀×, ω×]] where v = [ω; v₀].
    Used for: v × u = [v]× @ u (spatial motion cross product).
    """
    wx, wy, wz = v[0], v[1], v[2]
    vx, vy, vz = v[3], v[4], v[5]
    return np.array(
        [
            [0, -wz, wy, 0, 0, 0],
            [wz, 0, -wx, 0, 0, 0],
            [-wy, wx, 0, 0, 0, 0],
            [0, -vz, vy, 0, -wz, wy],
            [vz, 0, -vx, wz, 0, -wx],
            [-vy, vx, 0, -wy, wx, 0],
        ]
    )


def _force_cross_matrix(v: np.ndarray) -> np.ndarray:
    """6×6 spatial force cross product matrix [v]×* = -[v]×ᵀ for constant v.

    Used for: v ×* f = [v]×* @ f (spatial force cross product).
    """
    return -_motion_cross_matrix(v).T


def _spatial_inertia_from_dynamic_params(params: np.ndarray) -> np.ndarray:
    """Build 6x6 spatial inertia from 10 inertial parameters (iDynTree ordering).

    params: [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    Returns [[I_origin, [h]×], [[h]×ᵀ, m·I₃]] which is valid for ANY parameter values.
    """
    m, hx, hy, hz = params[0], params[1], params[2], params[3]
    I_origin = np.array(
        [
            [params[4], params[5], params[6]],
            [params[5], params[7], params[8]],
            [params[6], params[8], params[9]],
        ]
    )
    hx_mat = np.array([[0, -hz, hy], [hz, 0, -hx], [-hy, hx, 0]])
    I = np.empty((6, 6))
    I[:3, :3] = I_origin
    I[:3, 3:] = hx_mat
    I[3:, :3] = hx_mat.T
    I[3:, 3:] = m * np.eye(3)
    return I


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrix from roll-pitch-yaw (URDF convention: Rz·Ry·Rx)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _parse_kinematic_tree(urdf_file: str, joint_names: list[str], link_names: list[str]) -> dict:
    """Parse URDF to extract kinematic tree structure for RNEA computation.

    Returns dict with parent/child body indices, joint axes, origin transforms,
    and the mapping from iDynTree link indices to kinematic tree body indices
    (handling fixed-joint merging).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Parse all URDF joints
    urdf_joints: dict[str, dict] = {}
    parent_of_link: dict[str, str] = {}
    fixed_parent: dict[str, str] = {}

    for j_elem in root.findall("joint"):
        jname = j_elem.get("name", "")
        jtype = j_elem.get("type", "fixed")
        parent_elem = j_elem.find("parent")
        child_elem = j_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent = parent_elem.get("link", "")
        child = child_elem.get("link", "")

        origin = j_elem.find("origin")
        xyz = np.array([float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
        rpy = np.array([float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
        axis_elem = j_elem.find("axis")
        axis = (
            np.array([float(x) for x in axis_elem.get("xyz", "0 0 1").split()])
            if axis_elem is not None
            else np.array([0.0, 0.0, 1.0])
        )
        axis = axis / max(np.linalg.norm(axis), 1e-30)

        urdf_joints[jname] = {"type": jtype, "parent": parent, "child": child, "xyz": xyz, "rpy": rpy, "axis": axis}
        parent_of_link[child] = parent
        if jtype == "fixed":
            fixed_parent[child] = parent

    # Body list: root (not a child of any movable joint) + children of movable joints in DOF order
    movable_children = {urdf_joints[jn]["child"] for jn in joint_names}
    root_link = next(ln for ln in link_names if ln not in movable_children)
    body_links = [root_link] + [urdf_joints[jn]["child"] for jn in joint_names]
    body_name_to_idx = {name: i for i, name in enumerate(body_links)}

    # Per-DOF kinematic data
    parent_body: list[int] = []
    child_body: list[int] = []
    joint_axes_list: list[np.ndarray] = []
    joint_xyz_list: list[np.ndarray] = []
    joint_rpy_list: list[np.ndarray] = []

    for jn in joint_names:
        info = urdf_joints[jn]
        # Traverse up through fixed joints to find the body containing the parent link
        p = info["parent"]
        while p not in body_name_to_idx and p in parent_of_link:
            p = parent_of_link[p]
        parent_body.append(body_name_to_idx.get(p, 0))
        child_body.append(body_name_to_idx[info["child"]])
        joint_axes_list.append(info["axis"])
        joint_xyz_list.append(info["xyz"])
        joint_rpy_list.append(info["rpy"])

    # Map iDynTree link index -> (body index, 6×6 spatial transform from link frame to body frame)
    # For links that ARE bodies, the transform is identity.
    # For fixed-joint children, we chain the fixed-joint transforms up to the parent body.
    link_to_body: dict[int, int] = {}
    link_to_body_Xform: dict[int, np.ndarray] = {}

    for i, lname in enumerate(link_names):
        if lname in body_name_to_idx:
            link_to_body[i] = body_name_to_idx[lname]
            link_to_body_Xform[i] = np.eye(6)
        else:
            # Traverse up through fixed joints, accumulating the spatial transforms
            chain: list[str] = []
            p = lname
            while p in fixed_parent:
                # The fixed joint connecting p to its parent
                for jn_fix, jinfo in urdf_joints.items():
                    if jinfo["child"] == p and jinfo["type"] == "fixed":
                        chain.append(jn_fix)
                        break
                p = fixed_parent[p]
            link_to_body[i] = body_name_to_idx.get(p, 0)

            # Build cumulative transform: link_frame -> body_frame
            # Each fixed joint has origin (xyz, rpy) giving transform from parent to child.
            # We need the inverse: child -> parent, chained for all fixed joints in the path.
            X_acc = np.eye(6)
            for jn_fix in chain:
                jinfo = urdf_joints[jn_fix]
                R_fix = _rpy_to_rotation(jinfo["rpy"])
                p_fix = jinfo["xyz"]
                p_skew = _skew3(p_fix)
                # Plücker: ^child X_parent = [[R^T, 0], [-R^T @ [p]×, R^T]]
                Rcp = R_fix.T
                X_fix = np.zeros((6, 6))
                X_fix[:3, :3] = Rcp
                X_fix[3:, 3:] = Rcp
                X_fix[3:, :3] = -Rcp @ p_skew
                # We need parent <- child (inverse): X_fix_inv
                # For Plücker: X^{-1} = [[R, 0], [R @ [p]×, R]] (undo the transform)
                X_inv = np.zeros((6, 6))
                X_inv[:3, :3] = R_fix
                X_inv[3:, 3:] = R_fix
                X_inv[3:, :3] = R_fix @ p_skew
                X_acc = X_inv @ X_acc
            link_to_body_Xform[i] = X_acc

    # Precompute tree topology: descendants and ancestors for each joint
    nd = len(joint_names)
    children_of_body: dict[int, list[int]] = {}
    for d in range(nd):
        children_of_body.setdefault(child_body[d], [])
        children_of_body.setdefault(parent_body[d], []).append(d)

    def _get_descendants(d: int) -> list[int]:
        """Get all descendant joints of joint d (in tree traversal order)."""
        result: list[int] = []
        ci = child_body[d]
        for dd in children_of_body.get(ci, []):
            result.append(dd)
            result.extend(_get_descendants(dd))
        return result

    descendants: list[list[int]] = [_get_descendants(d) for d in range(nd)]

    # Ancestors: joints on the path from joint d to the root (excluding d itself)
    def _get_ancestors(d: int) -> list[int]:
        """Get ancestor joints from d back to root (nearest first)."""
        result: list[int] = []
        body = parent_body[d]
        while body != 0 or any(parent_body[dd] == 0 and child_body[dd] == body for dd in range(nd)):
            # Find the joint whose child_body is this body
            for dd in range(nd):
                if child_body[dd] == body:
                    result.append(dd)
                    body = parent_body[dd]
                    break
            else:
                break
        return result

    ancestors: list[list[int]] = [_get_ancestors(d) for d in range(nd)]

    return {
        "parent_body": parent_body,
        "child_body": child_body,
        "joint_axes": np.array(joint_axes_list),
        "joint_xyz": np.array(joint_xyz_list),
        "joint_rpy": np.array(joint_rpy_list),
        "n_bodies": len(body_links),
        "link_to_body": link_to_body,
        "link_to_body_Xform": link_to_body_Xform,
        "descendants": descendants,
        "ancestors": ancestors,
    }


def _build_body_spatial_inertias(w_iner: np.ndarray, n_links: int, tree_info: dict) -> list[np.ndarray]:
    """Build per-body 6x6 spatial inertias from contraction vector, handling fixed-link merging.

    For links connected via fixed joints to a body, the spatial inertia is transformed
    from the link's frame to the body's frame before summing. This is necessary when
    fixed joints have non-zero offsets (e.g., sensor mounts, backpack, head on a humanoid).
    """
    nb = tree_info["n_bodies"]
    body_I = [np.zeros((6, 6)) for _ in range(nb)]
    for link_idx in range(n_links):
        body_idx = tree_info["link_to_body"][link_idx]
        I_link = _spatial_inertia_from_dynamic_params(w_iner[10 * link_idx : 10 * (link_idx + 1)])
        X = tree_info["link_to_body_Xform"][link_idx]  # ^body X_link (link-to-body motion)
        # Spatial inertia congruence: I_body = Y^T @ I_link @ Y where Y = ^link X_body
        # Since X maps link→body, we need Y = X^{-1}
        Y = np.linalg.inv(X)
        body_I[body_idx] += Y.T @ I_link @ Y
    return body_I


def _compute_rnea_derivatives_batch(
    body_I: list[np.ndarray],
    tree_info: dict,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    is_floating: bool,
    fb: int,
    n_dofs_out: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dtau/dq, dtau/dv, dtau/da for all samples using analytical RNEA derivatives.

    Uses Featherstone's spatial algebra with 6x6 inertia matrices directly,
    which handles arbitrary (non-physical) parameter vectors without issues.

    Returns:
        dtau_dq: (n_samples, n_dofs_out, n_dofs) — position derivative
        dtau_dv: (n_samples, n_dofs_out, n_dofs) — velocity derivative
        M_out:   (n_samples, n_dofs_out, n_dofs) — mass matrix (= dtau/da)
    """
    nd = positions.shape[1]
    ns = positions.shape[0]
    nb = tree_info["n_bodies"]
    n_out = n_dofs_out
    parent_body = tree_info["parent_body"]
    child_body = tree_info["child_body"]
    descendants = tree_info["descendants"]
    ancestors = tree_info["ancestors"]

    # Joint motion subspaces S[d]: (6,) — [axis; 0] for revolute joints
    S = [np.concatenate([tree_info["joint_axes"][d], np.zeros(3)]) for d in range(nd)]

    # Precompute 6×6 cross-product matrices for each joint axis (constant, used in hot loops)
    Sx = [_motion_cross_matrix(S[d]) for d in range(nd)]  # [S_d]×
    Sfx = [_force_cross_matrix(S[d]) for d in range(nd)]  # [S_d]×*

    # --- Precompute constant origin transforms (rotation part only) ---
    R_origin = [_rpy_to_rotation(tree_info["joint_rpy"][d]) for d in range(nd)]

    # --- Compute per-joint spatial transforms for all samples ---
    # X[d]: (ns, 6, 6) — Plücker transform from parent body to child body
    # Xf[d]: (ns, 6, 6) — transpose (force transform child→parent)
    X: list[np.ndarray] = [np.empty(0)] * nd
    Xf: list[np.ndarray] = [np.empty(0)] * nd

    for d in range(nd):
        axis = tree_info["joint_axes"][d]
        p_origin = tree_info["joint_xyz"][d]
        angles = positions[:, d]  # (ns,)

        # Rodrigues: R_joint(q) = I + sin(q)*K + (1-cos(q))*K²
        K = _skew3(axis)
        K2 = K @ K
        sin_q = np.sin(angles)[:, None, None]
        cos_q = np.cos(angles)[:, None, None]
        R_joint = np.eye(3) + sin_q * K + (1 - cos_q) * K2  # (ns, 3, 3)

        # R_cp = (R_origin @ R_joint)^T — rotation from parent to child frame
        R_pc = np.einsum("ij,njk->nik", R_origin[d], R_joint)  # (ns, 3, 3)
        R_cp = np.swapaxes(R_pc, -2, -1)  # (ns, 3, 3)

        # Plücker: ^child X_parent = [[E, 0], [-E·[p]×, E]]
        # where E = R_cp (rotation parent→child), p = p_origin (child origin in parent frame)
        # Ref: Featherstone (2008) Eq. 2.27
        p_skew = _skew3(p_origin)  # (3, 3), constant for all samples

        X_d = np.zeros((ns, 6, 6))
        X_d[:, :3, :3] = R_cp
        X_d[:, 3:, 3:] = R_cp
        # Lower-left block: -R_cp @ [p_origin]×
        X_d[:, 3:, :3] = -np.einsum("nij,jk->nik", R_cp, p_skew)

        X[d] = X_d
        Xf[d] = np.swapaxes(X_d, -2, -1)  # force transform = X^T

    # --- Forward pass: body velocities and accelerations ---
    v = [np.zeros((ns, 6)) for _ in range(nb)]
    a = [np.zeros((ns, 6)) for _ in range(nb)]
    vJ_arr = [np.zeros((ns, 6)) for _ in range(nd)]

    # Root: v=0, a=-gravity
    a_gravity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81])
    a[0] = np.tile(-a_gravity, (ns, 1))

    for d in range(nd):
        pi = parent_body[d]
        ci = child_body[d]

        # Joint velocity: vJ = S * dq
        vJ_d = np.outer(velocities[:, d], S[d])  # (ns, 6)
        vJ_arr[d] = vJ_d

        # v_child = X @ v_parent + vJ
        v_from_parent = np.einsum("nij,nj->ni", X[d], v[pi])
        v[ci] = v_from_parent + vJ_d

        # Coriolis: c = v × vJ (spatial motion cross product)
        w_ci = v[ci][:, :3]
        vo_ci = v[ci][:, 3:]
        wJ = vJ_d[:, :3]
        vJ_lin = vJ_d[:, 3:]
        c = np.empty((ns, 6))
        c[:, :3] = _batch_cross(w_ci, wJ)
        c[:, 3:] = _batch_cross(vo_ci, wJ) + _batch_cross(w_ci, vJ_lin)

        # a_child = X @ a_parent + S*ddq + c
        a_from_parent = np.einsum("nij,nj->ni", X[d], a[pi])
        a[ci] = a_from_parent + np.outer(accelerations[:, d], S[d]) + c

    # --- Compute forces and precompute I@v ---
    f = [np.zeros((ns, 6)) for _ in range(nb)]
    Iv = [np.zeros((ns, 6)) for _ in range(nb)]

    for i in range(nb):
        I_i = body_I[i]  # (6, 6), symmetric
        Iv[i] = v[i] @ I_i  # (ns, 6) — batch since I is symmetric
        Ia = a[i] @ I_i  # (ns, 6)
        # f = I@a + v ×* (I@v): force cross product
        w_i = v[i][:, :3]
        vo_i = v[i][:, 3:]
        Iv_ang = Iv[i][:, :3]
        Iv_lin = Iv[i][:, 3:]
        fc = np.empty((ns, 6))
        fc[:, :3] = _batch_cross(w_i, Iv_ang) + _batch_cross(vo_i, Iv_lin)
        fc[:, 3:] = _batch_cross(w_i, Iv_lin)
        f[i] = Ia + fc

    # --- Backward pass: accumulate subtree forces ---
    f_total = [fi.copy() for fi in f]
    for d in range(nd - 1, -1, -1):
        f_total[parent_body[d]] += np.einsum("nij,nj->ni", Xf[d], f_total[child_body[d]])

    # --- Derivative computation ---
    dtau_dq = np.zeros((ns, n_out, nd))
    dtau_dv = np.zeros((ns, n_out, nd))
    M_out = np.zeros((ns, n_out, nd))

    def _propagate_perturbation(d_start: int, dv0: np.ndarray, da0: np.ndarray) -> list[np.ndarray]:
        """Forward-propagate velocity/acceleration perturbation from joint d_start
        through the subtree, compute force perturbations, backward-accumulate.
        Handles tree-structured robots (not just serial chains)."""
        # Per-body perturbation state (only for bodies in the subtree of d_start)
        body_dv: dict[int, np.ndarray] = {}
        body_da: dict[int, np.ndarray] = {}
        body_df: list[np.ndarray | None] = [None] * nb

        ci = child_body[d_start]
        body_dv[ci] = dv0
        body_da[ci] = da0

        def _compute_force_pert(body_idx: int) -> np.ndarray:
            """Compute df = I@da + dv×*(I@v) + v×*(I@dv) at a body."""
            dv_b = body_dv[body_idx]
            da_b = body_da[body_idx]
            I_b = body_I[body_idx]
            Ida = da_b @ I_b
            Idv = dv_b @ I_b
            dw = dv_b[:, :3]
            dvo = dv_b[:, 3:]
            w_b = v[body_idx][:, :3]
            vo_b = v[body_idx][:, 3:]
            Iv_a = Iv[body_idx][:, :3]
            Iv_l = Iv[body_idx][:, 3:]
            Idv_a = Idv[:, :3]
            Idv_l = Idv[:, 3:]
            df = np.empty((ns, 6))
            df[:, :3] = (
                Ida[:, :3] + np.cross(dw, Iv_a) + np.cross(dvo, Iv_l) + np.cross(w_b, Idv_a) + np.cross(vo_b, Idv_l)
            )
            df[:, 3:] = Ida[:, 3:] + np.cross(dw, Iv_l) + np.cross(w_b, Idv_l)
            return df

        # Force at the starting body
        body_df[ci] = _compute_force_pert(ci)

        # Propagate to all descendants (in tree order — descendants list is pre-ordered)
        for dd in descendants[d_start]:
            ci_dd = child_body[dd]
            pi_dd = parent_body[dd]  # parent body of joint dd

            # Get perturbation from parent body and propagate through joint dd's transform
            dv_parent = body_dv[pi_dd]
            da_parent = body_da[pi_dd]
            dv_new = np.einsum("nij,nj->ni", X[dd], dv_parent)
            da_new = np.einsum("nij,nj->ni", X[dd], da_parent)
            # da += dv × vJ (Coriolis coupling)
            da_new[:, :3] += np.cross(dv_new[:, :3], vJ_arr[dd][:, :3])
            da_new[:, 3:] += np.cross(dv_new[:, 3:], vJ_arr[dd][:, :3]) + np.cross(dv_new[:, :3], vJ_arr[dd][:, 3:])

            body_dv[ci_dd] = dv_new
            body_da[ci_dd] = da_new
            body_df[ci_dd] = _compute_force_pert(ci_dd)

        # Backward accumulation (reverse topological order is range(nd-1, ..., -1))
        df_acc: list[np.ndarray] = [bd.copy() if bd is not None else np.zeros((ns, 6)) for bd in body_df]
        for dd in range(nd - 1, -1, -1):
            df_acc[parent_body[dd]] += np.einsum("nij,nj->ni", Xf[dd], df_acc[child_body[dd]])
        return df_acc

    # --- dtau/dq: perturb each joint angle ---
    for d in range(nd):
        ci = child_body[d]
        # Velocity perturbation at joint d: S × (v_child - vJ) = S × (X @ v_parent)
        v_from_parent = v[ci] - vJ_arr[d]  # = X[d] @ v[parent_body[d]]
        Sd = S[d]

        # dv = -[S]× @ v_from_parent: use precomputed 6×6 matrix, single BLAS matmul
        neg_Sx_d = -Sx[d]  # -[S_d]×
        dv0 = v_from_parent @ neg_Sx_d.T  # (ns, 6)

        # da = -[S]× @ a_from_parent + [dv]× @ vJ
        # Recompute a_from_parent = X @ a_parent = a[ci] - S*ddq - c
        w_ci = v[ci][:, :3]
        vo_ci = v[ci][:, 3:]
        wJ = vJ_arr[d][:, :3]
        vJ_lin = vJ_arr[d][:, 3:]
        c_d = np.empty((ns, 6))
        c_d[:, :3] = np.cross(w_ci, wJ)
        c_d[:, 3:] = np.cross(vo_ci, wJ) + np.cross(w_ci, vJ_lin)
        a_from_parent = a[ci] - np.outer(accelerations[:, d], Sd) - c_d

        da0 = a_from_parent @ neg_Sx_d.T  # -[S]× @ a_from_parent
        # Add [dv]× @ vJ = -(vJ × dv) component-wise
        da0[:, :3] += np.cross(dv0[:, :3], vJ_arr[d][:, :3])
        da0[:, 3:] += np.cross(dv0[:, 3:], vJ_arr[d][:, :3]) + np.cross(dv0[:, :3], vJ_arr[d][:, 3:])

        df_acc = _propagate_perturbation(d, dv0, da0)

        # --- Transform perturbation correction (Featherstone Alg. 9.3) ---
        # When q_d changes, the spatial transform X_d changes. This affects how the
        # subtree force f_total[child_d] is propagated back to ancestor bodies.
        # The extra force at parent_body[d] is: Xf[d] @ (S_d ×* f_total[child_d])
        # where ×* is the spatial force cross product.
        ci_d = child_body[d]
        f_sub = f_total[ci_d]  # (ns, 6): subtree force at child body of joint d

        # S ×* f using precomputed 6×6 force cross matrix
        df_xform = f_sub @ Sfx[d].T  # (ns, 6)

        # Negate to match our sign convention (dX/dq = -S × X → force term is -S ×* f)
        # Actually: (dX/dq)^T @ f = X^T · [S]×* @ f, but dX/dq = -[S]× · X
        # So: (-[S]× · X)^T @ f = -X^T · [S]×^T @ f = X^T · [S]×* @ f
        # Transform to parent: Xf[d] @ df_xform
        df_at_parent = np.einsum("nij,nj->ni", Xf[d], df_xform)

        # Propagate backward through ancestor joints to root
        df_ancestor = df_at_parent
        for j in ancestors[d]:
            cj = child_body[j]
            # Add torque correction at ancestor joint j
            df_acc[cj] += df_ancestor
            # Continue propagating toward root
            df_ancestor = np.einsum("nij,nj->ni", Xf[j], df_ancestor)

        # Add correction to root (for base wrench)
        df_acc[0] += df_ancestor

        # Extract torques
        for j in range(nd):
            dtau_dq[:, fb + j, d] = df_acc[child_body[j]] @ S[j]
        if is_floating:
            # Reorder base wrench: Featherstone [moment;force] → iDynTree [force;moment]
            dtau_dq[:, :3, d] = df_acc[0][:, 3:]  # force (linear)
            dtau_dq[:, 3:6, d] = df_acc[0][:, :3]  # moment (angular)

    # --- dtau/dv: perturb each joint velocity ---
    for d in range(nd):
        ci = child_body[d]
        Sd = S[d]

        # dv = S (direct velocity perturbation)
        dv0 = np.tile(Sd, (ns, 1))

        # da = [v]× @ S: v varies per sample, S is constant.
        # For revolute: S = [s_ω; 0], so [v]× @ S = [ω × s_ω; v₀ × s_ω].
        # varying × const = varying @ skew(const)
        s_w = Sd[:3]
        sw_skew = _skew3(s_w)  # (3,3)
        da0 = np.empty((ns, 6))
        da0[:, :3] = v[ci][:, :3] @ sw_skew
        da0[:, 3:] = v[ci][:, 3:] @ sw_skew

        df_acc = _propagate_perturbation(d, dv0, da0)

        for j in range(nd):
            dtau_dv[:, fb + j, d] = df_acc[child_body[j]] @ S[j]
        if is_floating:
            dtau_dv[:, :3, d] = df_acc[0][:, 3:]
            dtau_dv[:, 3:6, d] = df_acc[0][:, :3]

    # --- dtau/da (mass matrix): perturb each joint acceleration ---
    # Since a appears linearly, da = S at joint d, propagated as pure acceleration (no velocity perturbation)
    for d in range(nd):
        dv0 = np.zeros((ns, 6))
        da0 = np.tile(S[d], (ns, 1))

        df_acc = _propagate_perturbation(d, dv0, da0)

        for j in range(nd):
            M_out[:, fb + j, d] = df_acc[child_body[j]] @ S[j]
        if is_floating:
            M_out[:, :3, d] = df_acc[0][:, 3:]
            M_out[:, 3:6, d] = df_acc[0][:, :3]

    return dtau_dq, dtau_dv, M_out


def _compute_sensitivities_analytical(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    w1_iner: np.ndarray,
    w2_iner: np.ndarray,
    params_iner: np.ndarray,
    u1: np.ndarray,
    u_last: np.ndarray,
    c1: float,
    c2: float,
    n_dofs_out: int,
    n_dofs: int,
    is_floating: bool,
    has_visc_friction: bool,
    w1_visc: np.ndarray,
    w2_visc: np.ndarray,
    params_visc: np.ndarray,
    fb: int,
    torque_absmax_idx: np.ndarray,
    sample_indices: list[int],
    n_samples: int,
    n_links: int,
    tree_info: dict,
    gravity_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Phase A sensitivities using analytical RNEA derivatives.

    Uses Featherstone spatial algebra with 6x6 spatial inertia matrices directly.
    This handles non-physical parameter vectors (from SVD contraction) without issues,
    unlike Pinocchio's (mass, COM, I_com) decomposition which requires positive mass.
    """
    nd = n_dofs

    if gravity_only:
        w1_iner = _expand_gravity_only_params(w1_iner, n_links)
        w2_iner = _expand_gravity_only_params(w2_iner, n_links)
        params_iner = _expand_gravity_only_params(params_iner, n_links)

    pos = positions[sample_indices]
    vel = velocities[sample_indices]
    acc = accelerations[sample_indices]

    # Build spatial inertias for each contraction vector
    body_I_w1 = _build_body_spatial_inertias(w1_iner, n_links, tree_info)
    body_I_w2 = _build_body_spatial_inertias(w2_iner, n_links, tree_info)

    # Compute RNEA derivatives for w1 and w2
    dtau_dq_w1, dtau_dv_w1, M_w1 = _compute_rnea_derivatives_batch(
        body_I_w1, tree_info, pos, vel, acc, is_floating, fb, n_dofs_out
    )
    dtau_dq_w2, dtau_dv_w2, M_w2 = _compute_rnea_derivatives_batch(
        body_I_w2, tree_info, pos, vel, acc, is_floating, fb, n_dofs_out
    )

    # Contract: sens[t, d] = c1 * u1_t @ dtau_w1[t, :, d] + c2 * u_last_t @ dtau_w2[t, :, d]
    sens_q = np.zeros((n_samples, nd))
    sens_dq = np.zeros((n_samples, nd))
    sens_ddq = np.zeros((n_samples, nd))

    for idx, t in enumerate(sample_indices):
        row_start = t * n_dofs_out
        u1_t = u1[row_start : row_start + n_dofs_out]
        u_last_t = u_last[row_start : row_start + n_dofs_out]

        # Position sensitivities
        sens_q[t, :] = c1 * (u1_t @ dtau_dq_w1[idx]) + c2 * (u_last_t @ dtau_dq_w2[idx])

        # Velocity sensitivities (with viscous friction correction)
        dv_w1 = dtau_dv_w1[idx].copy()
        dv_w2 = dtau_dv_w2[idx].copy()
        if has_visc_friction:
            for d_joint in range(nd):
                dv_w1[fb + d_joint, d_joint] += w1_visc[d_joint]
                dv_w2[fb + d_joint, d_joint] += w2_visc[d_joint]
        sens_dq[t, :] = c1 * (u1_t @ dv_w1) + c2 * (u_last_t @ dv_w2)

        # Acceleration sensitivities
        sens_ddq[t, :] = c1 * (u1_t @ M_w1[idx]) + c2 * (u_last_t @ M_w2[idx])

    # Torque gradient at absmax samples
    torque_grad_samples = set(int(idx) for idx in torque_absmax_idx)
    dtau_dq_out = np.zeros((nd, nd))
    dtau_ddq_state_out = np.zeros((nd, nd))
    dtau_dddq_out = np.zeros((nd, nd))

    # Check if any absmax samples are in our sample_indices
    needed_samples = torque_grad_samples.intersection(sample_indices)
    if needed_samples:
        body_I_p = _build_body_spatial_inertias(params_iner, n_links, tree_info)
        dtau_dq_p, dtau_dv_p, M_p = _compute_rnea_derivatives_batch(
            body_I_p, tree_info, pos, vel, acc, is_floating, fb, n_dofs_out
        )

        for idx, t in enumerate(sample_indices):
            if t in torque_grad_samples:
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        dtau_dq_out[n, :] = dtau_dq_p[idx, fb + n, :]
                        dtau_ddq_state_out[n, :] = dtau_dv_p[idx, fb + n, :]
                        if has_visc_friction:
                            dtau_ddq_state_out[n, n] += params_visc[n]
                        dtau_dddq_out[n, :] = M_p[idx, fb + n, :]

    return sens_q, sens_dq, sens_ddq, dtau_dq_out, dtau_ddq_state_out, dtau_dddq_out


def _dopt_gradient_worker_func(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute D-optimality regressor sensitivities via FD.

    Instead of contracting with SVD vectors (condition number), uses the Frobenius
    inner product with D-optimality weights: sens[t,d] = <W_t, dY_t/dq_d>.
    This is numerically stable at any condition number.
    """
    (
        sample_indices,
        positions,
        velocities,
        accelerations,
        W_iner,
        params_iner,
        n_dofs_out,
        epsilon,
        is_floating,
        row_slice_start,
        has_visc_friction,
        W_visc,
        params_visc,
        fb,
        torque_absmax_idx,
    ) = args

    model = _worker_state["model"]
    nd = model.num_dofs
    inv_eps = 1.0 / epsilon

    q_buf = iDynTree.JointPosDoubleArray(nd)
    dq_buf = iDynTree.JointDOFsDoubleArray(nd)
    ddq_buf = iDynTree.JointDOFsDoubleArray(nd)
    base_acc = iDynTree.Vector6()
    reg_buf = iDynTree.MatrixDynSize()
    fb_identity = iDynTree.Transform.Identity() if is_floating else None
    fb_zero_twist = iDynTree.Twist.Zero() if is_floating else None
    row_slice = slice(None) if is_floating else slice(6, None)

    torque_grad_samples = set(int(idx) for idx in torque_absmax_idx)

    n_chunk = len(sample_indices)
    sens_q = np.zeros((n_chunk, nd))
    sens_dq = np.zeros((n_chunk, nd))
    sens_ddq = np.zeros((n_chunk, nd))
    torque_sens: dict[str, np.ndarray] = {}

    for ci, t in enumerate(sample_indices):
        pos_t = positions[t]
        vel_t = velocities[t]
        acc_t = accelerations[t]

        for i in range(nd):
            q_buf.setVal(i, float(pos_t[i]))
            dq_buf.setVal(i, float(vel_t[i]))
            ddq_buf.setVal(i, float(acc_t[i]))

        # Per-sample D-opt weight block (n_out × n_inertial)
        W_t = W_iner[t * n_dofs_out : (t + 1) * n_dofs_out]
        need_torque = t in torque_grad_samples

        # --- Baseline evaluation ---
        if is_floating:
            model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
        else:
            model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
        model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
        Y_base = reg_buf.toNumPy()[row_slice]
        score_base = np.sum(W_t * Y_base)
        if need_torque:
            Yp_base = Y_base @ params_iner

        # --- Position perturbations ---
        for d in range(nd):
            orig = float(pos_t[d])
            q_buf.setVal(d, orig + epsilon)
            if is_floating:
                model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
            else:
                model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Y_plus = reg_buf.toNumPy()[row_slice]
            q_buf.setVal(d, orig)

            sens_q[ci, d] = (np.sum(W_t * Y_plus) - score_base) * inv_eps

            if need_torque:
                dp = (Y_plus @ params_iner - Yp_base) * inv_eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"dq_{n}_{d}"] = dp[fb + n]

        # --- Velocity perturbations ---
        if is_floating:
            model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
        else:
            model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)

        for d in range(nd):
            orig = float(vel_t[d])
            dq_buf.setVal(d, orig + epsilon)
            if is_floating:
                model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
            else:
                model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Y_plus = reg_buf.toNumPy()[row_slice]
            dq_buf.setVal(d, orig)

            dopt_contrib = (np.sum(W_t * Y_plus) - score_base) * inv_eps
            # Viscous friction contribution (analytical, not captured by inertial regressor FD)
            if has_visc_friction:
                dopt_contrib += W_visc[t * n_dofs_out + fb + d]

            sens_dq[ci, d] = dopt_contrib

            if need_torque:
                dp = (Y_plus @ params_iner - Yp_base) * inv_eps
                if has_visc_friction:
                    dp[fb + d] += params_visc[d]
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"ddq_{n}_{d}"] = dp[fb + n]

        # --- Acceleration perturbations ---
        for d in range(nd):
            orig = float(acc_t[d])
            ddq_buf.setVal(d, orig + epsilon)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Y_plus = reg_buf.toNumPy()[row_slice]
            ddq_buf.setVal(d, orig)

            sens_ddq[ci, d] = (np.sum(W_t * Y_plus) - score_base) * inv_eps

            if need_torque:
                dp = (Y_plus @ params_iner - Yp_base) * inv_eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"dddq_{n}_{d}"] = dp[fb + n]

    return sens_q, sens_dq, sens_ddq, torque_sens


def _gradient_worker_init(urdf_file: str, config_dict: dict) -> None:
    """Initialize a worker process with its own iDynTree model.

    Called once when the worker is spawned. The model is cached in
    the module-level _worker_state dict for reuse across tasks.
    """
    import matplotlib

    matplotlib.use("Agg")  # prevent display in workers

    from identification.model import Model

    config_dict["jointNames"] = iDynTree.StringVector([])
    iDynTree.dofsListFromURDF(urdf_file, config_dict["jointNames"])
    config_dict["num_dofs"] = len(config_dict["jointNames"])

    model = Model(config_dict, urdf_file)
    _worker_state["model"] = model


def _gradient_worker_func(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Process a chunk of samples for the gradient computation.

    Each worker has its own iDynTree model (from _gradient_worker_init).
    Returns (sens_q_chunk, sens_dq_chunk, sens_ddq_chunk, torque_sens_dict).
    """
    (
        sample_indices,
        positions,
        velocities,
        accelerations,
        w1_iner,
        w2_iner,
        params_iner,
        u1,
        u_last,
        c1,
        c2,
        n_dofs_out,
        epsilon,
        is_floating,
        row_slice_start,
        has_visc_friction,
        w1_visc,
        w2_visc,
        params_visc,
        fb,
        torque_absmax_idx,
    ) = args

    model = _worker_state["model"]
    nd = model.num_dofs
    inv_eps = 1.0 / epsilon

    q_buf = iDynTree.JointPosDoubleArray(nd)
    dq_buf = iDynTree.JointDOFsDoubleArray(nd)
    ddq_buf = iDynTree.JointDOFsDoubleArray(nd)
    base_acc = iDynTree.Vector6()
    reg_buf = iDynTree.MatrixDynSize()
    fb_identity = iDynTree.Transform.Identity() if is_floating else None
    fb_zero_twist = iDynTree.Twist.Zero() if is_floating else None
    row_slice = slice(None) if is_floating else slice(6, None)

    torque_grad_samples = set(int(idx) for idx in torque_absmax_idx)

    n_chunk = len(sample_indices)
    sens_q = np.zeros((n_chunk, nd))
    sens_dq = np.zeros((n_chunk, nd))
    sens_ddq = np.zeros((n_chunk, nd))
    torque_sens: dict[str, np.ndarray] = {}  # "dtau_{state}_{n}_{d}" -> value

    for ci, t in enumerate(sample_indices):
        pos_t = positions[t]
        vel_t = velocities[t]
        acc_t = accelerations[t]

        for i in range(nd):
            q_buf.setVal(i, float(pos_t[i]))
            dq_buf.setVal(i, float(vel_t[i]))
            ddq_buf.setVal(i, float(acc_t[i]))

        row_start = t * n_dofs_out
        u1_t = u1[row_start : row_start + n_dofs_out]
        u_last_t = u_last[row_start : row_start + n_dofs_out]
        need_torque = t in torque_grad_samples

        # --- Baseline evaluation ---
        if is_floating:
            model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
        else:
            model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
        model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
        Y_base_view = reg_buf.toNumPy()[row_slice]
        Yw1_base = Y_base_view @ w1_iner
        Yw2_base = Y_base_view @ w2_iner
        if need_torque:
            Yp_base = Y_base_view @ params_iner

        # --- Position perturbations (forward difference) ---
        for d in range(nd):
            orig = float(pos_t[d])
            q_buf.setVal(d, orig + epsilon)
            if is_floating:
                model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
            else:
                model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Yw1_plus = reg_buf.toNumPy()[row_slice] @ w1_iner
            Yw2_plus = reg_buf.toNumPy()[row_slice] @ w2_iner
            q_buf.setVal(d, orig)

            dw1 = (Yw1_plus - Yw1_base) * inv_eps
            dw2 = (Yw2_plus - Yw2_base) * inv_eps
            sens_q[ci, d] = c1 * np.dot(u1_t, dw1) + c2 * np.dot(u_last_t, dw2)

            if need_torque:
                dp = (reg_buf.toNumPy()[row_slice] @ params_iner - Yp_base) * inv_eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"dq_{n}_{d}"] = dp[fb + n]

        # --- Velocity perturbations (forward difference) ---
        if is_floating:
            model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
        else:
            model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)

        for d in range(nd):
            orig = float(vel_t[d])
            dq_buf.setVal(d, orig + epsilon)
            if is_floating:
                model.kinDyn.setRobotState(fb_identity, q_buf, fb_zero_twist, dq_buf, model.gravity_vec)
            else:
                model.kinDyn.setRobotState(q_buf, dq_buf, model.gravity_vec)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Yw1_plus = reg_buf.toNumPy()[row_slice] @ w1_iner
            Yw2_plus = reg_buf.toNumPy()[row_slice] @ w2_iner
            dq_buf.setVal(d, orig)

            dw1 = (Yw1_plus - Yw1_base) * inv_eps
            dw2 = (Yw2_plus - Yw2_base) * inv_eps
            if has_visc_friction:
                dw1[fb + d] += w1_visc[d]
                dw2[fb + d] += w2_visc[d]

            sens_dq[ci, d] = c1 * np.dot(u1_t, dw1) + c2 * np.dot(u_last_t, dw2)

            if need_torque:
                dp = (reg_buf.toNumPy()[row_slice] @ params_iner - Yp_base) * inv_eps
                if has_visc_friction:
                    dp[fb + d] += params_visc[d]
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"ddq_{n}_{d}"] = dp[fb + n]

        # --- Acceleration perturbations (forward difference, no setState) ---
        for d in range(nd):
            orig = float(acc_t[d])
            ddq_buf.setVal(d, orig + epsilon)
            model.kinDyn.inverseDynamicsInertialParametersRegressor(base_acc, ddq_buf, reg_buf)
            Yw1_plus = reg_buf.toNumPy()[row_slice] @ w1_iner
            Yw2_plus = reg_buf.toNumPy()[row_slice] @ w2_iner
            ddq_buf.setVal(d, orig)

            dw1 = (Yw1_plus - Yw1_base) * inv_eps
            dw2 = (Yw2_plus - Yw2_base) * inv_eps
            sens_ddq[ci, d] = c1 * np.dot(u1_t, dw1) + c2 * np.dot(u_last_t, dw2)

            if need_torque:
                dp = (reg_buf.toNumPy()[row_slice] @ params_iner - Yp_base) * inv_eps
                for n in range(nd):
                    if torque_absmax_idx[n] == t:
                        torque_sens[f"dddq_{n}_{d}"] = dp[fb + n]

    return sens_q, sens_dq, sens_ddq, torque_sens


def _get_gradient_pool(n_jobs: int, urdf: str, config: dict) -> multiprocessing.pool.Pool:
    """Get or create the worker pool for parallel gradient computation."""
    global _gradient_pool, _gradient_pool_size
    if _gradient_pool is None or _gradient_pool_size != n_jobs:
        if _gradient_pool is not None:
            _gradient_pool.terminate()
        # Strip non-picklable objects from config
        config_clean = {k: v for k, v in config.items() if k != "jointNames"}
        _gradient_pool = multiprocessing.Pool(
            n_jobs,
            initializer=_gradient_worker_init,
            initargs=(urdf, config_clean),
        )
        _gradient_pool_size = n_jobs
    return _gradient_pool


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


def _collision_fd_single_pair(
    optimizer: TrajectoryOptimizer,
    pair_idx: int,
    q_star: np.ndarray,
    nd: int,
    epsilon: float,
    inv_2eps: float,
) -> np.ndarray:
    """Compute d(distance)/dq for a single collision pair via central finite differences.

    Used as fallback when capsule geometry is not available for a pair.
    """
    ddist = np.zeros(nd)
    l0, l1 = optimizer._collision_pairs[pair_idx]
    for j in range(nd):
        q_p = q_star.copy()
        q_p[j] += epsilon
        optimizer.setCollisionRobotState(q_p)
        d_plus = optimizer.getLinkDistance(l0, l1, q_p)

        q_m = q_star.copy()
        q_m[j] -= epsilon
        optimizer.setCollisionRobotState(q_m)
        d_minus = optimizer.getLinkDistance(l0, l1, q_m)
        ddist[j] = (d_plus - d_minus) * inv_2eps
    return ddist


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

    # ---- Phase 1: Regularized D-optimality gradient weights ----
    # Objective: f = -log(det(Y^T Y + δI)) * scale
    # Gradient: df/dY = -2 * scale * Y @ (Y^T Y + δI)^{-1}
    #                 = -2 * scale * U @ diag(s_i / (s_i² + δ)) @ V^T
    # The regularization δ ensures all terms in the sum are bounded by 1/(2√δ),
    # preventing numerical catastrophe from near-zero singular values.
    dopt_scale = cache["dopt_scale"]
    dopt_reg = config.get("doptRegularization", 1e-4)
    U, S, Vt = np.linalg.svd(YBase, full_matrices=False)
    delta = dopt_reg * S[0] ** 2  # δ relative to largest eigenvalue λ_max = s_max²
    # Regularized weights: s_i / (s_i² + δ) instead of 1/s_i
    reg_weights = S / (S**2 + delta)
    R_dopt = -2.0 * dopt_scale * ((U * reg_weights[np.newaxis, :]) @ Vt)

    proj = _get_projection_matrix(model)

    # D-optimality weight in standard parameter space: W_std = R_dopt @ Pb^T
    W_std = R_dopt @ proj.T  # shape (n_samples * n_dofs_out, n_identified_params)

    # Parameter vector for torque gradient (matches regressor column order)
    params = model.xStdModel[model.identified_params]

    # ---- Phase 2: wf derivatives (numerical FD of trajectory) ----
    dq_dwf, ddq_dwf, dddq_dwf = _compute_wf_derivatives(optimizer, times)

    # ---- Phase A: D-optimality regressor sensitivities via FD ----
    epsilon = config.get("analyticalGradientEpsilon", 1e-7)
    inv_2eps = 1.0 / (2.0 * epsilon)  # used by collision gradient FD
    is_floating = model.opt["floatingBase"]

    n_inertial = model.num_model_params
    if model.opt.get("identifyGravityParamsOnly", False):
        n_inertial -= len(model.inertia_params)
    W_iner = W_std[:, :n_inertial]
    params_iner = params[:n_inertial]

    has_friction = model.opt.get("identifyFriction", False)
    has_visc_friction = has_friction and not model.opt.get("identifyGravityParamsOnly", False)
    params_visc_arr = np.zeros(nd)
    # D-opt weight for viscous friction: d(v*Fv_d)/d(dq_d) = Fv_d at row (fb+d),
    # so the weight is W_std[t*n_out+fb+d, visc_col_d]
    W_visc = np.zeros(n_samples * n_dofs_out)
    if has_visc_friction:
        visc_offset = n_inertial + nd  # skip inertial + coulomb friction columns
        for d in range(nd):
            if visc_offset + d < W_std.shape[1]:
                for t in range(n_samples):
                    W_visc[t * n_dofs_out + fb + d] = W_std[t * n_dofs_out + fb + d, visc_offset + d]
        params_visc_arr = params[visc_offset : visc_offset + nd]

    torque_absmax_idx = cache["torque_absmax_idx"]
    subsample = config.get("analyticalGradientSubsample", 1)
    sample_indices = list(range(0, n_samples, subsample))

    n_jobs = config.get("analyticalGradientJobs", 1)
    if n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 1) - 2)

    # D-optimality FD worker
    worker_common = (
        positions,
        velocities,
        accelerations,
        W_iner,
        params_iner,
        n_dofs_out,
        epsilon,
        is_floating,
        0 if is_floating else 6,
        has_visc_friction,
        W_visc,
        params_visc_arr,
        fb,
        torque_absmax_idx,
    )

    if n_jobs > 1 and len(sample_indices) > n_jobs:
        grad_pool = _get_gradient_pool(n_jobs, config["urdf"], config)
        chunks = np.array_split(sample_indices, n_jobs)
        fd_work_items = [(list(chunk), *worker_common) for chunk in chunks]
        results = grad_pool.map(_dopt_gradient_worker_func, fd_work_items)
    else:
        if "model" not in _worker_state:
            _worker_state["model"] = model
        results = [_dopt_gradient_worker_func((sample_indices, *worker_common))]
        chunks = [np.array(sample_indices)]

    # Aggregate results
    sens_q = np.zeros((n_samples, nd))
    sens_dq = np.zeros((n_samples, nd))
    sens_ddq = np.zeros((n_samples, nd))
    dtau_dq = np.zeros((nd, nd))
    dtau_ddq_state = np.zeros((nd, nd))
    dtau_dddq = np.zeros((nd, nd))

    for chunk_idx, (sq, sdq, sddq, torque_dict) in enumerate(results):
        chunk = list(chunks[chunk_idx])
        for ci, t in enumerate(chunk):
            sens_q[t] = sq[ci]
            sens_dq[t] = sdq[ci]
            sens_ddq[t] = sddq[ci]
        for key, val in torque_dict.items():
            parts = key.split("_")
            state_type, n_i, d_i = parts[0], int(parts[1]), int(parts[2])
            if state_type == "dq":
                dtau_dq[n_i, d_i] = val
            elif state_type == "ddq":
                dtau_ddq_state[n_i, d_i] = val
            elif state_type == "dddq":
                dtau_dddq[n_i, d_i] = val

    if subsample > 1:
        sens_q *= subsample
        sens_dq *= subsample
        sens_ddq *= subsample

    # ---- Phase B: Chain rule with trajectory Jacobians (vectorized) ----
    obj_grad = np.zeros(n_vars)
    wf = optimizer.trajectory.w_f_global
    use_deg = config["useDeg"]
    deg_factor = np.pi / 180.0 if use_deg else 1.0
    # cond_scale and log10 scaling are already folded into c1/c2, so no extra multiplier needed

    # wf contribution: sum over all samples and joints
    obj_grad[0] += np.sum(sens_q * dq_dwf) + np.sum(sens_dq * ddq_dwf) + np.sum(sens_ddq * dddq_dwf)

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
                obj_grad[vi_a] += np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val)

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
                obj_grad[vi_b] += np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val)

            # q0: only position, through q_center
            q0_vi = q0_start + d
            obj_grad[q0_vi] += np.sum(s_q) * deg_factor

        else:
            # Unbounded (classic Swevers) mode
            inv_wl = 1.0 / wl
            for l_idx in range(nf_d):
                vi_a = a_offset + l_idx
                dq_val = sin_wlt[:, l_idx] * inv_wl[l_idx]
                ddq_val = cos_wlt[:, l_idx]
                dddq_val = -wl[l_idx] * sin_wlt[:, l_idx]
                obj_grad[vi_a] += np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val)

                vi_b = b_offset + l_idx
                dq_val = -cos_wlt[:, l_idx] * inv_wl[l_idx]
                ddq_val = sin_wlt[:, l_idx]
                dddq_val = wl[l_idx] * cos_wlt[:, l_idx]
                obj_grad[vi_b] += np.dot(s_q, dq_val) + np.dot(s_dq, ddq_val) + np.dot(s_ddq, dddq_val)

            q0_vi = q0_start + d
            obj_grad[q0_vi] += np.sum(s_q) * nf_d * deg_factor

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
        use_capsule = config.get("collisionMode", "convex") == "capsule" and optimizer._capsules

        # Group pairs by their argmin sample for efficiency
        sample_pairs: dict[int, list[int]] = {}
        for g_cnt in coll_cache:
            t_star = coll_cache[g_cnt][0]
            sample_pairs.setdefault(t_star, []).append(g_cnt)

        for t_star, pair_indices in sample_pairs.items():
            q_star = positions[t_star].copy()
            optimizer.setCollisionRobotState(q_star)

            if use_capsule:
                # Analytical capsule gradients — no FD needed
                for gi in pair_indices:
                    l0, l1 = optimizer._collision_pairs[gi]
                    if l0 in optimizer._capsules and l1 in optimizer._capsules:
                        _, ddist_dq_vec = optimizer.getCapsuleDistanceAndGradient(l0, l1)
                    else:
                        # fallback to FD for pairs without capsules
                        ddist_dq_vec = _collision_fd_single_pair(optimizer, gi, q_star, nd, epsilon, inv_2eps)
                    c_idx = offset + gi
                    for d_joint in range(nd):
                        dq_row, _, _ = _get_traj_jac(d_joint, t_star)
                        con_grad[c_idx, :] += ddist_dq_vec[d_joint] * dq_row
            else:
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
        print(f"Analytical gradient computed (cond={cache['cond']:.1e}, |grad|={np.linalg.norm(obj_grad):.3e})")

    return obj_grad, con_grad
