"""Capsule-based collision detection with analytical distance gradients.

A capsule is a cylinder with hemispherical caps, fully described by a line segment
(two endpoints) plus a radius. The distance between two capsules reduces to the
distance between two line segments, which has a closed-form expression — just a few
dot products, no iterative solver needed.

This enables analytical gradients d(distance)/d(q) by differentiating through forward
kinematics, eliminating the need for finite-difference perturbations used with FCL.

Based on the approach described in:
K. Ayusawa, A. Rioux, E. Yoshida, G. Venture, M. Gautier: "Generating
Persistently Exciting Trajectory Based on Condition Number Optimization,"
IEEE International Conference on Robotics and Automation (ICRA), Singapore,
pp. 6518–6524, 2017.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from identification.helpers import URDFHelpers


@dataclass
class Capsule:
    """A capsule collision primitive for a robot link.

    Defined by two endpoints in the link's local frame and a radius.
    When p0 == p1, this degenerates to a sphere.
    """

    link_name: str
    p0_local: np.ndarray  # (3,) first endpoint in link frame
    p1_local: np.ndarray  # (3,) second endpoint in link frame
    radius: float


def _parse_origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Parse <origin xyz="..." rpy="..."> into position and rotation matrix."""
    if element is None:
        return np.zeros(3), np.eye(3)

    xyz = element.attrib.get("xyz", "0 0 0")
    rpy = element.attrib.get("rpy", "0 0 0")
    pos = np.array([float(v) for v in xyz.split()])
    rpy_vals = [float(v) for v in rpy.split()]

    # RPY to rotation matrix (fixed-axis XYZ convention)
    cr, sr = np.cos(rpy_vals[0]), np.sin(rpy_vals[0])
    cp, sp = np.cos(rpy_vals[1]), np.sin(rpy_vals[1])
    cy, sy = np.cos(rpy_vals[2]), np.sin(rpy_vals[2])
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return pos, R


def _capsule_from_cylinder(
    origin_pos: np.ndarray, origin_rot: np.ndarray, length: float, radius: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a URDF cylinder to capsule endpoints in link frame.

    Cylinder axis is along local Z, centered at the origin.
    """
    half = length / 2.0
    p0_local = origin_pos + origin_rot @ np.array([0.0, 0.0, -half])
    p1_local = origin_pos + origin_rot @ np.array([0.0, 0.0, half])
    return p0_local, p1_local, radius


def _capsule_from_sphere(origin_pos: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a URDF sphere to a degenerate capsule (both endpoints at center)."""
    return origin_pos.copy(), origin_pos.copy(), radius


def _capsule_from_box(
    origin_pos: np.ndarray, origin_rot: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a box to a capsule along its longest axis.

    Endpoints are pulled inward by the radius so the capsule surface
    (segment + sphere caps) does not extend beyond the box bounds.
    Radius is the larger of the two shorter half-extents.
    """
    axis_idx = int(np.argmax(size))
    half_len = size[axis_idx] / 2.0
    direction = np.zeros(3)
    direction[axis_idx] = 1.0
    # radius from the larger of the two shorter half-extents
    other_dims = np.delete(size, axis_idx)
    radius = float(np.max(other_dims) / 2.0)
    # pull endpoints inward so capsule surface aligns with box face
    inward = min(radius, half_len)  # don't pull past center
    p0_local = origin_pos + origin_rot @ (-(half_len - inward) * direction)
    p1_local = origin_pos + origin_rot @ ((half_len - inward) * direction)
    return p0_local, p1_local, radius


def _merge_capsule_primitives(
    endpoints: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Merge multiple collision primitives into a single enclosing capsule.

    Finds the two most distant primitive endpoints and uses the max radius.
    The segment endpoints are then pulled inward by the radius so that the
    capsule *surface* (segment + radius) reaches the original extreme points
    rather than extending beyond them. This prevents neighboring links'
    capsules from overlapping at shared joints.
    """
    if len(endpoints) == 1:
        return endpoints[0]

    # Collect all primitive centers and extreme points
    all_points: list[np.ndarray] = []
    max_radius = 0.0
    for p0, p1, r in endpoints:
        all_points.append(p0)
        all_points.append(p1)
        max_radius = max(max_radius, r)

    points = np.array(all_points)

    # Find the two most distant points (approximate diameter)
    best_dist = 0.0
    best_i, best_j = 0, 1
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d > best_dist:
                best_dist = d
                best_i, best_j = i, j

    p0 = points[best_i].copy()
    p1 = points[best_j].copy()

    # Pull endpoints inward by the radius so the capsule surface (not the
    # segment center) aligns with the original extreme points. This avoids
    # the capsule protruding beyond the link into the neighbor's space.
    axis = p1 - p0
    axis_len = float(np.linalg.norm(axis))
    if axis_len > 2.0 * max_radius:
        axis_unit = axis / axis_len
        p0 = p0 + max_radius * axis_unit
        p1 = p1 - max_radius * axis_unit

    return p0, p1, max_radius


def fit_capsules_from_urdf(
    urdf_file: str,
    link_names: list[str],
    urdf_helpers: URDFHelpers,
    radius_scale: float = 1.0,
) -> dict[str, Capsule]:
    """Parse collision geometry from URDF and produce capsule approximations per link.

    Handles cylinder, sphere, box primitives directly. For mesh-only links, loads the
    mesh and fits a capsule from the axis-aligned bounding box.

    Args:
        urdf_file: Path to the URDF file.
        link_names: List of link names to process.
        urdf_helpers: URDFHelpers instance for mesh path resolution.
        radius_scale: Scale factor for capsule radii (default 1.0). Values < 1.0
            produce tighter capsules at the cost of not fully enclosing the geometry.
            Useful for compact humanoids where the conservative approximation causes
            false collisions.

    Returns:
        Dictionary mapping link name to Capsule. Links without collision geometry
        are omitted.
    """
    tree = urdf_helpers.parseURDF(urdf_file)
    capsules: dict[str, Capsule] = {}

    for link_elem in tree.findall("link"):
        link_name = link_elem.attrib["name"]
        if link_name not in link_names:
            continue

        collision_elems = link_elem.findall("collision")
        if not collision_elems:
            continue

        primitives: list[tuple[np.ndarray, np.ndarray, float]] = []

        for coll in collision_elems:
            origin_pos, origin_rot = _parse_origin(coll.find("origin"))
            geom = coll.find("geometry")
            if geom is None:
                continue

            cyl = geom.find("cylinder")
            sph = geom.find("sphere")
            box = geom.find("box")
            mesh_elem = geom.find("mesh")

            if cyl is not None:
                length = float(cyl.attrib["length"])
                radius = float(cyl.attrib["radius"])
                primitives.append(_capsule_from_cylinder(origin_pos, origin_rot, length, radius))
            elif sph is not None:
                radius = float(sph.attrib["radius"])
                primitives.append(_capsule_from_sphere(origin_pos, radius))
            elif box is not None:
                size = np.array([float(v) for v in box.attrib["size"].split()])
                primitives.append(_capsule_from_box(origin_pos, origin_rot, size))
            elif mesh_elem is not None:
                cap = _capsule_from_mesh(urdf_file, link_name, mesh_elem, origin_pos, origin_rot, urdf_helpers)
                if cap is not None:
                    primitives.append(cap)

        if primitives:
            p0, p1, r = _merge_capsule_primitives(primitives)
            capsules[link_name] = Capsule(link_name=link_name, p0_local=p0, p1_local=p1, radius=r * radius_scale)

    return capsules


def _capsule_from_mesh(
    urdf_file: str,
    link_name: str,
    mesh_elem: ET.Element,
    origin_pos: np.ndarray,
    origin_rot: np.ndarray,
    urdf_helpers: URDFHelpers,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit a capsule from a collision mesh using its axis-aligned bounding box.

    Uses the AABB of the scaled mesh (same as getBoundingBox), then fits a capsule
    along the longest axis with radius = max of the two shorter half-extents. This
    produces capsules that match the bounding box visualization exactly.
    """
    import os

    import trimesh

    # resolve mesh path using the same logic as the rest of the codebase
    mesh_path = urdf_helpers.getCollisionMeshPath(urdf_file, link_name)
    if mesh_path is None or not os.path.exists(mesh_path):
        mesh_path = urdf_helpers.getMeshPath(urdf_file, link_name)
    if mesh_path is None or not os.path.exists(mesh_path):
        return None

    mesh = trimesh.load_mesh(mesh_path)

    # apply per-axis scaling from URDF
    scale_str = mesh_elem.attrib.get("scale", "1 1 1")
    scale = np.array([float(s) for s in scale_str.split()])

    # AABB in mesh-local space (matches what getBoundingBox computes)
    bounds = mesh.bounding_box.bounds * scale  # (2, 3): min, max
    # ensure min < max for each axis (negative scaling can flip order)
    for ax in range(3):
        if bounds[0, ax] > bounds[1, ax]:
            bounds[0, ax], bounds[1, ax] = bounds[1, ax], bounds[0, ax]

    box_size = bounds[1] - bounds[0]  # (3,) extents
    box_center = 0.5 * (bounds[0] + bounds[1])  # center in local mesh space

    # apply collision origin transform to the box center
    box_center_link = origin_rot @ box_center + origin_pos

    # fit capsule from box: same logic as _capsule_from_box
    return _capsule_from_box(box_center_link, origin_rot, box_size)


# --------------------------------------------------------------------------
# Segment-segment distance (closed-form)
# --------------------------------------------------------------------------


def segment_segment_distance(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[float, float, float]:
    """Compute closest distance between two line segments and the parameters at closest approach.

    Segments are A(s) = a0 + s*(a1-a0) and B(t) = b0 + t*(b1-b0), with s,t in [0,1].

    Based on the algorithm from Ericson, "Real-Time Collision Detection", 2004, Section 5.1.9.

    Returns:
        (distance, s_star, t_star) where s_star and t_star are the parameters
        at the closest points on segments A and B respectively.
    """
    d1 = a1 - a0  # direction of segment A
    d2 = b1 - b0  # direction of segment B
    r = a0 - b0

    a = float(d1 @ d1)  # squared length of A
    e = float(d2 @ d2)  # squared length of B
    f = float(d2 @ r)

    EPSILON = 1e-10

    # both segments degenerate to points
    if a <= EPSILON and e <= EPSILON:
        return float(np.linalg.norm(r)), 0.0, 0.0

    if a <= EPSILON:
        # segment A degenerates to a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPSILON:
            # segment B degenerates to a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # general case
            b = float(d1 @ d2)
            denom = a * e - b * b  # always >= 0

            # if segments not parallel, compute closest point on line A to line B
            # and clamp to segment A. Otherwise pick arbitrary s (0).
            if denom > EPSILON:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            # compute point on line B closest to A(s)
            t = (b * s + f) / e

            # if t outside [0,1], clamp and recompute s
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest_a = a0 + s * d1
    closest_b = b0 + t * d2
    dist = float(np.linalg.norm(closest_a - closest_b))
    return dist, float(s), float(t)


def capsule_distance(
    cap_a: Capsule,
    cap_b: Capsule,
    rot_a: np.ndarray,
    pos_a: np.ndarray,
    rot_b: np.ndarray,
    pos_b: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance between two capsules given their link world transforms.

    Args:
        cap_a, cap_b: Capsule primitives.
        rot_a, pos_a: 3x3 rotation and 3-vector position for cap_a's link.
        rot_b, pos_b: Same for cap_b's link.

    Returns:
        (distance, s_star, t_star, a0_world, a1_world, b0_world, b1_world)
        where distance is the capsule surface distance (negative means overlap).
    """
    a0_world = rot_a @ cap_a.p0_local + pos_a
    a1_world = rot_a @ cap_a.p1_local + pos_a
    b0_world = rot_b @ cap_b.p0_local + pos_b
    b1_world = rot_b @ cap_b.p1_local + pos_b

    seg_dist, s_star, t_star = segment_segment_distance(a0_world, a1_world, b0_world, b1_world)
    dist = seg_dist - cap_a.radius - cap_b.radius
    return dist, s_star, t_star, a0_world, a1_world, b0_world, b1_world


# --------------------------------------------------------------------------
# Analytical gradient d(capsule_distance)/d(q)
# --------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for cross product: skew(v) @ u = v x u."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _point_jacobian(
    jac_full: np.ndarray,
    point_world: np.ndarray,
    link_origin_world: np.ndarray,
    col_start: int,
    num_dofs: int,
) -> np.ndarray:
    """Compute the Jacobian of a world-frame point on a link w.r.t. joint angles.

    Given the 6x(6+N) free-floating Jacobian from iDynTree (linear on top, angular
    on bottom), compute d(point_world)/d(q) for joint DOFs only.

    Args:
        jac_full: (6, 6+num_dofs) free-floating Jacobian from iDynTree.
        point_world: (3,) the point in world frame.
        link_origin_world: (3,) the link origin in world frame.
        col_start: Column offset for joint DOFs (6 for floating base, 0 for fixed).
        num_dofs: Number of joint DOFs.

    Returns:
        (3, num_dofs) Jacobian of the point w.r.t. joint angles.
    """
    J_lin = jac_full[:3, col_start : col_start + num_dofs]  # (3, N)
    J_ang = jac_full[3:, col_start : col_start + num_dofs]  # (3, N)
    r = point_world - link_origin_world  # (3,)
    # v_point = J_lin + omega x r = J_lin + skew(r)^T @ J_ang = J_lin - skew(r) @ J_ang
    # (since skew(a) @ b = a x b, and we want omega x r = -r x omega = -skew(r) @ omega)
    return J_lin + _skew(r) @ J_ang


def capsule_distance_and_gradient(
    cap_a: Capsule,
    cap_b: Capsule,
    kinDyn: Any,
    num_dofs: int,
    is_floating: bool,
) -> tuple[float, np.ndarray]:
    """Compute capsule distance and its analytical gradient w.r.t. joint angles.

    The gradient chain is:
        d(dist)/d(q) = d(dist)/d(endpoints) @ d(endpoints)/d(q)

    For the segment distance d = |p_A - p_B| with p_A = A(s*), p_B = B(t*):
        dd/da0 = (1-s*) * n,  dd/da1 = s* * n
        dd/db0 = -(1-t*) * n, dd/db1 = -t* * n
    where n = (p_A - p_B) / |p_A - p_B| is the unit separation vector.

    Each endpoint Jacobian d(p_world)/d(q) comes from the geometric Jacobian via
    iDynTree's getFrameFreeFloatingJacobian.

    Args:
        cap_a, cap_b: Capsule primitives.
        kinDyn: iDynTree KinDynComputations object (state already set).
        num_dofs: Number of joint DOFs.
        is_floating: Whether the robot has a floating base.

    Returns:
        (distance, d_distance_d_q) where d_distance_d_q has shape (num_dofs,).
    """
    from idyntree import bindings as iDynTree

    # get link transforms from current kinDyn state
    t_a = kinDyn.getWorldTransform(cap_a.link_name)
    rot_a = t_a.getRotation().toNumPy()
    pos_a = t_a.getPosition().toNumPy()

    t_b = kinDyn.getWorldTransform(cap_b.link_name)
    rot_b = t_b.getRotation().toNumPy()
    pos_b = t_b.getPosition().toNumPy()

    # compute capsule distance and closest-approach parameters
    dist, s_star, t_star, a0w, a1w, b0w, b1w = capsule_distance(cap_a, cap_b, rot_a, pos_a, rot_b, pos_b)

    # closest points on segments
    p_A = a0w + s_star * (a1w - a0w)
    p_B = b0w + t_star * (b1w - b0w)
    diff = p_A - p_B
    seg_dist = float(np.linalg.norm(diff))

    if seg_dist < 1e-12:
        # capsules at same point — gradient is undefined, return zero
        return dist, np.zeros(num_dofs)

    # unit separation vector (from B toward A)
    n = diff / seg_dist

    # get geometric Jacobians for both links
    col_start = 6 if is_floating else 0
    total_cols = 6 + num_dofs

    jac_a = iDynTree.MatrixDynSize(6, total_cols)
    kinDyn.getFrameFreeFloatingJacobian(cap_a.link_name, jac_a)
    jac_a_np = jac_a.toNumPy()

    jac_b = iDynTree.MatrixDynSize(6, total_cols)
    kinDyn.getFrameFreeFloatingJacobian(cap_b.link_name, jac_b)
    jac_b_np = jac_b.toNumPy()

    # point Jacobians: d(endpoint_world)/d(q), shape (3, num_dofs)
    J_a0 = _point_jacobian(jac_a_np, a0w, pos_a, col_start, num_dofs)
    J_a1 = _point_jacobian(jac_a_np, a1w, pos_a, col_start, num_dofs)
    J_b0 = _point_jacobian(jac_b_np, b0w, pos_b, col_start, num_dofs)
    J_b1 = _point_jacobian(jac_b_np, b1w, pos_b, col_start, num_dofs)

    # chain rule: dd_seg/dq = dd/d(endpoints) @ d(endpoints)/dq
    # dd/da0 = (1-s*)*n, dd/da1 = s*n, dd/db0 = -(1-t*)*n, dd/db1 = -t*n
    ddist_dq = (1.0 - s_star) * (n @ J_a0) + s_star * (n @ J_a1) - (1.0 - t_star) * (n @ J_b0) - t_star * (n @ J_b1)

    return dist, ddist_dq


def find_colliding_links_capsule(
    capsules: dict[str, Capsule],
    kinDyn: Any,
    link_names: list[str],
    ignore_links: set[str],
    ignore_pairs: list[list[str]],
    neighbors: dict[str, dict[str, list[Any]]] | None = None,
    max_kin_distance: int = 0,
) -> set[str]:
    """Find all links whose capsules are currently overlapping.

    Uses the same capsule_distance function that the trajectory optimizer uses
    when useCapsuleCollision is enabled.

    Args:
        capsules: dict mapping link name to Capsule.
        kinDyn: iDynTree KinDynComputations with robot state already set.
        link_names: list of link names to check.
        ignore_links: links to skip.
        ignore_pairs: link pairs to skip.
        neighbors: optional neighbor dict to skip adjacent links.
        max_kin_distance: if >0, only check pairs within this kinematic distance.

    Returns:
        set of link names involved in at least one capsule collision.
    """
    effective = [ln for ln in link_names if ln in capsules and ln not in ignore_links]

    # fetch transforms once
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ln in effective:
        t = kinDyn.getWorldTransform(ln)
        transforms[ln] = (t.getRotation().toNumPy(), t.getPosition().toNumPy())

    ignore_set = {(a, b) for a, b in ignore_pairs} | {(b, a) for a, b in ignore_pairs}

    def _kin_distance(start: str, target: str) -> int:
        """BFS shortest path in the kinematic tree."""
        if neighbors is None:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == target:
                return dist
            for nb in neighbors.get(current, {}).get("links", []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return 999

    colliding: set[str] = set()
    for i, l0 in enumerate(effective):
        for l1 in effective[i + 1 :]:
            if (l0, l1) in ignore_set:
                continue
            if neighbors is not None:
                if l0 in neighbors and l1 in neighbors[l0].get("links", []):
                    continue
                if l1 in neighbors and l0 in neighbors[l1].get("links", []):
                    continue
            if max_kin_distance > 0 and _kin_distance(l0, l1) > max_kin_distance:
                continue
            rot0, pos0 = transforms[l0]
            rot1, pos1 = transforms[l1]
            dist, _, _, _, _, _, _ = capsule_distance(capsules[l0], capsules[l1], rot0, pos0, rot1, pos1)
            if dist < 0:
                colliding.add(l0)
                colliding.add(l1)

    return colliding
