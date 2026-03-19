"""Collision checking between robot links using FCL.

Reusable by both the trajectory optimizer and the visualizer. Uses collision meshes
when available, convex hulls of visual meshes as fallback, and bounding boxes as
last resort.
"""

from __future__ import annotations

import os
from typing import Any

import fcl
import numpy as np

from identification.helpers import URDFHelpers


class CollisionChecker:
    """Check pairwise link collisions using FCL geometry."""

    def __init__(
        self,
        urdf_helpers: URDFHelpers,
        urdf_file: str,
        link_cuboid_hulls: dict[str, Any],
        link_names: list[str],
        scale_collision_hull: float = 1.0,
    ) -> None:
        self._urdf_helpers = urdf_helpers
        self._urdf_file = urdf_file
        self._link_cuboid_hulls = link_cuboid_hulls
        self._link_names = link_names
        self._scale = scale_collision_hull
        self._geom_cache: dict[str, tuple[Any, np.ndarray]] = {}

    def _get_geometry(self, link_name: str) -> tuple[Any, np.ndarray]:
        """Get or cache the FCL collision geometry and center offset for a link.

        Tries collision mesh first, then visual mesh (convex hull for large meshes),
        then falls back to bounding box."""
        if link_name in self._geom_cache:
            return self._geom_cache[link_name]

        # try loading collision mesh, then visual mesh
        mesh_path = self._urdf_helpers.getCollisionMeshPath(self._urdf_file, link_name)
        is_collision_mesh = mesh_path is not None
        if mesh_path is None:
            mesh_path = self._urdf_helpers.getMeshPath(self._urdf_file, link_name)

        if mesh_path is not None and os.path.exists(mesh_path):
            import trimesh

            mesh = trimesh.load_mesh(mesh_path)
            scale_parts = self._urdf_helpers.mesh_scaling.split()
            scale = np.array([float(s) for s in scale_parts])
            verts = np.array(mesh.vertices) * scale
            faces = np.array(mesh.faces)

            # compute convex hull for visual meshes (not collision meshes)
            if not is_collision_mesh and len(faces) > 100:
                hull = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
                verts = np.array(hull.vertices)
                faces = np.array(hull.faces)

            # fix face winding if any scale axis is negative (mirrored mesh)
            if np.prod(scale) < 0:
                faces = faces[:, ::-1]

            p = np.array(self._link_cuboid_hulls[link_name][1])
            bvh = fcl.BVHModel()
            bvh.beginModel(len(faces), len(verts))
            bvh.addSubModel(verts, faces)
            bvh.endModel()
            self._geom_cache[link_name] = (bvh, p)
        else:
            # fallback to bounding box
            s = self._scale if link_name in self._link_names else 1
            b = np.array(self._link_cuboid_hulls[link_name][0]) * s
            p = np.array(self._link_cuboid_hulls[link_name][1])
            center = 0.5 * (b[0] + b[1]) + p
            box = fcl.Box(*(b[1] - b[0]))
            self._geom_cache[link_name] = (box, center)

        return self._geom_cache[link_name]

    def check_distance(
        self,
        l0_name: str,
        l1_name: str,
        transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Check distance between two links given their world transforms.

        Args:
            l0_name: first link name
            l1_name: second link name
            transforms: dict mapping link name → (rotation_3x3, position_3)

        Returns:
            positive distance if separated, negative penetration depth if colliding
        """
        rot0, pos0 = transforms[l0_name]
        rot1, pos1 = transforms[l1_name]
        geom0, offset0 = self._get_geometry(l0_name)
        geom1, offset1 = self._get_geometry(l1_name)

        o0 = fcl.CollisionObject(geom0, fcl.Transform(rot0, pos0 + offset0))
        o1 = fcl.CollisionObject(geom1, fcl.Transform(rot1, pos1 + offset1))

        distance = fcl.distance(o0, o1, fcl.DistanceRequest(True), fcl.DistanceResult())

        # FCL returns 0 (not negative) for mesh-mesh overlap, so use collide() to confirm
        if distance <= 0:
            cr = fcl.CollisionRequest()
            cr.enable_contact = True
            c_result = fcl.CollisionResult()
            fcl.collide(o0, o1, cr, c_result)
            if c_result.is_collision:
                if len(c_result.contacts):
                    distance = -abs(c_result.contacts[0].penetration_depth)
                else:
                    distance = -0.001  # collision confirmed but no contact info
            else:
                # distance == 0 but no actual collision (just touching)
                distance = 0.0

        return distance

    def find_colliding_links(
        self,
        kinDyn: Any,
        link_names: list[str],
        ignore_links: set[str],
        ignore_pairs: list[list[str]],
        neighbors: dict[str, dict[str, list[Any]]] | None = None,
    ) -> set[str]:
        """Find all links that are currently colliding.

        Args:
            kinDyn: iDynTree KinDynComputations with robot state already set
            link_names: list of link names to check
            ignore_links: links to skip
            ignore_pairs: link pairs to skip
            neighbors: optional neighbor dict to skip adjacent links

        Returns:
            set of link names involved in at least one collision
        """
        # build transforms for all relevant links
        transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        effective_links = [ln for ln in link_names if ln in self._link_cuboid_hulls and ln not in ignore_links]
        for ln in effective_links:
            t = kinDyn.getWorldTransform(ln)
            transforms[ln] = (t.getRotation().toNumPy(), t.getPosition().toNumPy())

        ignore_pairs_set = {(a, b) for a, b in ignore_pairs} | {(b, a) for a, b in ignore_pairs}

        colliding: set[str] = set()
        for i, l0 in enumerate(effective_links):
            for l1 in effective_links[i + 1 :]:
                if (l0, l1) in ignore_pairs_set:
                    continue
                # skip neighbors
                if neighbors is not None:
                    if l0 in neighbors and l1 in neighbors[l0].get("links", []):
                        continue
                    if l1 in neighbors and l0 in neighbors[l1].get("links", []):
                        continue
                d = self.check_distance(l0, l1, transforms)
                if d < 0:
                    colliding.add(l0)
                    colliding.add(l1)

        return colliding
