"""Tests for capsule-based collision detection and analytical gradients."""

from __future__ import annotations

import numpy as np

from excitation.capsule import (
    Capsule,
    _capsule_from_box,
    _capsule_from_cylinder,
    _capsule_from_sphere,
    _merge_capsule_primitives,
    _parse_origin,
    capsule_distance,
    segment_segment_distance,
)


class TestParseOrigin:
    """Tests for URDF origin parsing."""

    def test_none_element(self) -> None:
        """Returns zero position and identity rotation for None input."""
        pos, rot = _parse_origin(None)
        np.testing.assert_array_equal(pos, np.zeros(3))
        np.testing.assert_array_equal(rot, np.eye(3))

    def test_xyz_only(self) -> None:
        """Parses XYZ when RPY is absent."""
        import xml.etree.ElementTree as ET

        elem = ET.fromstring('<origin xyz="1.0 2.0 3.0"/>')
        pos, rot = _parse_origin(elem)
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(rot, np.eye(3))

    def test_xyz_and_rpy(self) -> None:
        """Parses both XYZ and RPY correctly."""
        import xml.etree.ElementTree as ET

        elem = ET.fromstring('<origin xyz="0 0 0.1" rpy="0 0 1.5707963"/>')
        pos, rot = _parse_origin(elem)
        np.testing.assert_array_almost_equal(pos, [0, 0, 0.1])
        # 90° yaw: x→y, y→-x
        np.testing.assert_array_almost_equal(rot @ np.array([1, 0, 0]), [0, 1, 0], decimal=5)


class TestCapsuleFromPrimitives:
    """Tests for converting URDF primitives to capsules."""

    def test_cylinder(self) -> None:
        """Cylinder along Z produces endpoints above and below origin."""
        p0, p1, r = _capsule_from_cylinder(np.zeros(3), np.eye(3), length=0.4, radius=0.05)
        np.testing.assert_array_almost_equal(p0, [0, 0, -0.2])
        np.testing.assert_array_almost_equal(p1, [0, 0, 0.2])
        assert r == 0.05

    def test_cylinder_with_offset(self) -> None:
        """Cylinder with origin offset translates endpoints."""
        origin = np.array([0.0, 0.03, 0.1])
        p0, p1, r = _capsule_from_cylinder(origin, np.eye(3), length=0.208, radius=0.07)
        np.testing.assert_array_almost_equal(p0, [0.0, 0.03, 0.1 - 0.104])
        np.testing.assert_array_almost_equal(p1, [0.0, 0.03, 0.1 + 0.104])

    def test_sphere(self) -> None:
        """Sphere produces degenerate capsule with both endpoints at center."""
        p0, p1, r = _capsule_from_sphere(np.array([1.0, 2.0, 3.0]), radius=0.1)
        np.testing.assert_array_equal(p0, p1)
        np.testing.assert_array_equal(p0, [1.0, 2.0, 3.0])
        assert r == 0.1

    def test_box(self) -> None:
        """Box produces capsule along longest axis, pulled inward by radius."""
        size = np.array([0.1, 0.2, 0.5])
        p0, p1, r = _capsule_from_box(np.zeros(3), np.eye(3), size)
        # radius from the larger of the two shorter half-extents: max(0.05, 0.1)
        expected_r = 0.1
        assert abs(r - expected_r) < 1e-10
        # longest axis is Z (0.5), endpoints pulled inward by radius
        # half_len=0.25, inward=0.1, so endpoints at +/-(0.25-0.1) = +/-0.15
        np.testing.assert_array_almost_equal(p0, [0, 0, -0.15])
        np.testing.assert_array_almost_equal(p1, [0, 0, 0.15])
        # total capsule extent: seg + 2*r = 0.3 + 0.2 = 0.5 = box longest dim
        seg = float(np.linalg.norm(p1 - p0))
        assert abs(seg + 2 * r - size[2]) < 1e-10


class TestMergeCapsulePrimitives:
    """Tests for merging multiple collision primitives into one capsule."""

    def test_single_primitive(self) -> None:
        """Single primitive is returned unchanged."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.5])
        r = 0.05
        result = _merge_capsule_primitives([(p0, p1, r)])
        np.testing.assert_array_equal(result[0], p0)
        np.testing.assert_array_equal(result[1], p1)
        assert result[2] == r

    def test_kuka_link_pattern(self) -> None:
        """Merge two spheres + cylinder (typical KUKA link pattern).

        Endpoints are pulled inward by the radius so the capsule surface
        aligns with the original extreme points without protruding.
        """
        # sphere at origin
        s1 = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.07)
        # sphere offset
        s2 = (np.array([0.0, 0.06, 0.2]), np.array([0.0, 0.06, 0.2]), 0.07)
        # cylinder between them
        c = (np.array([0.0, 0.03, -0.004]), np.array([0.0, 0.03, 0.204]), 0.07)
        p0, p1, r = _merge_capsule_primitives([s1, s2, c])
        assert r == 0.07
        # segment should be shorter than original span due to inward pull
        seg_len = float(np.linalg.norm(p1 - p0))
        # original span between most distant points is ~0.21m, minus 2*0.07 = ~0.07
        assert seg_len > 0.01
        # but the capsule surface (segment + radius) should still reach the original extent
        # i.e., the total extent is approximately seg_len + 2*r ≈ original span
        original_span = float(np.linalg.norm(np.array([0.0, 0.03, 0.204]) - np.array([0.0, 0.0, 0.0])))
        assert abs(seg_len + 2 * r - original_span) < 0.01


class TestSegmentSegmentDistance:
    """Tests for the closed-form segment-segment distance computation."""

    def test_parallel_segments(self) -> None:
        """Two parallel segments with known distance."""
        a0 = np.array([0.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        b0 = np.array([0.0, 1.0, 0.0])
        b1 = np.array([1.0, 1.0, 0.0])
        dist, s, t = segment_segment_distance(a0, a1, b0, b1)
        assert abs(dist - 1.0) < 1e-10

    def test_perpendicular_segments(self) -> None:
        """Two perpendicular crossing segments."""
        a0 = np.array([0.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        b0 = np.array([0.5, -1.0, 0.5])
        b1 = np.array([0.5, 1.0, 0.5])
        dist, s, t = segment_segment_distance(a0, a1, b0, b1)
        assert abs(dist - 0.5) < 1e-10
        assert abs(s - 0.5) < 1e-10
        assert abs(t - 0.5) < 1e-10

    def test_intersecting_segments(self) -> None:
        """Two segments that intersect have zero distance."""
        a0 = np.array([0.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        b0 = np.array([0.5, -0.5, 0.0])
        b1 = np.array([0.5, 0.5, 0.0])
        dist, s, t = segment_segment_distance(a0, a1, b0, b1)
        assert abs(dist) < 1e-10

    def test_point_to_point(self) -> None:
        """Degenerate case: both segments are points."""
        a0 = np.array([0.0, 0.0, 0.0])
        b0 = np.array([3.0, 4.0, 0.0])
        dist, s, t = segment_segment_distance(a0, a0, b0, b0)
        assert abs(dist - 5.0) < 1e-10
        assert s == 0.0
        assert t == 0.0

    def test_point_to_segment(self) -> None:
        """Degenerate case: one segment is a point."""
        a0 = np.array([0.5, 1.0, 0.0])
        b0 = np.array([0.0, 0.0, 0.0])
        b1 = np.array([1.0, 0.0, 0.0])
        dist, s, t = segment_segment_distance(a0, a0, b0, b1)
        assert abs(dist - 1.0) < 1e-10
        assert abs(t - 0.5) < 1e-10

    def test_endpoint_closest(self) -> None:
        """Case where the closest points are at segment endpoints."""
        a0 = np.array([0.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        b0 = np.array([2.0, 0.0, 0.0])
        b1 = np.array([3.0, 0.0, 0.0])
        dist, s, t = segment_segment_distance(a0, a1, b0, b1)
        assert abs(dist - 1.0) < 1e-10
        assert abs(s - 1.0) < 1e-10
        assert abs(t - 0.0) < 1e-10

    def test_skew_segments(self) -> None:
        """Non-parallel, non-intersecting segments in 3D."""
        a0 = np.array([0.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        b0 = np.array([0.0, 0.0, 1.0])
        b1 = np.array([0.0, 1.0, 1.0])
        dist, s, t = segment_segment_distance(a0, a1, b0, b1)
        assert abs(dist - 1.0) < 1e-10
        assert abs(s - 0.0) < 1e-10
        assert abs(t - 0.0) < 1e-10


class TestCapsuleDistance:
    """Tests for capsule-to-capsule distance computation."""

    def test_separated_capsules(self) -> None:
        """Two capsules separated along the Y axis."""
        cap_a = Capsule("a", np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.0, 0.1]), 0.05)
        cap_b = Capsule("b", np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.0, 0.1]), 0.05)
        # place cap_b at y=0.5
        dist, s, t, _, _, _, _ = capsule_distance(
            cap_a,
            cap_b,
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array([0.0, 0.5, 0.0]),
        )
        # segment distance = 0.5, capsule distance = 0.5 - 0.05 - 0.05 = 0.4
        assert abs(dist - 0.4) < 1e-10

    def test_touching_capsules(self) -> None:
        """Two capsules just touching (distance = 0)."""
        cap_a = Capsule("a", np.zeros(3), np.zeros(3), 0.1)  # sphere r=0.1
        cap_b = Capsule("b", np.zeros(3), np.zeros(3), 0.1)  # sphere r=0.1
        dist, _, _, _, _, _, _ = capsule_distance(
            cap_a,
            cap_b,
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array([0.2, 0.0, 0.0]),
        )
        assert abs(dist) < 1e-10

    def test_overlapping_capsules(self) -> None:
        """Two overlapping capsules have negative distance."""
        cap_a = Capsule("a", np.zeros(3), np.zeros(3), 0.1)
        cap_b = Capsule("b", np.zeros(3), np.zeros(3), 0.1)
        dist, _, _, _, _, _, _ = capsule_distance(
            cap_a,
            cap_b,
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.array([0.1, 0.0, 0.0]),
        )
        assert dist < 0

    def test_with_rotation(self) -> None:
        """Capsule distance with a rotated link."""
        cap_a = Capsule("a", np.array([0.0, 0.0, -0.2]), np.array([0.0, 0.0, 0.2]), 0.05)
        cap_b = Capsule("b", np.array([0.0, 0.0, -0.2]), np.array([0.0, 0.0, 0.2]), 0.05)
        # rotate cap_b by 90° around X, so its Z axis aligns with Y
        rot_b = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        dist, _, _, _, _, _, _ = capsule_distance(
            cap_a,
            cap_b,
            np.eye(3),
            np.zeros(3),
            rot_b,
            np.array([1.0, 0.0, 0.0]),
        )
        # segments are perpendicular, both centered at x=0 and x=1
        # segment distance = 1.0, capsule distance = 1.0 - 0.1 = 0.9
        assert abs(dist - 0.9) < 1e-10


class TestSymmetry:
    """Tests verifying distance is symmetric and invariant to argument order."""

    def test_segment_distance_symmetric(self) -> None:
        """Segment distance is the same regardless of which segment is A or B."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            pts = rng.standard_normal((4, 3))
            d1, _, _ = segment_segment_distance(pts[0], pts[1], pts[2], pts[3])
            d2, _, _ = segment_segment_distance(pts[2], pts[3], pts[0], pts[1])
            assert abs(d1 - d2) < 1e-10

    def test_capsule_distance_symmetric(self) -> None:
        """Capsule distance is the same regardless of argument order."""
        cap_a = Capsule("a", np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.0, 0.1]), 0.03)
        cap_b = Capsule("b", np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0]), 0.05)
        rot_a, pos_a = np.eye(3), np.zeros(3)
        rot_b, pos_b = np.eye(3), np.array([0.5, 0.3, 0.0])
        d1, _, _, _, _, _, _ = capsule_distance(cap_a, cap_b, rot_a, pos_a, rot_b, pos_b)
        d2, _, _, _, _, _, _ = capsule_distance(cap_b, cap_a, rot_b, pos_b, rot_a, pos_a)
        assert abs(d1 - d2) < 1e-10
