"""Tests for robot-vs-world collision detection (used by the optimizer and visualizer)."""

import os

from idyntree import bindings as iDynTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _checker_and_kindyn(monkeypatch):
    """Build a CollisionChecker + KinDynComputations for the threeLink model at q=0."""
    monkeypatch.chdir(PROJECT_ROOT)
    import yaml

    from identification.collision import CollisionChecker
    from identification.helpers import ParamHelpers, URDFHelpers
    from identification.model import Model

    urdf = "model/threeLinks.urdf"
    config = yaml.safe_load(open("configs/threeLinks.yaml"))
    config["urdf"] = urdf
    config["jointNames"] = iDynTree.StringVector([])
    iDynTree.dofsListFromURDF(urdf, config["jointNames"])
    config["num_dofs"] = len(config["jointNames"])
    config["verbose"] = 0
    model = Model(config, urdf)
    ph = ParamHelpers(model, config)
    uh = URDFHelpers(ph, model, config)
    hulls = {}
    for ln in model.linkNames:
        try:
            hulls[ln] = list(uh.getBoundingBox(input_urdf=urdf, old_com=[0, 0, 0], link_name=ln, scaling=False))
        except Exception:
            pass
    checker = CollisionChecker(uh, urdf, hulls, model.linkNames)

    kin = iDynTree.KinDynComputations()
    kin.loadRobotModel(model.kinDyn.model())
    nd = config["num_dofs"]
    grav = iDynTree.Vector3()
    grav.setVal(2, -9.81)
    q = iDynTree.JointPosDoubleArray(nd)
    dq = iDynTree.JointDOFsDoubleArray(nd)
    kin.setRobotState(iDynTree.Transform.Identity(), q, iDynTree.Twist(), dq, grav)
    return checker, kin, model.linkNames


def test_world_collision_detection(monkeypatch):
    """A world box overlapping the robot is flagged; a far one is not; the margin widens
    the detection radius."""
    checker, kin, link_names = _checker_and_kindyn(monkeypatch)
    ignore: set[str] = set()

    # large box at the origin overlaps the robot -> flagged
    overlap = {"obstacle": ([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])}
    hit = checker.find_world_colliding_links(kin, link_names, overlap, ignore, margin=0.0)
    assert "obstacle" in hit and len(hit) > 1  # the world box and at least one robot link

    # far box (100 m away) is not flagged
    far = {"obstacle": ([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0])}
    assert checker.find_world_colliding_links(kin, link_names, far, ignore, margin=0.0) == set()

    # but a margin large enough to reach it does flag it (margin extends the detection radius)
    assert checker.find_world_colliding_links(kin, link_names, far, ignore, margin=200.0)

    # an ignored world link is never flagged
    assert checker.find_world_colliding_links(kin, link_names, overlap, {"obstacle"}, margin=0.0) == set()


def test_world_collision_depends_on_base_pose(monkeypatch):
    """Robot-vs-world collisions depend on the floating-base pose: the same joint
    configuration collides with a fixed world box only when the base is moved to it.

    This is the property the suspended-base collision fix restores — checks must use the
    simulated (swung) base pose, not the fixed mount pose.
    """
    checker, kin, link_names = _checker_and_kindyn(monkeypatch)  # base at identity, q=0
    ignore: set[str] = set()
    box = {"obstacle": ([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0])}

    # base at origin: the robot is ~3 m from the box -> no collision
    assert checker.find_world_colliding_links(kin, link_names, box, ignore, margin=0.0) == set()

    # move the floating base to the box: same joints now collide
    ndof = kin.getNrOfDegreesOfFreedom()
    q = iDynTree.JointPosDoubleArray(ndof)
    dq = iDynTree.JointDOFsDoubleArray(ndof)
    grav = iDynTree.Vector3()
    grav.setVal(2, -9.81)
    moved = iDynTree.Transform(iDynTree.Rotation.Identity(), iDynTree.Position(3.0, 0.0, 0.0))
    kin.setRobotState(moved, q, iDynTree.Twist(), dq, grav)
    assert checker.find_world_colliding_links(kin, link_names, box, ignore, margin=0.0)
