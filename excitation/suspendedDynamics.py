"""Simulate base motion for a robot suspended from a ball joint.

Given joint trajectories from the trajectory optimizer, computes what base angular
motion would result if the robot were hanging from a virtual ball joint at a
configurable attachment frame (default: crane_ft). This produces physically
meaningful base dynamics for floating-base identification testing.

The ball joint constrains translation (fixed attachment point) but allows free
rotation. The rotational Newton-Euler equations are solved at each timestep to
find the angular acceleration, which is integrated to get orientation and velocity.
"""

from __future__ import annotations

import numpy as np
from idyntree import bindings as iDynTree

from excitation.simulationEffects import angular_velocity_to_rpy_rates


def simulate_suspended_base_motion(
    urdf_file: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    times: np.ndarray,
    attachment_frame: str = "crane_ft",
    base_link: str = "Waist",
    damping: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate base motion for a robot suspended from a ball joint.

    Uses attachment_frame as the floating base for dynamics (ball joint constraint:
    free rotation, fixed translation). Solves for the attachment frame's angular
    acceleration from the Newton-Euler equations, integrates to get orientation,
    then derives the actual base link (base_link) motion via forward kinematics.

    Integration uses semi-implicit (symplectic) Euler with viscous damping at the
    ball joint to model friction and ensure numerical stability.

    Args:
        urdf_file: path to the robot URDF
        positions: (N, n_dofs) joint positions over time
        velocities: (N, n_dofs) joint velocities over time
        accelerations: (N, n_dofs) joint accelerations over time
        times: (N,) timestamps
        attachment_frame: URDF link where the virtual chain attaches
        base_link: the floating-base link used by the identification pipeline
        damping: viscous damping coefficient at the ball joint (Nm·s/rad)

    Returns:
        base_rpy: (N, 3) base_link orientation in world frame
        base_position: (N, 3) base_link position in world frame
        base_velocity: (N, 6) base_link spatial velocity [linear(3), angular(3)]
        base_acceleration: (N, 6) base_link spatial acceleration [linear(3), angular(3)]
    """
    num_samples, n_dofs = positions.shape

    # set up iDynTree with the attachment frame as floating base
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_file):
        raise RuntimeError(f"Failed to load URDF: {urdf_file}")
    idyn_model = loader.model()

    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(idyn_model)

    # verify the attachment frame exists in the model before setting it as floating base
    # (iDynTree may silently fail for links without inertial elements)
    link_found = False
    for i in range(idyn_model.getNrOfLinks()):
        if idyn_model.getLinkName(i) == attachment_frame:
            link_found = True
            break
    if not link_found:
        raise RuntimeError(
            f"Attachment frame '{attachment_frame}' not found in iDynTree model loaded from "
            f"{urdf_file}. The link may be missing an <inertial> element in the URDF."
        )

    if not kinDyn.setFloatingBase(attachment_frame):
        raise RuntimeError(f"Failed to set floating base to '{attachment_frame}'.")

    idyn_n_dofs = kinDyn.getNrOfDegreesOfFreedom()
    if idyn_n_dofs != n_dofs:
        raise ValueError(f"URDF has {idyn_n_dofs} DOFs but trajectory has {n_dofs}")

    # pre-allocate iDynTree objects
    s = iDynTree.JointPosDoubleArray(n_dofs)
    ds = iDynTree.JointDOFsDoubleArray(n_dofs)
    ddq_zero = iDynTree.JointDOFsDoubleArray(n_dofs)
    for j in range(n_dofs):
        ddq_zero.setVal(j, 0.0)

    gravity_vec = iDynTree.Vector3()
    gravity_vec.setVal(0, 0.0)
    gravity_vec.setVal(1, 0.0)
    gravity_vec.setVal(2, -9.81)

    M_mat = iDynTree.MatrixDynSize(n_dofs + 6, n_dofs + 6)
    ext_wrenches = iDynTree.LinkWrenches(idyn_model)
    ext_wrenches.zero()
    gen_torques = iDynTree.FreeFloatingGeneralizedTorques(idyn_model)
    base_acc_zero = iDynTree.Vector6()

    # find the static equilibrium RPY at the initial joint configuration
    # (the orientation where gravity torque on the attachment frame is zero,
    # i.e. the COM hangs directly below the attachment point)
    att_rpy = _find_equilibrium_rpy(
        kinDyn,
        n_dofs,
        positions[0],
        gravity_vec,
        s,
        ds,
        ddq_zero,
        ext_wrenches,
        gen_torques,
        base_acc_zero,
    )
    att_omega = np.zeros(3)

    # output: base_link (Waist) state at each timestep
    waist_rpy_series = np.zeros((num_samples, 3))
    waist_pos_series = np.zeros((num_samples, 3))
    waist_vel_series = np.zeros((num_samples, 6))

    dt = float(times[1] - times[0]) if num_samples > 1 else 1.0 / 200.0

    for t in range(num_samples):
        # set joint state
        for j in range(n_dofs):
            s.setVal(j, positions[t, j])
            ds.setVal(j, velocities[t, j])

        # set attachment frame state (ball joint: fixed position, current orientation).
        # use Transform directly (NOT .inverse()) so that att_rpy directly controls
        # the world-frame orientation of the attachment frame.
        rot = iDynTree.Rotation.RPY(att_rpy[0], att_rpy[1], att_rpy[2])
        pos = iDynTree.Position.Zero()
        world_T_attachment = iDynTree.Transform(rot, pos)

        att_twist = iDynTree.Twist()
        att_twist.setVal(0, 0.0)  # no linear velocity
        att_twist.setVal(1, 0.0)
        att_twist.setVal(2, 0.0)
        att_twist.setVal(3, att_omega[0])
        att_twist.setVal(4, att_omega[1])
        att_twist.setVal(5, att_omega[2])

        kinDyn.setRobotState(world_T_attachment, s, att_twist, ds, gravity_vec)

        # get mass matrix M (partitioned as [base(6) | joints(n_dofs)])
        kinDyn.getFreeFloatingMassMatrix(M_mat)
        M = M_mat.toNumPy()
        M_bb_rot = M[3:6, 3:6]  # angular-angular block of base
        M_bj_rot = M[3:6, 6:]  # angular-joint coupling block

        # get bias forces h_b (Coriolis + gravity) by calling inverseDynamics with zero accelerations
        kinDyn.inverseDynamics(base_acc_zero, ddq_zero, ext_wrenches, gen_torques)
        h_b = gen_torques.baseWrench().toNumPy()
        h_b_rot = h_b[3:6]  # angular component

        # solve for attachment angular acceleration with implicit damping:
        # M * alpha = -M_bj * ddq - h_b - damping * omega_new
        # with omega_new = omega + alpha * dt, substitute and solve:
        # (M + damping * dt * I) * alpha = -M_bj * ddq - h_b - damping * omega
        ddq_t = accelerations[t]
        M_eff = M_bb_rot + damping * dt * np.eye(3)
        rhs = -M_bj_rot @ ddq_t - h_b_rot - damping * att_omega
        alpha = np.linalg.solve(M_eff, rhs)

        # record Waist (base_link) world transform before integrating.
        # getWorldTransform returns world_T_base; we store the rotation's RPY
        # in the convention used by model.py (RPY such that
        # Transform(RPY(r,p,y), pos).inverse() = world_T_base when pos=0).
        # Since Transform(R, 0).inverse() has rotation R^{-1}, and
        # world_T_base has rotation R_world_base, we need R such that
        # R^{-1} = R_world_base, i.e. R = R_world_base^{-1}.
        waist_transform = kinDyn.getWorldTransform(base_link)
        waist_rot_inv = waist_transform.getRotation().inverse()
        waist_rpy_series[t] = waist_rot_inv.asRPY().toNumPy()
        # position is stored directly in world frame
        waist_pos = waist_transform.getPosition()
        waist_pos_series[t] = waist_pos.toNumPy()
        # analytically compute Waist velocity from the kinematic chain
        # (much more stable than differentiating RPY)
        waist_twist = kinDyn.getFrameVel(base_link)
        waist_vel_series[t] = waist_twist.toNumPy()

        # semi-implicit Euler: update velocity first (includes implicit damping),
        # then use new velocity for position
        if t < num_samples - 1:
            att_omega = att_omega + alpha * dt
            rpy_dot = angular_velocity_to_rpy_rates(att_rpy, att_omega)
            att_rpy = att_rpy + rpy_dot * dt

            # soft clamp: models physical swing limits and prevents divergence.
            # at the limit, reverse and heavily damp the velocity (elastic bounce).
            max_swing = np.deg2rad(25)
            for ax in range(3):
                if att_rpy[ax] > max_swing:
                    att_rpy[ax] = max_swing
                    if att_omega[ax] > 0:
                        att_omega[ax] *= -0.3
                elif att_rpy[ax] < -max_swing:
                    att_rpy[ax] = -max_swing
                    if att_omega[ax] < 0:
                        att_omega[ax] *= -0.3

    # Base velocity from iDynTree's analytical frame velocity computation.
    # This is exact (computed from the Jacobian internally) and avoids the numerical
    # instability of differentiating RPY.
    base_velocity = waist_vel_series

    # Base acceleration by differentiating the smooth velocity signal.
    # The velocity from getFrameVel is smooth (no RPY discontinuities), so
    # central differences work well here.
    base_acceleration = np.zeros((num_samples, 6))
    for i in range(6):
        # central differences for interior, forward/backward at endpoints
        if num_samples > 2:
            base_acceleration[1:-1, i] = (base_velocity[2:, i] - base_velocity[:-2, i]) / (2 * dt)
            base_acceleration[0, i] = (base_velocity[1, i] - base_velocity[0, i]) / dt
            base_acceleration[-1, i] = (base_velocity[-1, i] - base_velocity[-2, i]) / dt

    print(
        f"  Suspended dynamics: attachment={attachment_frame}, "
        f"base RPY range: [{waist_rpy_series.min():.3f}, {waist_rpy_series.max():.3f}] rad"
    )

    return waist_rpy_series, base_velocity, base_acceleration, waist_pos_series


def _find_equilibrium_rpy(
    kinDyn: iDynTree.KinDynComputations,
    n_dofs: int,
    joint_positions: np.ndarray,
    gravity_vec: iDynTree.Vector3,
    s: iDynTree.JointPosDoubleArray,
    ds: iDynTree.JointDOFsDoubleArray,
    ddq_zero: iDynTree.JointDOFsDoubleArray,
    ext_wrenches: iDynTree.LinkWrenches,
    gen_torques: iDynTree.FreeFloatingGeneralizedTorques,
    base_acc_zero: iDynTree.Vector6,
    max_iterations: int = 200,
    tol: float = 0.01,
) -> np.ndarray:
    """Find the static equilibrium RPY for the attachment frame.

    Uses gradient descent on the gravity torque: at each step, nudge the RPY
    in the direction that reduces the angular bias torque. The step size is
    the torque divided by an approximate rotational stiffness (m*g*L).
    """
    for j in range(n_dofs):
        s.setVal(j, joint_positions[j])
        ds.setVal(j, 0.0)

    att_rpy = np.zeros(3)
    att_twist = iDynTree.Twist()

    # approximate rotational stiffness: total_mass * g * characteristic_length
    # for a 140kg robot with COM ~0.5m below attachment: k ≈ 140*9.81*0.5 ≈ 687
    step_scale = 1.0 / 700.0
    torque_norm = 0.0

    for i in range(max_iterations):
        rot = iDynTree.Rotation.RPY(att_rpy[0], att_rpy[1], att_rpy[2])
        pos = iDynTree.Position.Zero()
        world_T_base = iDynTree.Transform(rot, pos)

        kinDyn.setRobotState(world_T_base, s, att_twist, ds, gravity_vec)
        kinDyn.inverseDynamics(base_acc_zero, ddq_zero, ext_wrenches, gen_torques)
        h_b_rot = gen_torques.baseWrench().toNumPy()[3:6]

        torque_norm = float(np.linalg.norm(h_b_rot))
        if torque_norm < tol:
            break

        # gradient descent: RPY correction proportional to gravity torque.
        # the bias torque is the force needed to maintain stasis; to reach
        # equilibrium (where no force is needed), move opposite to the bias.
        att_rpy -= step_scale * h_b_rot

        # clamp to ±30° to stay near the hanging-down equilibrium
        att_rpy = np.clip(att_rpy, np.deg2rad(-30), np.deg2rad(30))

    print(
        f"  Equilibrium RPY: [{np.rad2deg(att_rpy[0]):.2f}, {np.rad2deg(att_rpy[1]):.2f}, "
        f"{np.rad2deg(att_rpy[2]):.2f}] deg (residual torque: {torque_norm:.4f} Nm, "
        f"{i + 1} iterations)"
    )
    return att_rpy
