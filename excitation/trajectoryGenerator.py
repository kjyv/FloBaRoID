from __future__ import annotations

from typing import Any

import numpy as np

from identification.data import Data
from identification.model import Model


def simulateTrajectory(
    config: dict, trajectory: Trajectory, model: Model | None = None, measurements: dict | None = None
) -> tuple[dict, Data]:
    # generate data arrays for simulation and regressor building
    old_sim = config["simulateTorques"]
    config["simulateTorques"] = True

    if config["floatingBase"]:
        fb = 6
    else:
        fb = 0

    if not model:
        if "urdf_real" in config and config["urdf_real"]:
            print('Simulating using "real" model parameters.')
            urdf = config["urdf_real"]
        else:
            urdf = config["urdf"]

        model = Model(config, urdf)

    data = Data(config)
    trajectory_data: dict[str, Any] = {}
    trajectory_data["target_positions"] = []
    trajectory_data["target_velocities"] = []
    trajectory_data["target_accelerations"] = []
    trajectory_data["torques"] = []
    trajectory_data["times"] = []

    freq = config["excitationFrequency"]
    num_dofs = config["num_dofs"]
    num_samples = int(trajectory.getPeriodLength() * freq)
    use_deg = config["useDeg"]

    # vectorized trajectory computation: compute all samples × dofs at once
    # using numpy broadcasting instead of per-sample per-dof Python loops
    if hasattr(trajectory, "oscillators") and trajectory.oscillators:
        times = np.arange(num_samples) / freq
        positions = np.empty((num_samples, num_dofs))
        velocities = np.empty((num_samples, num_dofs))
        accelerations = np.empty((num_samples, num_dofs))

        for d in range(num_dofs):
            osc = trajectory.oscillators[d]
            # harmonic indices [1, 2, ..., nf] and precompute wf*l*t for all times and harmonics
            l_arr = np.arange(1, osc.nf + 1)  # (nf,)
            wlt = osc.w_f * np.outer(times, l_arr)  # (num_samples, nf)
            sin_wlt = np.sin(wlt)
            cos_wlt = np.cos(wlt)

            a_arr = np.array(osc.a)
            b_arr = np.array(osc.b)

            if isinstance(osc, BoundedOscillationGenerator):
                # bounded mode: raw signal → tanh mapping → position within joint limits
                raw = cos_wlt @ b_arr + sin_wlt @ a_arr
                th = np.tanh(raw)
                sech2 = 1.0 - th**2

                positions[:, d] = osc.q_center + osc.q_range * th

                # raw' and raw'' for chain rule
                wl = osc.w_f * l_arr
                raw_dot = cos_wlt @ (a_arr * wl) - sin_wlt @ (b_arr * wl)
                raw_ddot = -sin_wlt @ (a_arr * wl**2) - cos_wlt @ (b_arr * wl**2)

                velocities[:, d] = osc.q_range * sech2 * raw_dot
                accelerations[:, d] = osc.q_range * (sech2 * raw_ddot - 2.0 * th * sech2 * raw_dot**2)
            else:
                # classic Swevers (1997) mode: integrate velocity harmonics
                a_coeff = a_arr / (osc.w_f * l_arr)
                b_coeff = b_arr / (osc.w_f * l_arr)

                positions[:, d] = sin_wlt @ a_coeff - cos_wlt @ b_coeff + osc.nf * osc.q0
                velocities[:, d] = cos_wlt @ a_arr + sin_wlt @ b_arr

                wl = osc.w_f * l_arr
                accelerations[:, d] = -sin_wlt @ (a_arr * wl) + cos_wlt @ (b_arr * wl)

        if use_deg:
            positions = np.deg2rad(positions)
            velocities = np.deg2rad(velocities)
            accelerations = np.deg2rad(accelerations)
    else:
        # fallback for non-pulsed trajectories (FixedPositionTrajectory etc.)
        positions = np.empty((num_samples, num_dofs))
        velocities = np.empty((num_samples, num_dofs))
        accelerations = np.empty((num_samples, num_dofs))
        times = np.empty(num_samples)

        for t in range(num_samples):
            t_sec = t / freq
            trajectory.setTime(t_sec)
            times[t] = t_sec
            for d in range(num_dofs):
                positions[t, d] = trajectory.getAngle(d)
                velocities[t, d] = trajectory.getVelocity(d)
                accelerations[t, d] = trajectory.getAcceleration(d)

        if use_deg:
            positions = np.deg2rad(positions)
            velocities = np.deg2rad(velocities)
            accelerations = np.deg2rad(accelerations)

    trajectory_data["target_positions"] = positions
    trajectory_data["positions"] = positions
    trajectory_data["target_velocities"] = velocities
    trajectory_data["velocities"] = velocities
    trajectory_data["target_accelerations"] = accelerations
    trajectory_data["accelerations"] = accelerations
    trajectory_data["torques"] = np.zeros((num_samples, num_dofs + fb))
    trajectory_data["times"] = times
    trajectory_data["measured_frequency"] = freq
    trajectory_data["base_velocity"] = np.zeros((num_samples, 6))
    trajectory_data["base_acceleration"] = np.zeros((num_samples, 6))

    trajectory_data["base_rpy"] = np.zeros((num_samples, 3))

    # for floating-base robots, the base is assumed stationary (fixed contact with
    # the environment). The base wrench from inverse dynamics gives the reaction
    # forces at the base link, which is what a force/torque sensor at the contact
    # point would measure. No separate contact wrench needs to be specified.
    trajectory_data["contacts"] = np.array({})

    if measurements:
        trajectory_data["positions"] = measurements["Q"]
        trajectory_data["velocities"] = measurements["V"]
        trajectory_data["accelerations"] = measurements["Vdot"]
        trajectory_data["measured_frequency"] = measurements["measured_frequency"]

    old_skip = config["skipSamples"]
    config["skipSamples"] = 0
    old_offset = config["startOffset"]
    config["startOffset"] = 0
    data.init_from_data(trajectory_data)
    model.computeRegressors(data)
    trajectory_data["torques"][:, :] = data.samples["torques"][:, :]

    """
    if config['floatingBase']:
        # add force of contact to keep robot fixed in space (always accelerate exactly against gravity)
        # floating base orientation has to be rotated so that accelerations resulting from hanging
        # are zero, i.e. the vector COM - contact point is parallel to gravity.
        if contacts:
            # get jacobian of contact frame at current posture
            dim = model.num_dofs+fb
            jacobian = iDynTree.MatrixDynSize(6, dim)
            model.dynComp.getFrameJacobian(contactFrame, jacobian)
            jacobian = jacobian.toNumPy()

            # get base link vel and acc and torques that result from contact force / acceleration
            contacts_torq = np.zeros(dim)
            contacts_torq = jacobian.T.dot(contact_wrench)
            trajectory_data['base_acceleration'] += contacts_torq[0:6]  # / 139.122
            data.samples['base_acceleration'][:,:] = trajectory_data['base_acceleration'][:,:]
            # simulate again with proper base acceleration
            model.computeRegressors(data, only_simulate=True)
            trajectory_data['torques'][:,:] = data.samples['torques'][:,:]
    """

    config["skipSamples"] = old_skip
    config["startOffset"] = old_offset
    config["simulateTorques"] = old_sim

    return trajectory_data, data


class Trajectory:
    """base trajectory class"""

    def getAngle(self, dof):
        raise NotImplementedError()

    def getVelocity(self, dof):
        raise NotImplementedError()

    def getAcceleration(self, dof):
        raise NotImplementedError()

    def getPeriodLength(self):
        raise NotImplementedError()

    def setTime(self, time):
        raise NotImplementedError()

    def wait_for_zero_vel(self, t_elapsed):
        raise NotImplementedError()


class PulsedTrajectory(Trajectory):
    """pulsating trajectory generator for one joint using fourier series from
    Swevers, Gansemann (1997). Gives values for one time instant (at the current
    internal time value)
    """

    def __init__(self, dofs: int, use_deg: bool = False) -> None:
        self.dofs = dofs
        self.oscillators: list[OscillationGenerator | BoundedOscillationGenerator] = list()
        self.use_deg = use_deg
        self.joint_limits: list[tuple[float, float]] | None = None
        self.w_f_global = 1.0

    def initWithRandomParams(self):
        # init with random params
        # TODO: use specified bounds
        a: list[Any] = [0] * self.dofs
        b: list[Any] = [0] * self.dofs
        nf: Any = np.random.randint(1, 4, self.dofs)
        q: Any = np.random.rand(self.dofs) * 2 - 1
        for i in range(0, self.dofs):
            maximum = 2.0 - np.abs(q[i])
            a[i] = np.random.rand(nf[i]) * maximum - maximum / 2
            b[i] = np.random.rand(nf[i]) * maximum - maximum / 2

        # random values are in rad, so convert
        if self.use_deg:
            q = np.rad2deg(q)
        # print a
        # print b
        # print q

        self.a = a
        self.b = b
        self.q = q
        self.nf = nf

        self.oscillators = list()
        for i in range(0, self.dofs):
            self.oscillators.append(
                OscillationGenerator(
                    w_f=self.w_f_global,
                    a=np.array(a[i]),
                    b=np.array(b[i]),
                    q0=q[i],
                    nf=nf[i],
                    use_deg=self.use_deg,
                )
            )
        return self

    def initWithParams(
        self,
        a: Any,
        b: Any,
        q: Any,
        nf: Any,
        wf: Any = None,
        joint_limits: list[tuple[float, float]] | None = None,
    ) -> PulsedTrajectory:
        """Init with given params.

        a - list of dof coefficients a
        b - list of dof coefficients b
        q - list of dof coefficients q_0
        nf - list of dof coefficients n_f
        joint_limits - optional list of (lower, upper) per joint in rad for bounded mode
        (also see docstring of OscillationGenerator / BoundedOscillationGenerator)
        """

        if len(nf) != self.dofs or len(q) != self.dofs:
            raise Exception("Need DOFs many values for nf and q!")

        self.a = a
        self.b = b
        self.q = q
        self.nf = nf
        self.joint_limits = joint_limits
        if wf:
            self.w_f_global = wf

        self.oscillators = list()
        for i in range(0, self.dofs):
            if joint_limits is not None:
                self.oscillators.append(
                    BoundedOscillationGenerator(
                        w_f=self.w_f_global,
                        a=np.array(a[i]),
                        b=np.array(b[i]),
                        q0=q[i],
                        nf=nf[i],
                        use_deg=self.use_deg,
                        q_lower=joint_limits[i][0],
                        q_upper=joint_limits[i][1],
                    )
                )
            else:
                self.oscillators.append(
                    OscillationGenerator(
                        w_f=self.w_f_global,
                        a=np.array(a[i]),
                        b=np.array(b[i]),
                        q0=q[i],
                        nf=nf[i],
                        use_deg=self.use_deg,
                    )
                )
        return self

    def getAngle(self, dof):
        """get angle at current time for joint dof"""
        return self.oscillators[dof].getAngle(self.time)

    def getVelocity(self, dof):
        """get velocity at current time for joint dof"""
        return self.oscillators[dof].getVelocity(self.time)

    def getAcceleration(self, dof):
        """get acceleration at current time for joint dof"""
        return self.oscillators[dof].getAcceleration(self.time)

    def getPeriodLength(self):
        """get the period length of the oscillation in seconds"""
        return 2 * np.pi / self.w_f_global

    def setTime(self, time):
        """set current time in seconds"""
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        self.setTime(t_elapsed)
        if self.use_deg:
            thresh = 5.0
        else:
            thresh = np.deg2rad(5.0)
        return abs(self.getVelocity(0)) < thresh


class OscillationGenerator:
    def __init__(self, w_f, a, b, q0, nf, use_deg):
        """
        generate periodic oscillation from fourier series (Swevers, 1997)

        - w_f is the global pulsation (frequency is w_f / 2pi)
        - a and b are (arrays of) amplitudes of the sine/cosine
          functions for each joint
        - q0 is the joint angle offset (center of pulsation)
        - nf is the desired amount of coefficients for this fourier series
        """
        self.w_f = float(w_f)
        self.a = a
        self.b = b
        self.use_deg = use_deg
        self.q0 = float(q0)
        if use_deg:
            self.q0 = np.deg2rad(self.q0)
        self.nf = nf

    def getAngle(self, t):
        # - t is the current time
        q = 0.0
        for l in range(1, self.nf + 1):
            q += (self.a[l - 1] / (self.w_f * l)) * np.sin(self.w_f * l * t) - (
                self.b[l - 1] / (self.w_f * l)
            ) * np.cos(self.w_f * l * t)
        q += self.nf * self.q0
        if self.use_deg:
            q = np.rad2deg(q)
        return q

    def getVelocity(self, t):
        dq = 0.0
        for l in range(1, self.nf + 1):
            dq += self.a[l - 1] * np.cos(self.w_f * l * t) + self.b[l - 1] * np.sin(self.w_f * l * t)
        if self.use_deg:
            dq = np.rad2deg(dq)
        return dq

    def getAcceleration(self, t):
        ddq = 0.0
        for l in range(1, self.nf + 1):
            ddq += -self.a[l - 1] * self.w_f * l * np.sin(self.w_f * l * t) + self.b[l - 1] * self.w_f * l * np.cos(
                self.w_f * l * t
            )
        if self.use_deg:
            ddq = np.rad2deg(ddq)
        return ddq


class BoundedOscillationGenerator:
    """Generate joint-limit-bounded periodic oscillation using tanh mapping.

    Instead of integrating Fourier velocity coefficients (which can produce
    unbounded positions), this generator maps an unbounded Fourier signal
    through tanh to guarantee positions stay within [q_lower, q_upper].

    The internal signal is: raw(t) = sum(a[l]*sin(wf*l*t) + b[l]*cos(wf*l*t))
    The output position is: q(t) = q_center + q_range * tanh(raw(t))

    Velocity and acceleration are computed analytically via chain rule.
    This eliminates position limit constraints from the optimizer entirely.
    """

    def __init__(
        self,
        w_f: float,
        a: np.ndarray,
        b: np.ndarray,
        q0: float,
        nf: int,
        use_deg: bool,
        q_lower: float,
        q_upper: float,
    ) -> None:
        self.w_f = float(w_f)
        self.a = a
        self.b = b
        self.nf = nf
        self.use_deg = use_deg

        # q0 is the center offset, convert to rad if needed
        self.q0 = float(q0)
        if use_deg:
            self.q0 = np.deg2rad(self.q0)

        # joint limits (always in rad)
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.q_center = 0.5 * (q_lower + q_upper) + self.q0
        # range from center to limit (use the smaller side for safety)
        half_range = 0.5 * (q_upper - q_lower)
        # leave a small margin so tanh doesn't need to hit exactly ±1
        self.q_range = half_range * 0.95

    def _raw(self, t: float) -> float:
        """Unbounded internal Fourier signal."""
        s = 0.0
        for l in range(1, self.nf + 1):
            s += self.a[l - 1] * np.sin(self.w_f * l * t) + self.b[l - 1] * np.cos(self.w_f * l * t)
        return s

    def _raw_dot(self, t: float) -> float:
        """Time derivative of the internal signal."""
        s = 0.0
        for l in range(1, self.nf + 1):
            wl = self.w_f * l
            s += self.a[l - 1] * wl * np.cos(wl * t) - self.b[l - 1] * wl * np.sin(wl * t)
        return s

    def _raw_ddot(self, t: float) -> float:
        """Second time derivative of the internal signal."""
        s = 0.0
        for l in range(1, self.nf + 1):
            wl = self.w_f * l
            s += -self.a[l - 1] * wl**2 * np.sin(wl * t) - self.b[l - 1] * wl**2 * np.cos(wl * t)
        return s

    def getAngle(self, t: float) -> float:
        """Position: q_center + q_range * tanh(raw(t))."""
        q = self.q_center + self.q_range * np.tanh(self._raw(t))
        if self.use_deg:
            q = np.rad2deg(q)
        return q

    def getVelocity(self, t: float) -> float:
        """Velocity via chain rule: q_range * sech²(raw) * raw'."""
        raw = self._raw(t)
        sech2 = 1.0 - np.tanh(raw) ** 2
        dq = self.q_range * sech2 * self._raw_dot(t)
        if self.use_deg:
            dq = np.rad2deg(dq)
        return dq

    def getAcceleration(self, t: float) -> float:
        """Acceleration via chain rule: q_range * (sech² * raw'' - 2*tanh*sech² * raw'²)."""
        raw = self._raw(t)
        th = np.tanh(raw)
        sech2 = 1.0 - th**2
        rd = self._raw_dot(t)
        rdd = self._raw_ddot(t)
        ddq = self.q_range * (sech2 * rdd - 2.0 * th * sech2 * rd**2)
        if self.use_deg:
            ddq = np.rad2deg(ddq)
        return ddq


class FixedPositionTrajectory(Trajectory):
    """generate static 'trajectories'"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.time = 0.0
        self.use_deg = self.config["useDeg"]
        self.angles: list[dict[str, Any]] | None = None

    def initWithAngles(self, angles: list[dict[str, Any]]) -> None:
        """angles is a list containing for each posture a dict {
            start_time: float    # starting time in seconds of posture
            angles: List[float]  # angles for each joint
        }
        """
        self.angles = angles
        self.posLength = angles[1]["start_time"] - angles[0]["start_time"]

    def getAngle(self, dof: int) -> float:
        """get angle at current time for joint dof"""

        if self.angles is not None:
            for angle_set in self.angles:
                if angle_set["start_time"] >= self.time - self.posLength:
                    return angle_set["angles"][dof]

            # if no angle found (shouldn't happen)
            print(f"Warning: no angle found for time {self.time}")
            return 0.0
        else:
            # Walk-Man:
            # ['LHipLat', 'LHipYaw', 'LHipSag', 'LKneeSag', 'LAnkSag', 'LAnkLat',
            #  'RHipLat', 'RHipYaw', 'RHipSag', 'RKneeSag', 'RAnkSag', 'RAnkLat',
            #  'WaistSag', 'WaistYaw', #WaistLat is fixed atm
            #  'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2',
            #  'RShSag', 'RShLat', 'RShYaw', 'RElbj', 'RForearmPlate', 'RWrj1', 'RWrj2']
            """
            # posture #0
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,      #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #1
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    20.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    20.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #2
            return [0.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    85.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    85.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #3
            return [0.0, 0.0, -70.0, 90.0, -79.0, 0.0,       #left leg
                    0.0, 0.0, -70.0, 90.0, -79.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 90.0, 0.0, -90.0, 0.0, -45.0, 0.0,    #left arm
                    0.0, -90.0, 0.0, -90.0, 0.0, -45.0, 0.0,   #right arm
                    ][dof]
            # posture #4
            return [44.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    -44.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    0.0, 0.0,                           #Waist
                    0.0, 45.0, 0.0, -90.0, 0.0, 0.0, 79.0,    #left arm
                    0.0, -45.0, 0.0, -90.0, 0.0, 0.0, -79.0,   #right arm
                    ][dof]
            # posture #5
            return [44.0, 0.0, -70.0, 90.0, -20.0, 0.0,       #left leg
                    -44.0, 0.0, -70.0, 90.0, -20.0, 0.0,         #right leg
                    35.0, 0.0,                           #Waist
                    20.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    20.0, -45.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            # posture #6
            return [35.0, 0.0, 0.0, 130.0, -20.0, 0.0,       #left leg
                    -35.0, 0.0, 0.0, 130.0, -20.0, 0.0,         #right leg
                    -20.0, -45.0,                           #Waist
                    20.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #left arm
                    85.0, -45.0, 0.0, 0.0, 0.0, 0.0, 0.0,   #right arm
                    ][dof]
            """
            # posture #7
            return [
                20.0,
                0.0,
                -85.0,
                0.0,
                0.0,
                0.0,  # left leg
                -20.0,
                0.0,
                -85.0,
                0.0,
                0.0,
                0.0,  # right leg
                0.0,
                0.0,  # Waist
                0.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # left arm
                0.0,
                -0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # right arm
            ][dof]

    def getVelocity(self, dof):
        """get velocity at current time for joint dof"""
        return 0.0

    def getAcceleration(self, dof):
        """get acceleration at current time for joint dof"""
        return 0.0

    def getPeriodLength(self):
        """get the length of the trajectory in seconds"""
        if self.angles is None:
            raise RuntimeError("angles not initialized; call initWithAngles() first")
        return self.angles[1]["start_time"] * len(self.angles)

    def setTime(self, time):
        """set current time in seconds"""
        self.time = time

    def wait_for_zero_vel(self, t_elapsed):
        return True
