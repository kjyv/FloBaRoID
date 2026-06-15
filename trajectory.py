#!/usr/bin/env python

import argparse
import sys
from typing import Any

import numpy as np
import yaml
from idyntree import bindings as iDynTree

from excitation.postureOptimizer import PostureOptimizer
from excitation.simulationEffects import add_sudden_stops
from excitation.trajectoryGenerator import (
    FixedPositionTrajectory,
    PulsedTrajectory,
    computeTrajectoryDynamics,
    minimum_jerk_transition,
)
from excitation.trajectoryOptimizer import TrajectoryOptimizer
from identification.model import Model
from identifier import Identification

parser = argparse.ArgumentParser(description="Generate excitation trajectories, save to <filename>.")
parser.add_argument(
    "--filename",
    type=str,
    help="the filename to save the trajectory to, otherwise <model>.trajectory.npz",
)
parser.add_argument("--config", required=True, type=str, help="use options from given config file")
parser.add_argument("--model", required=True, type=str, help="the file to load the robot model from")
parser.add_argument(
    "--model_real",
    required=False,
    type=str,
    help='the file to load the "real" robot model from',
)
parser.add_argument("--world", required=False, type=str, help="the file to load world links from")
args = parser.parse_args()

with open(args.config) as stream:
    try:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)

config["urdf"] = args.model
config["urdf_real"] = args.model_real
if config["useStaticTrajectories"] and not config["urdf_real"]:
    print("When optimizing static postures, need model_real argument!")
    sys.exit()
config["jointNames"] = iDynTree.StringVector([])
if not iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"]):
    sys.exit()
config["num_dofs"] = len(config["jointNames"])
config["skipSamples"] = 0


def main():
    # save either optimized or random trajectory parameters to filename
    if args.filename:
        traj_file = args.filename
    else:
        traj_file = config["urdf"] + ".trajectory.npz"

    if config["optimizeTrajectory"]:
        # find trajectory params by optimization
        old_sim = config["simulateTorques"]
        config["simulateTorques"] = True
        model = Model(config, config["urdf"])
        if config["useStaticTrajectories"]:
            old_gravity = config["identifyGravityParamsOnly"]
            idf = Identification(
                config,
                config["urdf"],
                config["urdf_real"],
                measurements_files=None,
                regressor_file=None,
                validation_file=None,
            )
            trajectoryOptimizer: PostureOptimizer | TrajectoryOptimizer = PostureOptimizer(
                config, idf, model, simulation_func=computeTrajectoryDynamics, world=args.world
            )
            config["identifyGravityParamsOnly"] = old_gravity
        else:
            idf = Identification(
                config,
                config["urdf"],
                urdf_file_real=None,
                measurements_files=None,
                regressor_file=None,
                validation_file=None,
            )
            trajectoryOptimizer = TrajectoryOptimizer(
                config, idf, model, simulation_func=computeTrajectoryDynamics, world=args.world
            )

        trajectory = trajectoryOptimizer.optimizeTrajectory()
        config["simulateTorques"] = old_sim
    else:
        # use some random params
        print("no optimized trajectory found, generating random one")
        trajectory = PulsedTrajectory(config["num_dofs"], use_deg=config["useDeg"]).initWithRandomParams()
        print(f"a {[t_a.tolist() for t_a in trajectory.a]}")
        print(f"b {[t_b.tolist() for t_b in trajectory.b]}")
        print(f"q {trajectory.q.tolist()}")
        print(f"nf {trajectory.nf.tolist()}")
        print(f"wf {trajectory.w_f_global}")

    # sample the optimized trajectory into position/velocity/acceleration arrays
    # reuse the model from optimization if available, otherwise create one
    if "model" not in dir() or model is None:
        model = Model(config, config["urdf"])
    trajectory_data, _ = computeTrajectoryDynamics(config, trajectory, model=model)
    freq = config["excitationFrequency"]
    num_dofs = config["num_dofs"]

    times = trajectory_data["target_positions"][:, 0:0].squeeze()  # dummy, replaced below
    positions = trajectory_data["target_positions"]
    velocities = trajectory_data["target_velocities"]
    accelerations = trajectory_data["target_accelerations"]
    times = trajectory_data["times"]

    # insert sudden stops/restarts into the periodic part BEFORE adding transitions
    # (stops reintegrate positions, which breaks periodicity — must happen before
    # transitions so the ramp-in/out paths aren't corrupted by the drift)
    num_stops = config.get("simulateNumStops", 0)
    if num_stops > 0:
        seed = config.get("simulateRandomSeed", 42)
        rng = np.random.default_rng(seed)
        positions, velocities, accelerations = add_sudden_stops(
            times, positions, velocities, accelerations, freq, num_stops=num_stops, rng=rng
        )

    # add smooth transition segments (ramp-in from zero, ramp-out to zero)
    # Sudden stops are applied before this, so the periodic trajectory may have
    # small position drift. The Fourier series itself is periodic (pos[0] ≈ pos[-1]),
    # so both transitions follow nearly the same path.
    transition_duration = config.get("transitionDuration", 3.0)
    if transition_duration > 0:
        q_start = positions[0]
        q_end = positions[-1]
        zero_pos = np.zeros(num_dofs)

        ri_t, ri_pos, ri_vel, ri_acc = minimum_jerk_transition(zero_pos, q_start, transition_duration, freq)
        ro_t, ro_pos, ro_vel, ro_acc = minimum_jerk_transition(q_end, zero_pos, transition_duration, freq)
        ro_t += times[-1] + 1.0 / freq

        times = np.concatenate([ri_t, times + ri_t[-1] + 1.0 / freq, ro_t])
        positions = np.concatenate([ri_pos, positions, ro_pos])
        velocities = np.concatenate([ri_vel, velocities, ro_vel])
        accelerations = np.concatenate([ri_acc, accelerations, ro_acc])

    # append static postures with smooth transitions for gravity term excitation
    if config.get("staticPostures") and not config.get("floatingBase", 0):
        valid_postures = [p[:num_dofs] for p in config["staticPostures"] if len(p) >= num_dofs]
        if valid_postures:
            samples_per = config.get("simulateStaticSamplesPerPosture", 100)
            t_offset = times[-1] + 1.0 / freq
            current_pos = positions[-1]

            segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for posture in valid_postures:
                target = np.array(posture[:num_dofs], dtype=float)
                if transition_duration > 0 and not np.allclose(current_pos, target):
                    tr_t, tr_pos, tr_vel, tr_acc = minimum_jerk_transition(
                        current_pos, target, transition_duration, freq
                    )
                    segments.append((tr_t + t_offset, tr_pos, tr_vel, tr_acc))
                    t_offset += tr_t[-1] + 1.0 / freq

                hold_times = np.arange(samples_per) / freq + t_offset
                segments.append(
                    (
                        hold_times,
                        np.tile(target, (samples_per, 1)),
                        np.zeros((samples_per, num_dofs)),
                        np.zeros((samples_per, num_dofs)),
                    )
                )
                t_offset = hold_times[-1] + 1.0 / freq
                current_pos = target

            times = np.concatenate([times] + [s[0] for s in segments])
            positions = np.concatenate([positions] + [s[1] for s in segments])
            velocities = np.concatenate([velocities] + [s[2] for s in segments])
            accelerations = np.concatenate([accelerations] + [s[3] for s in segments])

    print(f"Saving trajectory to {traj_file}")

    if config["useStaticTrajectories"]:
        if not isinstance(trajectory, FixedPositionTrajectory) or trajectory.angles is None:
            raise RuntimeError("Expected initialized FixedPositionTrajectory for static trajectories")
        save_dict: dict[str, Any] = {
            "static": True,
            "angles": np.array(trajectory.angles, dtype=object),
        }
    else:
        if not isinstance(trajectory, PulsedTrajectory):
            raise RuntimeError("Expected PulsedTrajectory for non-static trajectories")
        a_arr = np.array(trajectory.a, dtype=object)
        b_arr = np.array(trajectory.b, dtype=object)
        save_dict = {
            "use_deg": trajectory.use_deg,
            "static": False,
            "a": a_arr,
            "b": b_arr,
            "q": trajectory.q,
            "nf": trajectory.nf,
            "wf": trajectory.w_f_global,
        }
        if trajectory.joint_limits is not None:
            save_dict["joint_limits"] = np.array(trajectory.joint_limits)

    # save sampled kinematics (torques are computed by the simulator)
    save_dict["positions"] = positions
    save_dict["velocities"] = velocities
    save_dict["accelerations"] = accelerations
    save_dict["times"] = times
    save_dict["frequency"] = np.float64(freq)

    # observability analysis: identify which parameters can't be excited by this trajectory
    if hasattr(model, "YBase") and model.YBase is not None and model.YBase.size > 0:
        U_obs, S_obs, Vt_obs = np.linalg.svd(model.YBase, full_matrices=False)
        obs_thresh = config.get("observabilityThreshold", 1e-6)
        n_unobs = int(np.sum(S_obs < S_obs[0] * obs_thresh))
        n_base = model.YBase.shape[1]
        n_obs = n_base - n_unobs

        if n_unobs > 0:
            # map unobservable base-parameter directions to identified (standard) parameters
            V_unobs = Vt_obs[-n_unobs:, :]  # unobservable right singular vectors
            unobs_energy = np.sum(V_unobs**2, axis=0)  # energy per base param in unobservable subspace
            proj = model.Pb if not model.opt.get("useBasisProjection", False) else model.B
            unobs_per_param = proj @ unobs_energy  # map to identified param space

            # parameters with >50% energy in the unobservable subspace
            unobs_param_indices = np.where(unobs_per_param > 0.5)[0].tolist()

            param_names = ["m", "hx", "hy", "hz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]
            print(f"\nObservability: {n_obs}/{n_base} base parameters observable (threshold={obs_thresh})")
            print(f"Unobservable identified parameters ({len(unobs_param_indices)}):")
            for idx in unobs_param_indices:
                if idx < model.num_model_params:
                    link_idx = idx // 10
                    ptype = idx % 10
                    name = f"{param_names[ptype]} of {model.linkNames[link_idx]}"
                else:
                    jidx = idx - model.num_model_params
                    if jidx < num_dofs:
                        name = f"Fc_{model.jointNames[jidx]}"
                    else:
                        name = f"Fv_{model.jointNames[jidx - num_dofs]}"
                print(f"  [{idx}] {name} (score={unobs_per_param[idx]:.2f})")
        else:
            unobs_param_indices = []
            print(f"\nObservability: all {n_base} base parameters are observable")

        save_dict["unobservable_params"] = np.array(unobs_param_indices, dtype=np.int64)
        save_dict["observability_threshold"] = obs_thresh
        save_dict["n_observable_base_params"] = n_obs

    np.savez(traj_file, **save_dict)


if __name__ == "__main__":
    main()
