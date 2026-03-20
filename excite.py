#!/usr/bin/env python

import argparse
import sys

import numpy as np
import yaml
from idyntree import bindings as iDynTree

from excitation.optimizer import plotter
from excitation.trajectoryGenerator import ArrayTrajectory, FixedPositionTrajectory, computeTrajectoryDynamics

parser = argparse.ArgumentParser(description="Send an excitation trajectory and record measurements to <filename>.")
parser.add_argument("--model", required=True, type=str, help="the file to load the robot model from")
parser.add_argument("--filename", type=str, help="the filename to save the measurements to")
parser.add_argument("--trajectory", type=str, help="the file to load the trajectory from")
parser.add_argument("--config", required=True, type=str, help="use options from given config file")
parser.add_argument("--dryrun", help="don't actually send the trajectory", action="store_true")

parser.add_argument("--plot", help="plot measured data", action="store_true")
parser.add_argument(
    "--plot-targets",
    dest="plot_targets",
    help="plot targets instead of measurements",
    action="store_true",
)
parser.set_defaults(plot=False, plot_targets=False, dryrun=False, filename="measurements.npz")
args = parser.parse_args()

with open(args.config) as stream:
    try:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)

config["args"] = args
config["urdf"] = args.model
config["plot_targets"] = args.plot_targets
config["jointNames"] = iDynTree.StringVector([])
if not iDynTree.dofsListFromURDF(config["urdf"], config["jointNames"]):
    sys.exit()
config["num_dofs"] = len(config["jointNames"])

# append parent dir for relative import
# import os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

traj_data: dict[str, np.ndarray] = {}  # hold some global data vars in here


def main():
    if args.trajectory:
        traj_file = args.trajectory
    else:
        traj_file = config["urdf"] + ".trajectory.npz"

    # load trajectory from file
    try:
        tf = np.load(traj_file, encoding="latin1", allow_pickle=True)
        trajectory: ArrayTrajectory | FixedPositionTrajectory
        if "static" in tf and tf["static"]:
            trajectory = FixedPositionTrajectory(config)
            trajectory.initWithAngles(tf["angles"])
            print(f"using static postures from file {traj_file}")
        elif "positions" in tf:
            trajectory = ArrayTrajectory(tf["times"], tf["positions"], tf["velocities"], tf["accelerations"])
            print(f"using trajectory from file {traj_file} ({len(tf['times'])} samples)")
        else:
            print(f"Error: {traj_file} has no saved positions. Regenerate with trajectory.py.")
            sys.exit(1)
    except OSError:
        print(f"No trajectory file found, can't excite ({traj_file})!")
        sys.exit(1)

    if args.dryrun:
        return

    excite_method = config.get("exciteMethod")
    if not excite_method:
        print("Error: exciteMethod must be set to 'yarp' or 'ros'.")
        print("For simulated measurements, use simulator.py instead.")
        sys.exit(1)

    # compute trajectory dynamics for reference torques
    traj_data, data = computeTrajectoryDynamics(config, trajectory)

    # excite real robot
    if excite_method == "yarp":
        from excitation.robotCommunication import yarp_gym

        yarp_gym.main(config, trajectory, traj_data)
    elif excite_method == "ros":
        from excitation.robotCommunication import ros_moveit

        ros_moveit.main(config, trajectory, traj_data)
    else:
        print(f"Error: unknown exciteMethod '{excite_method}'. Use 'yarp' or 'ros'.")
        sys.exit(1)

    # adapt measured array sizes to input array sizes
    traj_data["Q"] = np.resize(traj_data["Q"], data.samples["positions"].shape)
    traj_data["V"] = np.resize(traj_data["V"], data.samples["velocities"].shape)
    traj_data["Tau"] = np.resize(traj_data["Tau"], data.samples["torques"].shape)
    traj_data["T"] = np.resize(traj_data["T"], data.samples["times"].shape)

    # generate some empty arrays, will be calculated in preprocess()
    if "Vdot" not in traj_data:
        traj_data["Vdot"] = np.zeros_like(traj_data["V"])
    traj_data["Vraw"] = np.zeros_like(traj_data["V"])
    traj_data["Qraw"] = np.zeros_like(traj_data["Q"])
    traj_data["TauRaw"] = np.zeros_like(traj_data["Tau"])

    # filter, differentiate, convert, etc.
    data.preprocess(
        Q=traj_data["Q"],
        Q_raw=traj_data["Qraw"],
        V=traj_data["V"],
        V_raw=traj_data["Vraw"],
        Vdot=traj_data["Vdot"],
        Tau=traj_data["Tau"],
        Tau_raw=traj_data["TauRaw"],
        T=traj_data["T"],
        Fs=traj_data["measured_frequency"],
    )

    saveMeasurements(args.filename, traj_data)


def saveMeasurements(filename, data):
    """Write measured sample arrays to data file."""
    np.savez(
        filename,
        positions=data["Q"],
        positions_raw=data["Qraw"],
        velocities=data["V"],
        velocities_raw=data["Vraw"],
        accelerations=data["Vdot"],
        torques=data["Tau"],
        torques_raw=data["TauRaw"],
        target_positions=np.deg2rad(data["Qsent"]),
        target_velocities=np.deg2rad(data["QdotSent"]),
        target_accelerations=np.deg2rad(data["QddotSent"]),
        base_velocity=data["base_velocity"],
        base_acceleration=data["base_acceleration"],
        base_rpy=data["base_rpy"],
        contacts=data["contacts"],
        times=data["T"],
        frequency=data["measured_frequency"],
    )
    print(f"saved measurements to {args.filename}")


if __name__ == "__main__":
    main()
    if args.plot:
        plotter(config, filename=args.filename)
