#!/usr/bin/env python

import argparse
import re

import numpy as np
from idyntree import bindings as iDynTree
from IPython import embed

parser = argparse.ArgumentParser(description="Open a previously taken measurements.npz file and drop into ipython")
parser.add_argument(
    "--filename",
    required=True,
    type=str,
    help="the filename to load the measurements from",
)
parser.add_argument("--model", required=False, type=str, help="the file to load the robot model from")
parser.add_argument("--fb", required=False, type=bool, help="is the model floating base?")
# parser.add_argument('--config', required=True, type=str, help="use options from given config file")

# parser.add_argument('--plot', help='plot measured data', action='store_true')
# parser.set_defaults(plot=False,)
args = parser.parse_args()


def mapToJointNames(matrix, row=None):
    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(args.model)
    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(loader.model())
    jointNames = re.sub(r"DOF Index: \d+ Name: ", "", kinDyn.getDescriptionOfDegreesOfFreedom()).split()

    if args.fb:
        fb = 6
    else:
        fb = 0

    if row:
        return {jointNames[j - fb]: matrix[row, j] for j in range(matrix.shape[1])}
    else:
        return {jointNames[j - fb]: matrix[:, j] for j in range(matrix.shape[1])}


def main():
    data = np.load(args.filename)
    print("loaded file into 'data'")
    print("data.keys():")
    print(data.keys())
    print("enter e.g. mapToJointNames(data['torques'], row=50)")
    print("")
    embed()


if __name__ == "__main__":
    main()
