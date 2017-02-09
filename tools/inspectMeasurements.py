#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
import numpy as np
import matplotlib.pyplot as plt

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from IPython import embed
import re

import argparse
parser = argparse.ArgumentParser(description='Open a previously taken measurements.npz file and drop into ipython')
parser.add_argument('--filename', required=True, type=str, help='the filename to load the measurements from')
parser.add_argument('--model', required=False, type=str, help='the file to load the robot model from')
#parser.add_argument('--config', required=True, type=str, help="use options from given config file")

#parser.add_argument('--plot', help='plot measured data', action='store_true')
#parser.set_defaults(plot=False,)
args = parser.parse_args()

def mapToJointNames(matrix, row=None):
    generator = iDynTree.DynamicsRegressorGenerator()
    generator.loadRobotAndSensorsModelFromFile(args.model)
    regrXml = '''
    <regressor>
      <jointTorqueDynamics>
        <allJoints/>
      </jointTorqueDynamics>
    </regressor>'''
    generator.loadRegressorStructureFromString(regrXml)
    jointNames = re.sub(r"DOF Index: \d+ Name: ", "", generator.getDescriptionOfDegreesOfFreedom()).split()

    if row:
        return {jointNames[j]:matrix[row,j] for j in range(matrix.shape[1])}
    else:
        return {jointNames[j]:matrix[:,j] for j in range(matrix.shape[1])}

def main():
    data = np.load(args.filename)
    print("loaded file into 'data'")
    print("data.keys():")
    print(data.keys())
    print("enter e.g. mapToJointNames(data['torques'], row=50)")
    print("")
    embed()

if __name__ == '__main__':
    main()
