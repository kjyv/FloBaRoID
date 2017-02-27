#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import input
from builtins import zip
from builtins import range
from builtins import object
import sys
from typing import Dict, List, Iterable

# math
import numpy as np
import numpy.linalg as la
import scipy
import scipy.linalg as sla
import scipy.stats as stats

# plotting
import matplotlib.pyplot as plt

# kinematics, dynamics and URDF reading
import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

from IPython import embed

import argparse
parser = argparse.ArgumentParser(description='Scale mass and inertia from <model>.')
parser.add_argument('--ref_model', required=True, type=str, help='the file to load the reference parameters from')
parser.add_argument('--model', required=True, nargs='+', action='append', type=str, help='the file to load the parameters to compare from')
args = parser.parse_args()

def loadModelfromURDF(urdf_file):
    # type: (AnyStr) -> (iDynTree.DynamicsComputations)
    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(urdf_file)

    return dynComp

def matToNumPy(mat):
    return np.fromstring(mat.toString(), sep=' ').reshape(mat.rows(), mat.cols())

def vecToNumPy(vec):
    return np.fromstring(vec.toString(), sep=' ')

def plotErrors(errors, num_links, labels):
    fig, ax = plt.subplots()

    num_vals = len(errors[0])
    std_dev = np.zeros(num_vals)

    index = np.arange(num_vals)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    colors = ['r', 'g', 'b']

    for i in range(len(errors)):
        plt.bar(index+bar_width*i, errors[i], bar_width,
                     alpha=opacity,
                     color=colors[i],
                     yerr=std_dev,
                     error_kw=error_config,
                     label=labels[i])

    plt.xlabel('Link Index')
    plt.ylabel('Error Norm')
    plt.title('Param error by method')
    plt.xticks(index + bar_width / 2, linkNames, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()

def getParamErrors(ref_model, p_model, num_links, group="COM"):
    # type: (iDynTree.DynamicsComputations) -> (List[float])
    """ give error for groups of params """

    errors = []

    for link_id in range(num_links):
        p_link_name = linkNames[link_id]
        p_link_id = p_model.getLinkIndex(p_link_name)
        p_spat_inertia = p_model.getLinkInertia(p_link_id)
        p_inertia = matToNumPy(p_spat_inertia.getRotationalInertiaWrtCenterOfMass())
        p_mass = p_spat_inertia.getMass()
        p_com = vecToNumPy(p_spat_inertia.getCenterOfMass())

        ref_link_name = linkNames[link_id]
        ref_link_id = ref_model.getLinkIndex(ref_link_name)
        ref_spat_inertia = ref_model.getLinkInertia(ref_link_id)
        ref_inertia = matToNumPy(ref_spat_inertia.getRotationalInertiaWrtCenterOfMass())
        ref_mass = ref_spat_inertia.getMass()
        ref_com = vecToNumPy(ref_spat_inertia.getCenterOfMass())

        if group == 'COM':
            errors.append(ref_com - p_com)
        elif group == 'mass':
            errors.append(ref_mass - p_mass)
        elif group == 'inertia':
            errors.append(ref_inertia - p_inertia)

    return errors

if __name__ == '__main__':

    ref_model = loadModelfromURDF(args.ref_model)
    generator = iDynTree.DynamicsRegressorGenerator()
    if not generator.loadRobotAndSensorsModelFromFile(args.ref_model):
        sys.exit()

    regrXml = '''
    <regressor>
      <jointTorqueDynamics>
        <allJoints/>
      </jointTorqueDynamics>
    </regressor>'''
    generator.loadRegressorStructureFromString(regrXml)
    num_links = generator.getNrOfLinks() - generator.getNrOfFakeLinks()

    linkNames = []
    import re
    for d in generator.getDescriptionOfParameters().strip().split("\n"):
        link = re.findall(r"of link (.*)", d)[0]
        if link not in linkNames:
            linkNames.append(link)

    methods_com_errors = []
    methods_inertia_errors = []
    for file in args.model:
        p_model = loadModelfromURDF(file[0])

        # mass error norm
        mass_errors = np.linalg.norm(getParamErrors(ref_model, p_model, num_links, group='mass'))

        # com error norm
        com_errors = np.linalg.norm(getParamErrors(ref_model, p_model, num_links, group='COM'), axis=1)

        # inertia error norm
        inertia_error_tensors = getParamErrors(ref_model, p_model, num_links, group='inertia')
        inertia_errors = []
        for i in range(len(inertia_error_tensors)):
            inertia_errors.append(np.linalg.norm(inertia_error_tensors[i]))

        #methods_mass_errors.append(mass_errors)
        methods_com_errors.append(com_errors)
        methods_inertia_errors.append(inertia_errors)

    plotErrors(methods_com_errors, num_links, labels=['ID COM (Kown Mass)', "ID Inertia (Known Mass + ID'd COM)", 'ID Inertia + COM (Known Mass)'])
    plotErrors(methods_inertia_errors, num_links, labels=['ID COM (Kown Mass)', "ID Inertia (Known Mass + ID'd COM)", 'ID Inertia + COM (Known Mass)'])
