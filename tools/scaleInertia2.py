#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
open urdf, scale all masses and inertia with given value, save to new file
'''

import re
from typing import AnyStr
import numpy as np

import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()

import argparse
parser = argparse.ArgumentParser(description='Scale mass and inertia from <model>.')
parser.add_argument('--model', required=True, type=str, help='the file to load the robot model from')
parser.add_argument('--output', required=True, type=str, help='the file to write the changed model to')
parser.add_argument('--scale', required=True, type=float, help='the value to scale with')
args = parser.parse_args()

def loadModel(urdf_file):
    # type: (AnyStr) -> (iDynTree.DynamicsComputations)
    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(urdf_file)

    return dynComp

def toNumPy(mat):
    return np.fromstring(mat.toString(), sep=' ').reshape(mat.rows(), mat.cols())

if __name__ == '__main__':

    scaling = args.scale

    dynComp = loadModel(args.model)

    import xml.etree.ElementTree as ET
    # preserve comments
    class PCBuilder(ET.TreeBuilder):
        def comment(self, data):
            self.start(ET.Comment, {})
            self.data(data)
            self.end(ET.Comment)
    tree = ET.parse(args.model, parser=ET.XMLParser(target=PCBuilder()))

    for link_id in range(dynComp.getNrOfLinks()):
        link_name = dynComp.getFrameName(link_id)  # not really clean (when are there other frames than link frames, sensors?)
        link_id = dynComp.getLinkIndex(link_name)  # so, make double sure
        spat_inertia = dynComp.getLinkInertia(link_id)

        inertia = toNumPy(spat_inertia.getRotationalInertiaWrtCenterOfMass())
        mass = spat_inertia.getMass()

        mass_line = '<mass value="{}"/>'
        inertia_line = '<inertia ixx="{}" ixy="{}" ixz="{}" iyy="{}" iyz="{}" izz="{}"/>'

        if mass > 0:
            print("link {} {}".format(link_id, link_name))
            print(mass_line.format(mass))
            print(inertia_line.format(inertia[0,0], inertia[0,1], inertia[0,2],
                                                    inertia[1,1], inertia[1,2],
                                                                  inertia[2,2]))
            print("")

        inertia *= scaling
        mass *= scaling

        if mass > 0:
            print(mass_line.format(mass))
            print(inertia_line.format(inertia[0,0], inertia[0,1], inertia[0,2],
                                                    inertia[1,1], inertia[1,2],
                                                                  inertia[2,2]))

        for l in tree.findall('link'):
            if l.attrib['name'] == link_name:
                try:
                    l.find('inertial/mass').attrib['value'] = '{}'.format(mass)
                except AttributeError:
                    continue
                inert = l.find('inertial/inertia')
                inert.attrib['ixx'] = '{}'.format(inertia[0,0])
                inert.attrib['ixy'] = '{}'.format(inertia[0,1])
                inert.attrib['ixz'] = '{}'.format(inertia[0,2])
                inert.attrib['iyy'] = '{}'.format(inertia[1,1])
                inert.attrib['iyz'] = '{}'.format(inertia[1,2])
                inert.attrib['izz'] = '{}'.format(inertia[2,2])

    tree.write(args.output, xml_declaration=True)
