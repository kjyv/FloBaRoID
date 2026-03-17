#!/usr/bin/env python

"""
open urdf, scale all masses and inertia with given value, save to new file
"""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np
from idyntree import bindings as iDynTree

parser = argparse.ArgumentParser(description="Scale mass and inertia from <model>.")
parser.add_argument("--model", required=True, type=str, help="the file to load the robot model from")
parser.add_argument("--output", required=True, type=str, help="the file to write the changed model to")
parser.add_argument("--scale", required=True, type=float, help="the value to scale with")
args = parser.parse_args()


def loadModel(urdf_file: str | bytes) -> iDynTree.Model:
    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(urdf_file)
    return loader.model()


def toNumPy(mat):
    return np.fromstring(mat.toString(), sep=" ").reshape(mat.rows(), mat.cols())


if __name__ == "__main__":
    scaling = args.scale

    model = loadModel(args.model)

    import xml.etree.ElementTree as ET

    # preserve comments
    class PCBuilder(ET.TreeBuilder):
        def comment(self, data):
            comment_tag = cast(str, ET.Comment)  # ET.Comment is a callable used as a special tag sentinel
            self.start(comment_tag, {})
            self.data(data)
            self.end(comment_tag)

    tree = ET.parse(args.model, parser=ET.XMLParser(target=PCBuilder()))

    for link_id in range(model.getNrOfLinks()):
        link_name = model.getLinkName(link_id)
        spat_inertia = model.getLink(link_id).getInertia()

        inertia = toNumPy(spat_inertia.getRotationalInertiaWrtCenterOfMass())
        mass = spat_inertia.getMass()

        mass_line = '<mass value="{}"/>'
        inertia_line = '<inertia ixx="{}" ixy="{}" ixz="{}" iyy="{}" iyz="{}" izz="{}"/>'

        if mass > 0:
            print(f"link {link_id} {link_name}")
            print(mass_line.format(mass))
            print(
                inertia_line.format(
                    inertia[0, 0],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 1],
                    inertia[1, 2],
                    inertia[2, 2],
                )
            )
            print("")

        inertia *= scaling
        mass *= scaling

        if mass > 0:
            print(mass_line.format(mass))
            print(
                inertia_line.format(
                    inertia[0, 0],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 1],
                    inertia[1, 2],
                    inertia[2, 2],
                )
            )

        for l in tree.findall("link"):
            if l.attrib["name"] == link_name:
                mass_el = l.find("inertial/mass")
                if mass_el is None:
                    continue
                mass_el.attrib["value"] = f"{mass}"
                inert = l.find("inertial/inertia")
                if inert is None:
                    continue
                inert.attrib["ixx"] = f"{inertia[0, 0]}"
                inert.attrib["ixy"] = f"{inertia[0, 1]}"
                inert.attrib["ixz"] = f"{inertia[0, 2]}"
                inert.attrib["iyy"] = f"{inertia[1, 1]}"
                inert.attrib["iyz"] = f"{inertia[1, 2]}"
                inert.attrib["izz"] = f"{inertia[2, 2]}"

    tree.write(args.output, xml_declaration=True)
