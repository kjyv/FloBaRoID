from __future__ import annotations

import os

# define exception for python < 3
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from identification.model import Model

import numpy as np
import numpy.linalg as la
from colorama import Fore
from idyntree import bindings as iDynTree
from tqdm import tqdm


def getNRMSE(
    data_ref: np.ndarray, data_est: np.ndarray, normalize: bool = True, limits: np.ndarray | list | None = None
) -> float:
    """get (normalized) root mean square error between estimated values and "standard".
    if limits is supplied, normalization is done from maximum range of torques rather than observed
    range in the data"""

    error = data_est - data_ref
    rmsd = np.sqrt(np.mean(error**2, axis=0))

    if normalize:
        if limits:
            # get min/max from urdf
            ymax = np.array(limits)
            ymin = -np.array(limits)
        else:
            # get min/max from data (not always informative)
            ymax = np.max(data_ref, axis=0)
            ymin = np.min(data_ref, axis=0)
        range = ymax - ymin
        if range.shape[0] < rmsd.shape[0]:
            # floating base
            return float(np.mean(rmsd[6:] / range) * 100)
        else:
            # fixed base
            return float(np.mean(rmsd / range) * 100)
    else:
        return float(np.mean(rmsd) * 100)


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class Progress:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config

    def progress(self, iter: Iterable) -> Iterable:
        if self.config["verbose"]:
            return tqdm(iter)
        else:
            return iter


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class ParamHelpers:
    def __init__(self, model: Model, opt: dict[str, Any]) -> None:
        self.model = model
        self.opt = opt

    def checkPhysicalConsistency(self, params: np.ndarray, full: bool = False) -> dict[int, bool]:
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite, triangle inequaltiy for eigenvalues of inertia tensor expressed at COM)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link

        when full is True, a 10 parameter per link vector is expected, regardless of global options
        """
        cons: dict[int, bool] = {}
        if self.opt["identifyGravityParamsOnly"] and not full:
            for i in range(0, self.model.num_links):
                # masses need to be positive
                cons[i] = cast(bool, params[i * 4] > 0)
        else:
            for i in range(0, params.shape[0]):
                if (i % 10 == 0) and i < self.model.num_model_params:  # for each link (and not friction)
                    p_vec = iDynTree.Vector10()
                    for j in range(0, 10):
                        p_vec.setVal(j, params[i + j])
                    si = iDynTree.SpatialInertia()
                    si.fromVector(p_vec)
                    cons[i // 10] = si.isPhysicallyConsistent()
        return cons

    def checkPhysicalConsistencyNoTriangle(self, params: np.ndarray, full: bool = False) -> dict[int, bool]:
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link

        when full is True, a 10 parameter per link vector is expected, regardless of global options
        """
        cons: dict[int, bool] = {}

        if self.opt["identifyGravityParamsOnly"] and not full:
            for i in range(0, self.model.num_links):
                # masses need to be positive
                cons[i] = cast(bool, params[i * 4] > 0)
        else:
            tensors = self.inertiaTensorFromParams(params)
            for i in range(0, len(params)):
                if (i % 10 == 0) and i < self.model.num_model_params:
                    if params[i] <= 0:  # masses need to be positive
                        cons[i // 10] = False
                        continue
                    # check if inertia tensor is positive definite (only then cholesky decomp exists)
                    try:
                        la.cholesky(tensors[i // 10])
                        cons[i // 10] = True
                    except la.LinAlgError:
                        cons[i // 10] = False
                else:
                    # TODO: check friction params >0
                    pass

        """
        if False in cons.values():
            print(Fore.RED + "Params are not consistent but ATM ignored" + Fore.RESET)
            print(cons)
        for k in cons:
            cons[k] = True
        """
        return cons

    def isPhysicalConsistent(self, params: np.ndarray) -> bool:
        """give boolean consistency statement for a set of parameters"""
        return False not in self.checkPhysicalConsistencyNoTriangle(params).values()

    def invvech(self, params: np.ndarray) -> np.ndarray:
        """give full inertia tensor from vectorized form
        expect vector of 6 values (xx, xy, xz, yy, yz, zz).T"""
        tensor = np.zeros((3, 3))
        # xx of tensor matrix
        value = params[0]
        tensor[0, 0] = value
        # xy
        value = params[1]
        tensor[0, 1] = value
        tensor[1, 0] = value
        # xz
        value = params[2]
        tensor[0, 2] = value
        tensor[2, 0] = value
        # yy
        value = params[3]
        tensor[1, 1] = value
        # yz
        value = params[4]
        tensor[1, 2] = value
        tensor[2, 1] = value
        # zz
        value = params[5]
        tensor[2, 2] = value
        return tensor

    def vech(self, params: np.ndarray) -> np.ndarray:
        """return vectorization of symmetric 3x3 matrix (only up to diagonal)"""
        vec = np.zeros(6)
        vec[0] = params[0, 0]
        vec[1] = params[0, 1]
        vec[2] = params[0, 2]
        vec[3] = params[1, 1]
        vec[4] = params[1, 2]
        vec[5] = params[2, 2]
        return vec

    def inertiaTensorFromParams(self, params: np.ndarray) -> list[np.ndarray]:
        """take a parameter vector and return list of full inertia tensors (one for each link)"""
        tensors = list()
        for i in range(len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:
                tensor = self.invvech(params[i + 4 : i + 10])
                tensors.append(tensor)
        return tensors

    def inertiaParams2RotationalInertiaRaw(self, params: np.ndarray) -> np.ndarray:
        """take values from inertia parameter vector and create iDynTree RotationalInertiaRaw matrix
        expects six parameter vector"""

        inertia = iDynTree.RotationalInertia()
        # xx of inertia matrix w.r.t. link origin
        value = params[0]
        inertia.setVal(0, 0, value)
        # xy
        value = params[1]
        inertia.setVal(0, 1, value)
        inertia.setVal(1, 0, value)
        # xz
        value = params[2]
        inertia.setVal(0, 2, value)
        inertia.setVal(2, 0, value)
        # yy
        value = params[3]
        inertia.setVal(1, 1, value)
        # yz
        value = params[4]
        inertia.setVal(1, 2, value)
        inertia.setVal(2, 1, value)
        # zz
        value = params[5]
        inertia.setVal(2, 2, value)
        return inertia

    def paramsLink2Bary(self, params: np.ndarray) -> np.ndarray:
        """convert params from iDynTree values (relative to link frame) to barycentric parameters
        (usable in URDF) (changed in place)"""

        # mass stays the same
        # linear com is first moment of mass, so com * mass. URDF uses com
        # linear inertia is expressed w.r.t. frame origin (-m*S(c).T*S(c)). URDF uses w.r.t com
        params = params.copy()
        for i in range(0, len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:  # for each link
                link_mass = params[i]
                # com
                com_x = params[i + 1]
                com_y = params[i + 2]
                com_z = params[i + 3]
                if link_mass != 0:
                    params[i + 1] = com_x / link_mass  # x of first moment -> x of com
                    params[i + 2] = com_y / link_mass  # y of first moment -> y of com
                    params[i + 3] = com_z / link_mass  # z of first moment -> z of com
                else:
                    params[i + 1] = params[i + 2] = params[i + 3] = 0
                p_com = iDynTree.Position(params[i + 1], params[i + 2], params[i + 3])

                # inertias
                rot_inertia_origin = self.inertiaParams2RotationalInertiaRaw(params[i + 4 : i + 10])
                s_inertia = iDynTree.SpatialInertia(link_mass, p_com, rot_inertia_origin)
                rot_inertia_com = s_inertia.getRotationalInertiaWrtCenterOfMass()
                params[i + 4] = rot_inertia_com.getVal(0, 0)  # xx w.r.t. com
                params[i + 5] = rot_inertia_com.getVal(0, 1)  # xy w.r.t. com
                params[i + 6] = rot_inertia_com.getVal(0, 2)  # xz w.r.t. com
                params[i + 7] = rot_inertia_com.getVal(1, 1)  # yy w.r.t. com
                params[i + 8] = rot_inertia_com.getVal(1, 2)  # yz w.r.t. com
                params[i + 9] = rot_inertia_com.getVal(2, 2)  # zz w.r.t. com
        return params

    def paramsBary2Link(self, params: np.ndarray) -> np.ndarray:
        params = params.copy()
        for i in range(0, len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:  # for each link
                link_mass = params[i]
                # com
                com_x = params[i + 1]
                com_y = params[i + 2]
                com_z = params[i + 3]
                params[i + 1] = com_x * link_mass  # x of first moment of mass
                params[i + 2] = com_y * link_mass  # y of first moment of mass
                params[i + 3] = com_z * link_mass  # z of first moment of mass
                p_com = iDynTree.Position(params[i + 1], params[i + 2], params[i + 3])

                # inertias
                rot_inertia_com = self.inertiaParams2RotationalInertiaRaw(params[i + 4 : i + 10])
                s_inertia = iDynTree.SpatialInertia(link_mass, p_com, rot_inertia_com)
                s_inertia.fromRotationalInertiaWrtCenterOfMass(link_mass, p_com, rot_inertia_com)
                rot_inertia = s_inertia.getRotationalInertiaWrtFrameOrigin()
                params[i + 4] = rot_inertia.getVal(0, 0)  # xx w.r.t. com
                params[i + 5] = rot_inertia.getVal(0, 1)  # xy w.r.t. com
                params[i + 6] = rot_inertia.getVal(0, 2)  # xz w.r.t. com
                params[i + 7] = rot_inertia.getVal(1, 1)  # yy w.r.t. com
                params[i + 8] = rot_inertia.getVal(1, 2)  # yz w.r.t. com
                params[i + 9] = rot_inertia.getVal(2, 2)  # zz w.r.t. com

        return params

    @staticmethod
    def addFrictionFromURDF(model: Model, urdf_file: str, params: np.ndarray) -> None:
        """get friction vals from urdf (joint friction = fc, damping= fv) and set in params vector"""

        friction = URDFHelpers.getJointFriction(urdf_file)
        nd = model.num_dofs
        start = model.num_model_params

        for i in range(len(model.jointNames)):
            j = model.jointNames[i]
            params[start + i] = friction[j]["f_constant"]

            if not model.opt["identifyGravityParamsOnly"]:
                params[start + nd + i] = friction[j]["f_velocity"]

                if not model.opt["identifySymmetricVelFriction"]:
                    # same value again for asymmetric value since urdf does only have one value
                    params[start + nd + nd + i] = friction[j]["f_velocity"]


class URDFHelpers:
    def __init__(self, paramHelpers: ParamHelpers, model: Model, opt: dict) -> None:
        self.paramHelpers = paramHelpers
        self.model = model
        self.opt = opt
        self.parsed_xml: dict[str, Any] = {}

    def parseURDF(self, input_urdf: str) -> ET.ElementTree[Any]:

        try:
            return self.parsed_xml[input_urdf]
        except KeyError:
            # preserve comments
            class PCBuilder(ET.TreeBuilder):
                def comment(self, data):
                    comment_tag = cast(str, ET.Comment)  # ET.Comment is a callable used as a special tag sentinel
                    self.start(comment_tag, {})
                    self.data(data)
                    self.end(comment_tag)

            tree = ET.parse(input_urdf, parser=ET.XMLParser(target=PCBuilder()))
            self.parsed_xml[input_urdf] = tree
            return tree

    def replaceParamsInURDF(self, input_urdf: str, output_urdf: str, new_params: np.ndarray) -> None:
        """set new inertia parameters from params and urdf_file, write to new temp file"""

        if self.opt["identifyGravityParamsOnly"]:
            per_link = 4
            xStdBary = new_params.copy()
            for i in range(self.model.num_links):
                xStdBary[i * per_link + 1 : i * per_link + 3 + 1] /= xStdBary[i * per_link]
        else:
            per_link = 10
            xStdBary = self.paramHelpers.paramsLink2Bary(new_params)

        tree = self.parseURDF(input_urdf)

        for l in tree.findall("link"):
            if l.attrib["name"] in self.model.linkNames:
                link_id = self.model.linkNames.index(l.attrib["name"])
                mass_el = l.find("inertial/mass")
                if mass_el is not None:
                    mass_el.attrib["value"] = f"{xStdBary[link_id * per_link]}"
                origin_el = l.find("inertial/origin")
                if origin_el is not None:
                    origin_el.attrib["xyz"] = (
                        f"{xStdBary[link_id * per_link + 1]} {xStdBary[link_id * per_link + 2]} {xStdBary[link_id * per_link + 3]}"
                    )
                if not self.opt["identifyGravityParamsOnly"]:
                    inert = l.find("inertial/inertia")
                    if inert is None:
                        continue
                    inert.attrib["ixx"] = f"{xStdBary[link_id * 10 + 4]}"
                    inert.attrib["ixy"] = f"{xStdBary[link_id * 10 + 5]}"
                    inert.attrib["ixz"] = f"{xStdBary[link_id * 10 + 6]}"
                    inert.attrib["iyy"] = f"{xStdBary[link_id * 10 + 7]}"
                    inert.attrib["iyz"] = f"{xStdBary[link_id * 10 + 8]}"
                    inert.attrib["izz"] = f"{xStdBary[link_id * 10 + 9]}"

        # write friction of joints
        for l in tree.findall("joint"):
            if l.attrib["name"] in self.model.jointNames:
                joint_id = self.model.jointNames.index(l.attrib["name"])
                if self.opt["identifyFriction"]:
                    f_c = cast(float, xStdBary[self.model.num_links * per_link + joint_id])
                    if self.opt["identifyGravityParamsOnly"]:
                        f_v = 0.0
                    else:
                        if self.opt["identifySymmetricVelFriction"]:
                            f_v = cast(
                                float,
                                xStdBary[self.model.num_model_params + self.model.num_dofs + joint_id],
                            )
                        else:
                            print(
                                Fore.RED
                                + "Can't write velocity dependent friction to URDF as identified values are asymmetric. URDF only supports symmetric values."
                                + Fore.RESET
                            )
                            sys.exit(1)
                else:
                    # parameters were identified assuming there was no friction
                    f_c = f_v = 0.0
                dynamics_el = l.find("dynamics")
                if dynamics_el is not None:
                    dynamics_el.attrib["friction"] = f"{f_c}"
                    if not self.opt["identifyGravityParamsOnly"]:
                        dynamics_el.attrib["damping"] = f"{f_v}"

        tree.write(output_urdf, xml_declaration=True)

    def getLinkNames(self, input_urdf: str) -> list[str]:

        links = []
        tree = self.parseURDF(input_urdf)
        for l in tree.findall("link"):
            if len(list(l)) > 0:  # ignore fake links
                links.append(l.attrib["name"])
        return links

    def getMeshPath(self, input_urdf: str, link_name: str) -> str | None:

        tree = self.parseURDF(input_urdf)
        link_found = False
        filepath: str | None = None
        for l in tree.findall("link"):
            if l.attrib["name"] == link_name:
                link_found = True
                m = l.find("visual/geometry/mesh")
                if m is not None:
                    filepath = m.attrib["filename"]
                    try:
                        self.mesh_scaling = m.attrib["scale"]
                    except KeyError:
                        self.mesh_scaling = "1 1 1"

        if not link_found or m is None:
            # print(Fore.RED + "No mesh information specified for link '{}' in URDF! Using a very large box.".format(link_name) + Fore.RESET)
            filepath = None

        # if path is ros package path, get absolute system path
        if filepath and (filepath.startswith("package") or filepath.startswith("model")):
            try:
                import resource_retriever  # ros package

                r = resource_retriever.get(filepath)
                filepath = r.url.replace("file://", "")
                # r.read() #get file into memory
            except ImportError:
                # if no ros installed, try to get stl files from 'meshes' dir relative to urdf files
                filename_parts = filepath.split("/")
                try:
                    filename_parts = filename_parts[filename_parts.index(self.opt["meshBaseDir"]) :]
                    filepath = "/".join(input_urdf.split("/")[:-1] + filename_parts)
                except ValueError:
                    filepath = None

        return filepath

    def getLinkGeometry(
        self, input_urdf: str, link_name: str
    ) -> tuple[list[float], list[float], list[float] | np.ndarray]:

        tree = self.parseURDF(input_urdf)

        def getBoxAttribs(m, l):
            box_size = m.attrib["size"]
            m = l.find("visual/origin")
            if m is not None:
                try:
                    box_pos = m.attrib["xyz"]
                except:
                    box_pos = "0 0 0"

                try:
                    box_rpy = m.attrib["rpy"]
                except:
                    box_rpy = "0 0 0"
            else:
                box_pos = box_rpy = "0 0 0"
            return box_size, box_pos, box_rpy

        box_size_s = box_pos_s = box_rpy_s = "0 0 0"
        for l in tree.findall("link"):
            if l.attrib["name"] == link_name:
                m = l.find("visual/geometry/box")
                if m is not None:
                    box_size_s, box_pos_s, box_rpy_s = getBoxAttribs(m, l)
                else:
                    m = l.find("visual/collision/box")
                    if m is not None:
                        box_size_s, box_pos_s, box_rpy_s = getBoxAttribs(m, l)
                break
        box_size = [float(i) for i in box_size_s.split()]
        box_pos = [float(i) for i in box_pos_s.split()]
        box_rpy = [float(i) for i in box_rpy_s.split()]
        return (box_size, box_pos, box_rpy)

    @staticmethod
    def getNeighbors(idyn_model: Any, connected: bool = True) -> dict[str, dict[str, list[Any]]]:

        # get neighbors for each link
        neighbors: dict[str, dict[str, list[str]]] = {}
        for l in range(idyn_model.getNrOfLinks()):
            link_name = idyn_model.getLinkName(l)
            # if link_name not in self.model.linkNames:  # ignore links that are ignored in the generator
            #    continue
            neighbors[link_name] = {"links": [], "joints": []}
            num_neighbors = idyn_model.getNrOfNeighbors(l)
            for n in range(num_neighbors):
                nb = idyn_model.getNeighbor(l, n)
                neighbors[link_name]["links"].append(idyn_model.getLinkName(nb.neighborLink))
                neighbors[link_name]["joints"].append(idyn_model.getJointName(nb.neighborJoint))

        if connected:
            # for each neighbor link, add links connected via a fixed joint also as neighbors
            neighbors_tmp = neighbors.copy()  # don't modify in place so no recursive loops happen
            for l in range(idyn_model.getNrOfLinks()):
                link_name = idyn_model.getLinkName(l)
                for nb in neighbors_tmp[link_name]["links"]:  # look at all neighbors of l
                    for j_name in neighbors_tmp[nb]["joints"]:  # check each joint of a neighbor of l
                        j = idyn_model.getJoint(idyn_model.getJointIndex(j_name))
                        # check all connected joints if they are fixed, if so add connected link as neighbor
                        if j.isFixedJoint():
                            j_l0 = j.getFirstAttachedLink()
                            j_l1 = j.getSecondAttachedLink()
                            if j_l0 == idyn_model.getLinkIndex(nb):
                                nb_fixed = j_l1
                            else:
                                nb_fixed = j_l0
                            nb_fixed_name = idyn_model.getLinkName(nb_fixed)
                            if nb_fixed != l and nb_fixed_name not in neighbors[link_name]["links"]:
                                neighbors[link_name]["links"].append(nb_fixed_name)

        return neighbors

    def getBoundingBox(
        self, input_urdf: str, old_com: list[float], link_name: str, scaling: bool = True
    ) -> tuple[list[list[float]], list[float], np.ndarray | list[float]]:
        """Return bounding box for one link derived from mesh file if possible.
        If no mesh file is found, a cube around the old COM is returned.
        Expects old_com in barycentric form!"""

        import trimesh

        filename = self.getMeshPath(input_urdf, link_name)

        # box around current COM in case no mesh is availabe
        length = self.opt["cubeSize"]
        cube = [
            [
                -0.5 * length + old_com[0],
                -0.5 * length + old_com[1],
                -0.5 * length + old_com[2],
            ],
            [
                0.5 * length + old_com[0],
                0.5 * length + old_com[1],
                0.5 * length + old_com[2],
            ],
        ]
        if scaling:
            hullScale = self.opt["hullScaling"]
        else:
            hullScale = 1.0
        pos_0 = [0.0, 0.0, 0.0]
        rot_0 = np.identity(3)

        if filename and os.path.exists(filename):
            mesh = trimesh.load_mesh(filename)

            # get geometry origin from URDF <visual>/<collision> <origin> tag
            tree = self.parseURDF(input_urdf)
            mesh_pos = pos_0
            mesh_rot = rot_0
            for l in tree.findall("link"):
                if l.attrib["name"] == link_name:
                    origin = l.find("visual/origin")
                    if origin is None:
                        origin = l.find("collision/origin")
                    if origin is not None:
                        xyz = origin.attrib.get("xyz", "0 0 0")
                        mesh_pos = [float(v) for v in xyz.split()]
                        rpy = origin.attrib.get("rpy", "0 0 0")
                        rpy_vals = [float(v) for v in rpy.split()]
                        if any(v != 0 for v in rpy_vals):
                            mesh_rot = eulerAnglesToRotationMatrix(rpy_vals)
                    break

            # gazebo and urdf use 1m for 1 stl unit
            scale_x = float(self.mesh_scaling.split()[0])
            scale_y = float(self.mesh_scaling.split()[1])
            scale_z = float(self.mesh_scaling.split()[2])

            bounding_box = mesh.bounding_box.bounds * scale_x * hullScale

            # switch order of min/max if scaling is negative
            for s in range(0, 3):
                if [scale_x, scale_y, scale_z][s] < 0:
                    bounding_box[0][s], bounding_box[1][s] = (
                        bounding_box[1][s],
                        bounding_box[0][s],
                    )

            return bounding_box, mesh_pos, mesh_rot
        else:
            # use <visual><box> or <collision><box> if specified
            box, pos, rot = self.getLinkGeometry(input_urdf, link_name)
            if np.any(np.array(box) != 0):
                return (
                    [
                        [
                            -0.5 * box[0] * hullScale,
                            -0.5 * box[1] * hullScale,
                            -0.5 * box[2] * hullScale,
                        ],
                        [
                            0.5 * box[0] * hullScale,
                            0.5 * box[1] * hullScale,
                            0.5 * box[2] * hullScale,
                        ],
                    ],
                    pos,
                    rot,
                )
            else:
                if self.opt["verbose"]:
                    print(
                        Fore.YELLOW
                        + f"Mesh file {filename} or box geometry not found for link '{link_name}'! Using a {length}m cube around a priori COM."
                        + Fore.RESET
                    )
                return cube, pos_0, rot_0

    @staticmethod
    def getJointLimits(input_urdf: str, use_deg: bool = False) -> dict[str, dict[str, float]]:
        import xml.etree.ElementTree as ET

        tree = ET.parse(input_urdf)
        limits: dict[str, dict[str, float]] = {}
        for j in tree.findall("joint"):
            name = j.attrib["name"]
            torque = 0.0
            lower = 0.0
            upper = 0.0
            velocity = 0.0
            if j.attrib["type"] == "revolute":
                l = j.find("limit")
                if l is not None:
                    torque = float(l.attrib["effort"])  # this is not really the physical limit but controller limit
                    lower = float(l.attrib["lower"])
                    upper = float(l.attrib["upper"])
                    velocity = float(l.attrib["velocity"])

                    limits[name] = {}
                    limits[name]["torque"] = float(torque)
                    if use_deg:
                        limits[name]["lower"] = np.rad2deg(lower)
                        limits[name]["upper"] = np.rad2deg(upper)
                        limits[name]["velocity"] = np.rad2deg(velocity)
                    else:
                        limits[name]["lower"] = lower
                        limits[name]["upper"] = upper
                        limits[name]["velocity"] = velocity
        return limits

    @staticmethod
    def getJointFriction(input_urdf: str) -> dict[str, dict[str, float]]:
        """return friction values for each revolute joint from a urdf"""

        import xml.etree.ElementTree as ET

        tree = ET.parse(input_urdf)
        friction: dict[str, dict[str, float]] = {}
        for j in tree.findall("joint"):
            name = j.attrib["name"]
            constant = 0.0
            vel_dependent = 0.0
            if j.attrib["type"] == "revolute":
                l = j.find("dynamics")
                if l is not None:
                    try:
                        constant = float(l.attrib["friction"])
                    except KeyError:
                        constant = 0

                    try:
                        vel_dependent = float(l.attrib["damping"])
                    except KeyError:
                        vel_dependent = 0

                friction[name] = {}
                friction[name]["f_constant"] = constant
                friction[name]["f_velocity"] = vel_dependent
        return friction
