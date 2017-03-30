from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import time
from typing import cast, Any, List, Dict, Iterable, Union, Tuple, AnyStr
import os

import numpy as np
import numpy.linalg as la

import xml.etree.ElementTree as ET

from colorama import Fore
from tqdm import tqdm

import iDynTree

#define exception for python < 3
import sys
if (sys.version_info < (3, 0)):
    class FileNotFoundError(OSError):
        pass

def getNRMSE(data_ref, data_est, normalize=True, limits=None):
    # type: (np._ArrayLike, np._ArrayLike, bool, np._ArrayLike) -> np._ArrayLike[float]
    '''get (normalized) root mean square error between estimated values and "standard".
    if limits is supplied, normalization is done from maximum range of torques rather than observed
    range in the data '''

    error = data_est - data_ref
    rmsd = np.sqrt(np.mean(error**2, axis=0))

    if normalize:
        if limits:
            #get min/max from urdf
            ymax = np.array(limits)
            ymin = -np.array(limits)
        else:
            # get min/max from data (not always informative)
            ymax = np.max(data_ref, axis=0)
            ymin = np.min(data_ref, axis=0)
        range = (ymax - ymin)
        if range.shape[0] < rmsd.shape[0]:
            # floating base
            return np.mean(rmsd[6:] / range) * 100
        else:
            # fixed base
            return np.mean(rmsd / range) * 100
    else:
        return np.mean(rmsd) * 100

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0               ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                   ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0               ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                   ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                  1]
                   ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

class Progress(object):
    def __init__(self, config):
        # type: (Dict[str, Any]) -> None
        self.config = config   # type: Dict[str, Any]

    def progress(self, iter):
        # type: (Iterable) -> Iterable
        if self.config['verbose']:
            return tqdm(iter)
        else:
            return iter


class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


class ParamHelpers(object):
    def __init__(self, model, opt):
        # type: (Model, Dict[str, Any]) -> None
        self.model = model
        self.opt = opt

    def checkPhysicalConsistency(self, params, full=False):
        # type: (np._ArrayLike, bool) -> (Dict[int, bool])
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite, triangle inequaltiy for eigenvalues of inertia tensor expressed at COM)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link

        when full is True, a 10 parameter per link vector is expected, regardless of global options
        """
        cons = {}  # type: (Dict[int, bool])
        if self.opt['identifyGravityParamsOnly'] and not full:
            for i in range(0, self.model.num_links):
                #masses need to be positive
                cons[i] = cast(bool, params[i*4] > 0)
        else:
            for i in range(0, params.shape[0]):
                if (i % 10 == 0) and i < self.model.num_model_params:   #for each link (and not friction)
                    p_vec = iDynTree.Vector10()
                    for j in range(0, 10):
                        p_vec.setVal(j, params[i+j])
                    si = iDynTree.SpatialInertia()
                    si.fromVector(p_vec)
                    cons[i // 10] = si.isPhysicallyConsistent()
        return cons

    def checkPhysicalConsistencyNoTriangle(self, params, full=False):
        # type: (np._ArrayLike, bool) -> (Dict[int, bool])
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link

        when full is True, a 10 parameter per link vector is expected, regardless of global options
        """
        cons = {}   # type: (Dict[int, bool])

        if self.opt['identifyGravityParamsOnly'] and not full:
            for i in range(0, self.model.num_links):
                #masses need to be positive
                cons[i] = cast(bool, params[i*4] > 0)
        else:
            tensors = self.inertiaTensorFromParams(params)
            for i in range(0, len(params)):
                if (i % 10 == 0) and i < self.model.num_model_params:
                    if params[i] <= 0:  #masses need to be positive
                        cons[i // 10] = False
                        continue
                    #check if inertia tensor is positive definite (only then cholesky decomp exists)
                    try:
                        la.cholesky(tensors[i // 10])
                        cons[i // 10] = True
                    except la.LinAlgError:
                        cons[i // 10] = False
                else:
                    #TODO: check friction params >0
                    pass

        '''
        if False in cons.values():
            print(Fore.RED + "Params are not consistent but ATM ignored" + Fore.RESET)
            print(cons)
        for k in cons:
            cons[k] = True
        '''
        return cons

    def isPhysicalConsistent(self, params):
        # type: (np._ArrayLike[float]) -> bool
        """give boolean consistency statement for a set of parameters"""
        return not (False in self.checkPhysicalConsistencyNoTriangle(params).values())

    def invvech(self, params):
        # type: (np._ArrayLike[float]) -> (np._ArrayLike[float])
        """give full inertia tensor from vectorized form
           expect vector of 6 values (xx, xy, xz, yy, yz, zz).T"""
        tensor = np.zeros((3,3))
        #xx of tensor matrix
        value = params[0]
        tensor[0, 0] = value
        #xy
        value = params[1]
        tensor[0, 1] = value
        tensor[1, 0] = value
        #xz
        value = params[2]
        tensor[0, 2] = value
        tensor[2, 0] = value
        #yy
        value = params[3]
        tensor[1, 1] = value
        #yz
        value = params[4]
        tensor[1, 2] = value
        tensor[2, 1] = value
        #zz
        value = params[5]
        tensor[2, 2] = value
        return tensor

    def vech(self, params):
        # type: (np._ArrayLike[float]) -> (np._ArrayLike[float])
        """return vectorization of symmetric 3x3 matrix (only up to diagonal)"""
        vec = np.zeros(6)
        vec[0] = params[0,0]
        vec[1] = params[0,1]
        vec[2] = params[0,2]
        vec[3] = params[1,1]
        vec[4] = params[1,2]
        vec[5] = params[2,2]
        return vec

    def inertiaTensorFromParams(self, params):
        # type: (np._ArrayLike[float]) -> (List[np._ArrayLike[float]])
        """take a parameter vector and return list of full inertia tensors (one for each link)"""
        tensors = list()
        for i in range(len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:
                tensor = self.invvech(params[i+4:i+10])
                tensors.append(tensor)
        return tensors

    def inertiaParams2RotationalInertiaRaw(self, params):
        # type: (np._ArrayLike[float]) -> (np._ArrayLike[float])
        """take values from inertia parameter vector and create iDynTree RotationalInertiaRaw matrix
        expects six parameter vector"""

        inertia = iDynTree.RotationalInertiaRaw()
        #xx of inertia matrix w.r.t. link origin
        value = params[0]
        inertia.setVal(0, 0, value)
        #xy
        value = params[1]
        inertia.setVal(0, 1, value)
        inertia.setVal(1, 0, value)
        #xz
        value = params[2]
        inertia.setVal(0, 2, value)
        inertia.setVal(2, 0, value)
        #yy
        value = params[3]
        inertia.setVal(1, 1, value)
        #yz
        value = params[4]
        inertia.setVal(1, 2, value)
        inertia.setVal(2, 1, value)
        #zz
        value = params[5]
        inertia.setVal(2, 2, value)
        return inertia

    def paramsLink2Bary(self, params):
        # type: (np._ArrayLike[float]) -> (np._ArrayLike[float])
        """convert params from iDynTree values (relative to link frame) to barycentric parameters
           (usable in URDF) (changed in place)"""

        #mass stays the same
        #linear com is first moment of mass, so com * mass. URDF uses com
        #linear inertia is expressed w.r.t. frame origin (-m*S(c).T*S(c)). URDF uses w.r.t com
        params = params.copy()
        for i in range(0, len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:   #for each link
                link_mass = params[i]
                #com
                com_x = params[i+1]
                com_y = params[i+2]
                com_z = params[i+3]
                if link_mass != 0:
                    params[i+1] = com_x / link_mass  #x of first moment -> x of com
                    params[i+2] = com_y / link_mass  #y of first moment -> y of com
                    params[i+3] = com_z / link_mass  #z of first moment -> z of com
                else:
                    params[i+1] = params[i+2] = params[i+3] = 0
                p_com = iDynTree.PositionRaw(params[i+1], params[i+2], params[i+3])

                #inertias
                rot_inertia_origin = self.inertiaParams2RotationalInertiaRaw(params[i+4:i+10])
                s_inertia = iDynTree.SpatialInertia(link_mass, p_com, rot_inertia_origin)
                rot_inertia_com = s_inertia.getRotationalInertiaWrtCenterOfMass()
                params[i+4] = rot_inertia_com.getVal(0, 0)    #xx w.r.t. com
                params[i+5] = rot_inertia_com.getVal(0, 1)    #xy w.r.t. com
                params[i+6] = rot_inertia_com.getVal(0, 2)    #xz w.r.t. com
                params[i+7] = rot_inertia_com.getVal(1, 1)    #yy w.r.t. com
                params[i+8] = rot_inertia_com.getVal(1, 2)    #yz w.r.t. com
                params[i+9] = rot_inertia_com.getVal(2, 2)    #zz w.r.t. com
        return params

    def paramsBary2Link(self, params):
        # type: (np._ArrayLike[float]) -> (np._ArrayLike[float])
        params = params.copy()
        for i in range(0, len(params)):
            if (i % 10 == 0) and i < self.model.num_model_params:   #for each link
                link_mass = params[i]
                #com
                com_x = params[i+1]
                com_y = params[i+2]
                com_z = params[i+3]
                params[i+1] = com_x * link_mass  #x of first moment of mass
                params[i+2] = com_y * link_mass  #y of first moment of mass
                params[i+3] = com_z * link_mass  #z of first moment of mass
                p_com = iDynTree.PositionRaw(params[i+1], params[i+2], params[i+3])

                #inertias
                rot_inertia_com = self.inertiaParams2RotationalInertiaRaw(params[i+4:i+10])
                s_inertia = iDynTree.SpatialInertia(link_mass, p_com, rot_inertia_com)
                s_inertia.fromRotationalInertiaWrtCenterOfMass(link_mass, p_com, rot_inertia_com)
                rot_inertia = s_inertia.getRotationalInertiaWrtFrameOrigin()
                params[i+4] = rot_inertia.getVal(0, 0)    #xx w.r.t. com
                params[i+5] = rot_inertia.getVal(0, 1)    #xy w.r.t. com
                params[i+6] = rot_inertia.getVal(0, 2)    #xz w.r.t. com
                params[i+7] = rot_inertia.getVal(1, 1)    #yy w.r.t. com
                params[i+8] = rot_inertia.getVal(1, 2)    #yz w.r.t. com
                params[i+9] = rot_inertia.getVal(2, 2)    #zz w.r.t. com

        return params

    @staticmethod
    def addFrictionFromURDF(model, urdf_file, params):
        # type: (model.Model, str, np._ArrayLike[float]) -> None
        ''' get friction vals from urdf (joint friction = fc, damping= fv) and set in params vector'''

        friction = URDFHelpers.getJointFriction(urdf_file)
        nd = model.num_dofs
        start = model.num_model_params

        for i in range(len(model.jointNames)):
            j = model.jointNames[i]
            params[start+i] = friction[j]['f_constant']

            if not model.opt['identifyGravityParamsOnly']:
                params[start+nd+i] = friction[j]['f_velocity']

                if not model.opt['identifySymmetricVelFriction']:
                    # same value again for asymmetric value since urdf does only have one value
                    params[start+nd+nd+i] = friction[j]['f_velocity']


class URDFHelpers(object):
    def __init__(self, paramHelpers, model, opt):
        # type: (ParamHelpers, model.Model, Dict) -> None
        self.paramHelpers = paramHelpers
        self.model = model
        self.opt = opt
        self.parsed_xml = {}   # type: Dict[ET]

    def parseURDF(self, input_urdf):
        # type: (str) -> ET

        try:
            return self.parsed_xml[input_urdf]
        except KeyError:
            # preserve comments
            class PCBuilder(ET.TreeBuilder):
                def comment(self, data):
                    self.start(ET.Comment, {})
                    self.data(data)
                    self.end(ET.Comment)
            tree = ET.parse(input_urdf, parser=ET.XMLParser(target=PCBuilder()))
            self.parsed_xml[input_urdf] = tree
            return tree

    def replaceParamsInURDF(self, input_urdf, output_urdf, new_params):
        # type: (str, str, np._ArrayLike[float]) -> None
        """ set new inertia parameters from params and urdf_file, write to new temp file """

        if self.opt['identifyGravityParamsOnly']:
            per_link = 4
            xStdBary = new_params.copy()
            for i in range(self.model.num_links):
                xStdBary[i*per_link+1:i*per_link+3+1] /= xStdBary[i*per_link]
        else:
            per_link = 10
            xStdBary = self.paramHelpers.paramsLink2Bary(new_params)

        tree = self.parseURDF(input_urdf)

        for l in tree.findall('link'):
            if l.attrib['name'] in self.model.linkNames:
                link_id = self.model.linkNames.index(l.attrib['name'])
                l.find('inertial/mass').attrib['value'] = '{}'.format(xStdBary[link_id*per_link])
                l.find('inertial/origin').attrib['xyz'] = '{} {} {}'.format(xStdBary[link_id*per_link+1],
                                                                            xStdBary[link_id*per_link+2],
                                                                            xStdBary[link_id*per_link+3])
                if not self.opt['identifyGravityParamsOnly']:
                    inert = l.find('inertial/inertia')
                    inert.attrib['ixx'] = '{}'.format(xStdBary[link_id*10+4])
                    inert.attrib['ixy'] = '{}'.format(xStdBary[link_id*10+5])
                    inert.attrib['ixz'] = '{}'.format(xStdBary[link_id*10+6])
                    inert.attrib['iyy'] = '{}'.format(xStdBary[link_id*10+7])
                    inert.attrib['iyz'] = '{}'.format(xStdBary[link_id*10+8])
                    inert.attrib['izz'] = '{}'.format(xStdBary[link_id*10+9])

        # write friction of joints
        for l in tree.findall('joint'):
            if l.attrib['name'] in self.model.jointNames:
                joint_id = self.model.jointNames.index(l.attrib['name'])
                if self.opt['identifyFriction']:
                    f_c = cast(float, xStdBary[self.model.num_links*per_link + joint_id])
                    if self.opt['identifyGravityParamsOnly']:
                        f_v = 0.0
                    else:
                        if self.opt['identifySymmetricVelFriction']:
                            f_v = cast(float, xStdBary[self.model.num_model_params + self.model.num_dofs + joint_id])
                        else:
                            print(Fore.RED + "Can't write velocity dependent friction to URDF as identified values are asymmetric. URDF only supports symmetric values." + Fore.RESET)
                            sys.exit(1)
                else:
                    # parameters were identified assuming there was no friction
                    f_c = f_v = 0.0
                l.find('dynamics').attrib['friction'] = '{}'.format(f_c)
                if not self.opt['identifyGravityParamsOnly']:
                    l.find('dynamics').attrib['damping'] = '{}'.format(f_v)

        tree.write(output_urdf, xml_declaration=True)

    def getLinkNames(self, input_urdf):
        # type: (str) -> List[str]

        links = []
        tree = self.parseURDF(input_urdf)
        for l in tree.findall('link'):
            if len(l.getchildren()) > 0:   # ignore fake links
                links.append(l.attrib['name'])
        return links

    def getMeshPath(self, input_urdf, link_name):
        # type: (AnyStr, AnyStr) -> AnyStr

        tree = self.parseURDF(input_urdf)
        link_found = False
        filepath = None  # type: AnyStr
        for l in tree.findall('link'):
            if l.attrib['name'] == link_name:
                link_found = True
                m = l.find('visual/geometry/mesh')
                if m is not None:
                    filepath = m.attrib['filename']
                    try:
                        self.mesh_scaling = m.attrib['scale']
                    except KeyError:
                        self.mesh_scaling = '1 1 1'

        if not link_found or m is None:
            #print(Fore.RED + "No mesh information specified for link '{}' in URDF! Using a very large box.".format(link_name) + Fore.RESET)
            filepath = None

        #if path is ros package path, get absolute system path
        if filepath and (filepath.startswith('package') or filepath.startswith('model')):
            try:
                import resource_retriever    #ros package
                r = resource_retriever.get(filepath)
                filepath = r.url.replace('file://', '')
                #r.read() #get file into memory
            except ImportError:
                #if no ros installed, try to get stl files from 'meshes' dir relative to urdf files
                filename = filepath.split('/')
                try:
                    filename = filename[filename.index(self.opt['meshBaseDir']):]
                    filepath = '/'.join(input_urdf.split('/')[:-1] + filename)
                except ValueError:
                    filepath = None

        return filepath

    def getLinkGeometry(self, input_urdf, link_name):
        # type: (str, str) -> Tuple[List[float], List[float], List[float]]

        tree = self.parseURDF(input_urdf)

        def getBoxAttribs(m, l):
            box_size = m.attrib['size']
            m = l.find('visual/origin')
            if m is not None:
                try:
                    box_pos = m.attrib['xyz']
                except:
                    box_pos = '0 0 0'

                try:
                    box_rpy = m.attrib['rpy']
                except:
                    box_rpy = '0 0 0'
            else:
                box_pos = box_rpy = '0 0 0'
            return box_size, box_pos, box_rpy

        box_size = box_pos = box_rpy = [0.0, 0.0, 0.0]
        for l in tree.findall('link'):
            if l.attrib['name'] == link_name:
                link_found = True
                m = l.find('visual/geometry/box')
                if m is not None:
                    box_size, box_pos, box_rpy = getBoxAttribs(m, l)
                else:
                    m = l.find('visual/collision/box')
                    if m is not None:
                        box_size, box_pos, box_rpy = getBoxAttribs(m, l)
                    else:
                        box_size = box_pos = box_rpy = '0 0 0'   # type: ignore
                box_size = [float(i) for i in box_size.split()]
                box_pos = [float(i) for i in box_pos.split()]
                box_rpy = [float(i) for i in box_rpy.split()]
                break
        return (box_size, box_pos, box_rpy)

    @staticmethod
    def getNeighbors(idyn_model, connected=True):
        # type: (iDynTree.Model, bool) -> Dict[str, Dict[str, List[int]]]

        # get neighbors for each link
        neighbors = {}   # type: Dict[str, Dict[str, List[int]]]
        for l in range(idyn_model.getNrOfLinks()):
            link_name = idyn_model.getLinkName(l)
            #if link_name not in self.model.linkNames:  # ignore links that are ignored in the generator
            #    continue
            neighbors[link_name] = {'links':[], 'joints':[]}
            num_neighbors = idyn_model.getNrOfNeighbors(l)
            for n in range(num_neighbors):
                nb = idyn_model.getNeighbor(l, n)
                neighbors[link_name]['links'].append(idyn_model.getLinkName(nb.neighborLink))
                neighbors[link_name]['joints'].append(idyn_model.getJointName(nb.neighborJoint))

        if connected:
            # for each neighbor link, add links connected via a fixed joint also as neighbors
            neighbors_tmp = neighbors.copy()  # don't modify in place so no recursive loops happen
            for l in range(idyn_model.getNrOfLinks()):
                link_name = idyn_model.getLinkName(l)
                for nb in neighbors_tmp[link_name]['links']:  # look at all neighbors of l
                    for j_name in neighbors_tmp[nb]['joints']:  # check each joint of a neighbor of l
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
                            if nb_fixed != l and nb_fixed_name not in neighbors[link_name]['links']:
                                neighbors[link_name]['links'].append(nb_fixed_name)

        return neighbors

    def getBoundingBox(self, input_urdf, old_com, link_name, scaling=True):
        # type: (str, List[float], str, bool) -> Tuple[List[List[float]], List[float], np._ArrayLike]
        ''' Return bounding box for one link derived from mesh file if possible.
            If no mesh file is found, a cube around the old COM is returned.
            Expects old_com in barycentric form! '''

        import trimesh
        filename = self.getMeshPath(input_urdf, link_name)

        # box around current COM in case no mesh is availabe
        length = self.opt['cubeSize']
        cube = [[-0.5*length+old_com[0], -0.5*length+old_com[1], -0.5*length+old_com[2]],
                [ 0.5*length+old_com[0],  0.5*length+old_com[1],  0.5*length+old_com[2]]]
        if scaling:
            hullScale = self.opt['hullScaling']
        else:
            hullScale = 1.0
        pos_0 = [0.0, 0.0, 0.0]
        rot_0 = np.identity(3)

        if filename and os.path.exists(filename):
            mesh = trimesh.load_mesh(filename)
            #TODO: get geometry origin attributes, rotate and shift mesh data

            #gazebo and urdf use 1m for 1 stl unit
            scale_x = float(self.mesh_scaling.split()[0])
            scale_y = float(self.mesh_scaling.split()[1])
            scale_z = float(self.mesh_scaling.split()[2])

            bounding_box = mesh.bounding_box.bounds * scale_x * hullScale

            # switch order of min/max if scaling is negative
            for s in range(0,3):
                if [scale_x, scale_y, scale_z][s] < 0:
                    bounding_box[0][s], bounding_box[1][s] = bounding_box[1][s], bounding_box[0][s]

            return bounding_box, pos_0, rot_0
        else:
            # use <visual><box> or <collision><box> if specified
            box, pos, rot = self.getLinkGeometry(input_urdf, link_name)
            if np.any(np.array(box) != 0):
                return [[-0.5*box[0]*hullScale, -0.5*box[1]*hullScale, -0.5*box[2]*hullScale],
                        [0.5*box[0]*hullScale,   0.5*box[1]*hullScale,  0.5*box[2]*hullScale]], pos, rot
            else:
                if self.opt['verbose']:
                    print(Fore.YELLOW + "Mesh file {} or box geometry not found for link '{}'! Using a {}m cube around a priori COM.".format(filename, link_name, length) + Fore.RESET)
                return cube, pos_0, rot_0

    @staticmethod
    def getJointLimits(input_urdf, use_deg=False):
        # type: (str, bool) -> Dict[str, Dict[str, float]]
        import xml.etree.ElementTree as ET
        tree = ET.parse(input_urdf)
        limits = {}    # type: Dict[str, Dict[str, float]]
        for j in tree.findall('joint'):
            name = j.attrib['name']
            torque = 0.0
            lower = 0.0
            upper = 0.0
            velocity = 0.0
            if j.attrib['type'] == 'revolute':
                l = j.find('limit')
                if l is not None:
                    torque = float(l.attrib['effort'])  #this is not really the physical limit but controller limit
                    lower = float(l.attrib['lower'])
                    upper = float(l.attrib['upper'])
                    velocity = float(l.attrib['velocity'])

                    limits[name] = {}
                    limits[name]['torque'] = float(torque)
                    if use_deg:
                        limits[name]['lower'] = np.rad2deg(lower)
                        limits[name]['upper'] = np.rad2deg(upper)
                        limits[name]['velocity'] = np.rad2deg(velocity)
                    else:
                        limits[name]['lower'] = lower
                        limits[name]['upper'] = upper
                        limits[name]['velocity'] = velocity
        return limits

    @staticmethod
    def getJointFriction(input_urdf):
        # type: (AnyStr) -> Dict[AnyStr, Dict[AnyStr, float]]
        ''' return friction values for each revolute joint from a urdf'''

        import xml.etree.ElementTree as ET
        tree = ET.parse(input_urdf)
        friction = {}  # type: Dict[AnyStr, Dict[AnyStr, float]]
        for j in tree.findall('joint'):
            name = j.attrib['name']
            constant = 0.0
            vel_dependent = 0.0
            if j.attrib['type'] == 'revolute':
                l = j.find('dynamics')
                if l is not None:
                    try:
                        constant = float(l.attrib['friction'])
                    except KeyError:
                        constant = 0

                    try:
                        vel_dependent = float(l.attrib['damping'])
                    except KeyError:
                        vel_dependent = 0

                friction[name] = {}
                friction[name]['f_constant'] = constant
                friction[name]['f_velocity'] = vel_dependent
        return friction
