import time
from IPython import embed
import iDynTree
import numpy as np

import colorama
from colorama import Fore, Back, Style

class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class ParamHelpers(object):
    def __init__(self, n_params):
        self.n_params = n_params

    def checkPhysicalConsistency(self, params):
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite, triangle inequaltiy for eigenvalues of inertia tensor expressed at COM)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link
        """
        cons = {}
        for i in range(0, self.n_params):
            if (i % 10 == 0):   #for each link
                p_vec = iDynTree.Vector10()
                for j in range(0, 10):
                    p_vec.setVal(j, params[i+j])
                si = iDynTree.SpatialInertia()
                si.fromVector(p_vec)
                cons[i / 10] = si.isPhysicallyConsistent()
        return cons

    def checkPhysicalConsistencyNoTriangle(self, params):
        """
        check params for physical consistency
        (mass positive, inertia tensor positive definite)

        expect params relative to link frame
        returns dictionary of link ids and boolean consistency for each link
        """
        cons = {}
        tensors = self.inertiaTensorFromParams(params)
        for i in range(0, self.n_params):
            if (i % 10 == 0):
                if params[i] <= 0:  #masses need to be positive
                    cons[i / 10] = False
                    continue
                #check if inertia tensor is positive definite (only then cholesky decomp exists)
                try:
                    np.linalg.cholesky(tensors[i / 10])
                    cons[i / 10] = True
                except np.linalg.linalg.LinAlgError:
                    cons[i / 10] = False
        return cons

    def isPhysicalConsistent(self, params):
        """give boolean consistency statement for a set of parameters"""
        return not (False in self.checkPhysicalConsistencyNoTriangle(params).values())

    def inertiaTensorFromParams(self, params):
        """take a parameter vector and return list of full inertia tensors (one for each link)"""
        tensors = list()
        for i in range(self.n_params):
            if (i % 10 == 0):
                inertia = np.zeros((3,3))
                #xx of inertia matrix
                value = params[i+4]
                inertia[0, 0] = value
                #xy
                value = params[i+5]
                inertia[0, 1] = value
                inertia[1, 0] = value
                #xz
                value = params[i+6]
                inertia[0, 2] = value
                inertia[2, 0] = value
                #yy
                value = params[i+7]
                inertia[1, 1] = value
                #yz
                value = params[i+8]
                inertia[1, 2] = value
                inertia[2, 1] = value
                #zz
                value = params[i+9]
                inertia[2, 2] = value
                tensors.append(inertia)
        return tensors

    def inertiaParams2RotationalInertiaRaw(self, params):
        #take values from inertia parameter vector and create iDynTree RotationalInertiaRaw matrix
        #expects six parameter vector

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
        ## convert params from iDynTree values (relative to link frame) to values usable in URDF (barycentric)
        ## (params are changed in place)

        #mass is mass
        #com in idyntree is represented as first moment of mass, so com * mass. URDF uses com
        #inertia in idyntree is represented w.r.t. frame origin. URDF uses w.r.t com
        params = params.copy()
        for i in range(0, self.n_params):
            if (i % 10 == 0):   #for each link
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
        params = params.copy()
        for i in range(0, self.n_params):
            if (i % 10 == 0):   #for each link
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

class URDFHelpers(object):
    def __init__(self, paramHelpers, link_names):
        self.n_params = paramHelpers.n_params
        self.paramHelpers = paramHelpers
        self.link_names = link_names

    def replaceParamsInURDF(self, input_urdf, output_urdf, new_params, link_names):
        """ set new inertia parameters from params and urdf_file, write to new temp file """

        xStdBary = self.paramHelpers.paramsLink2Bary(new_params)

        import xml.etree.ElementTree as ET
        tree = ET.parse(input_urdf)
        for l in tree.findall('link'):
            if l.attrib['name'] in link_names:
                link_id = link_names.index(l.attrib['name'])
                l.find('inertial/mass').attrib['value'] = str(xStdBary[link_id*10])
                l.find('inertial/origin').attrib['xyz'] = '{} {} {}'.format(xStdBary[link_id*10+1],
                                                                            xStdBary[link_id*10+2],
                                                                            xStdBary[link_id*10+3])
                inert = l.find('inertial/inertia')
                inert.attrib['ixx'] = str(xStdBary[link_id*10+4])
                inert.attrib['ixy'] = str(xStdBary[link_id*10+5])
                inert.attrib['ixz'] = str(xStdBary[link_id*10+6])
                inert.attrib['iyy'] = str(xStdBary[link_id*10+7])
                inert.attrib['iyz'] = str(xStdBary[link_id*10+8])
                inert.attrib['izz'] = str(xStdBary[link_id*10+9])

        tree.write(output_urdf, xml_declaration=True)

    def getMeshPath(self, input_urdf, link_name):
        import xml.etree.ElementTree as ET
        tree = ET.parse(input_urdf)

        link_found = False
        filepath = None
        for l in tree.findall('link'):
            if l.attrib['name'] == link_name:
                link_found = True
                m = l.find('visual/geometry/mesh')
                if m != None:
                    filepath = m.attrib['filename']
                    try:
                        self.mesh_scaling = m.attrib['scale']
                    except KeyError:
                        self.mesh_scaling = '1 1 1'

        if not link_found or m is None:
            print Fore.LIGHTRED_EX + "No mesh information specified for {} in URDF! Using approximation.".format(link_name) + Fore.RESET
            filepath = None
        else:
            if not filepath.lower().endswith('stl'):
                raise ValueError("Can't open other than STL files.")
                # TODO: could use pycollada for *.dae

        #if path is ros package path, get absolute system path
        if filepath and (filepath.startswith('package') or filepath.startswith('model')):
            try:
                import resource_retriever    #ros package
                r = resource_retriever.get(filepath)
                filepath = r.url.replace('file://', '')
                #r.read() #get file into memory
            except ImportError:
                #if no ros installed, try to get stl files from 'meshes' dir relative to urdf files
                filename = filepath.split('/')[-1]
                filepath = '/'.join(input_urdf.split('/')[:-1] + ['meshes'] + [filename])

        return filepath

    def getBoundingBox(self, input_urdf, link_nr, scale=1):
        from stl import mesh   #using numpy-stl

        filename = self.getMeshPath(input_urdf, self.link_names[link_nr])
        if filename:
            stl_mesh = mesh.Mesh.from_file(filename)

            #gazebo and urdf use 1m for 1 stl unit
            scale_x = float(self.mesh_scaling.split()[0])
            scale_y = float(self.mesh_scaling.split()[1])
            scale_z = float(self.mesh_scaling.split()[2])
            return [[stl_mesh.x.min()*scale_x*scale, stl_mesh.x.max()*scale_x*scale],
                    [stl_mesh.y.min()*scale_y*scale, stl_mesh.y.max()*scale_y*scale],
                    [stl_mesh.z.min()*scale_z*scale, stl_mesh.z.max()*scale_z*scale]]
        else:
            return [[100,100], [100,100], [100,100]]

            #TODO: in case we have no stl files:
            # take length of link (distance to next link, last one?) and assume width and height from it (proper
            # geometry is not known without CAD geometry)
            # w = h = l/2 ?
            # length: go through all links, getFrameIndex for each (dynamicsComputations) and then
            # of its parent, get position from Model::getFrameTransform

    @classmethod
    def getJointLimits(self, input_urdf, use_deg=True):
        import xml.etree.ElementTree as ET
        tree = ET.parse(input_urdf)
        limits = {}
        for j in tree.findall('joint'):
            name = j.attrib['name']
            torque = 0
            lower = 0
            upper = 0
            velocity = 0
            if j.attrib['type'] == 'revolute':
                l = j.find('limit')
                if l != None:
                    torque = l.attrib['effort']   #this is not really the physical limit but controller limit, but well
                    lower = l.attrib['lower']
                    upper = l.attrib['upper']
                    velocity = l.attrib['velocity']

                    limits[name] = {}
                    limits[name]['torque'] = float(torque)
                    if use_deg:
                        limits[name]['lower'] = np.rad2deg(float(lower))
                        limits[name]['upper'] = np.rad2deg(float(upper))
                        limits[name]['velocity'] = np.rad2deg(float(velocity))
                    else:
                        limits[name]['lower'] = float(lower)
                        limits[name]['upper'] = float(upper)
                        limits[name]['velocity'] = float(velocity)
        return limits

