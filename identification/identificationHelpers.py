import iDynTree

class IdentificationHelpers():
    def __init__(self, n_params):
        self.n_params = n_params

    def paramsFromiDyn2URDF(self, params):
        ## convert params from iDynTree values to values directly usable in URDF (mass, com, inertia)
        ## (params is changed in place)

        #mass is mass
        #com in idyntree is represented as first moment of mass, so com * mass. URDF uses com
        #inertia in idyntree is represented w.r.t. frame origin. URDF uses w.r.t com

        for i in range(0, self.n_params):
            if (i % 10 == 0):   #for each joint
                link_mass = params[i]

                #com
                com_x = params[i+1]
                com_y = params[i+2]
                com_z = params[i+3]
                params[i+1] = com_x / link_mass  #x of first moment -> x of com
                params[i+2] = com_y / link_mass  #y of first moment -> y of com
                params[i+3] = com_z / link_mass  #z of first moment -> z of com
                p_com = iDynTree.PositionRaw(params[i+1], params[i+2], params[i+3])

                #inertias
                rot_inertia_origin = iDynTree.RotationalInertiaRaw()
                #xx of inertia matrix w.r.t. link origin
                value = params[i+4]
                rot_inertia_origin.setVal(0, 0, value)
                #xy
                value = params[i+5]
                rot_inertia_origin.setVal(0, 1, value)
                rot_inertia_origin.setVal(1, 0, value)
                #xz
                value = params[i+6]
                rot_inertia_origin.setVal(0, 2, value)
                rot_inertia_origin.setVal(2, 0, value)
                #yy
                value = params[i+7]
                rot_inertia_origin.setVal(1, 1, value)
                #yz
                value = params[i+8]
                rot_inertia_origin.setVal(1, 2, value)
                #zz
                value = params[i+9]
                rot_inertia_origin.setVal(2, 2, value)

                s_inertia = iDynTree.SpatialInertiaRaw(link_mass, p_com, rot_inertia_origin)
                #s_inertia.fromRotationalInertiaWrtCenterOfMass(link_mass, p_com, rot_inertia)
                rot_inertia_com = s_inertia.getRotationalInertiaWrtCenterOfMass()
                params[i+4] = rot_inertia_com.getVal(0, 0)    #xx w.r.t. com
                params[i+5] = rot_inertia_com.getVal(0, 1)    #xy w.r.t. com
                params[i+6] = rot_inertia_com.getVal(0, 2)    #xz w.r.t. com
                params[i+7] = rot_inertia_com.getVal(1, 1)    #xx w.r.t. com
                params[i+8] = rot_inertia_com.getVal(1, 2)    #xy w.r.t. com
                params[i+9] = rot_inertia_com.getVal(2, 2)    #xz w.r.t. com

    def paramsFromURDF2iDyn(self, params):
        return
