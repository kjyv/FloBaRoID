#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import numpy.linalg as la
import math
import collections

import sys
import time

from OpenGL import GLU
from OpenGL.GL.shaders import compileShader, compileProgram

import pyglet
from pyglet import gl
from pyglet.window import key

# convert python list to gldouble array
def glvec(v):
    return (gl.GLdouble * len(v))(*v)

def glvecf(v):
    return (gl.GLfloat * len(v))(*v)

class Cube(object):
    ''' vertices for a cube of size 1 '''
    def __init__(self):
        self.vertices = np.array([-0.5,  0.5, 0.0,
                                  -0.5, -0.5, 0.0,
                                   0.5, -0.5, 0.0,
                                   0.5,  0.5, 0.0,
                                  -0.5,  0.5, -1.0,
                                  -0.5, -0.5, -1.0,
                                   0.5, -0.5, -1.0,
                                   0.5,  0.5, -1.0], np.float32)
        self.indices = np.array([0, 1, 2,
                                 0, 2, 3,
                                 0, 3, 7,
                                 0, 7, 4,
                                 0, 4, 5,
                                 0, 5, 1,
                                 3, 2, 6,
                                 3, 6, 7,
                                 1, 2, 5,
                                 2, 5, 6,
                                 5, 4, 7,
                                 7, 6, 5], np.ushort)

        # normals are unit vector from center to vertice
        #c = np.array([0,0,0.5])
        #self.normals = ((self.vertices.reshape((8,3)) - c) / np.sqrt(3)).flatten()
        self.normals = np.array([-0.28867513,  0.28867513, -0.28867513,
                                -0.28867513, -0.28867513, -0.28867513,
                                0.28867513, -0.28867513, -0.28867513,
                                0.28867513, 0.28867513, -0.28867513,
                                -0.28867513,  0.28867513,  0.28867513,
                               -0.28867513, -0.28867513,  0.28867513,
                               0.28867513, -0.28867513, 0.28867513,
                               0.28867513,  0.28867513,  0.28867513])


class Mesh(object):
    def __init__(self):
        pass


class FirstPersonCamera(object):
    DEFAULT_MOVEMENT_SPEED = 5.0
    DEFAULT_MOUSE_SENSITIVITY = 0.2
    DEFAULT_KEY_MAP = {
        'forward': key.W,
        'backward': key.S,
        'left': key.A,
        'right': key.D,
        'up': key.SPACE,
        'down': key.LSHIFT
    }

    class InputHandler(object):
        def __init__(self):
            self.pressed = collections.defaultdict(bool)
            self.dx = 0
            self.dy = 0

        def on_key_press(self, symbol, modifiers):
            self.pressed[symbol] = True

        def on_key_release(self, symbol, modifiers):
            self.pressed[symbol] = False

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self.dx = dx
                self.dy = dy

    def __init__(self, window, position=(0, 0, 0), pitch=-90.0, yaw=0.0, key_map=DEFAULT_KEY_MAP,
            movement_speed=DEFAULT_MOVEMENT_SPEED, mouse_sensitivity=DEFAULT_MOUSE_SENSITIVITY,
            y_inv=True):
        """Create camera object

        Arguments:
            window -- pyglet window which camera attach
            position -- position of camera
            key_map -- dict like FirstPersonCamera.DEFAULT_KEY_MAP
            movement_speed -- speed of camera move (scalar)
            mouse_sensitivity -- sensitivity of mouse (scalar)
            y_inv -- inversion turn above y-axis
        """

        self.__position = list(position)

        self.__yaw = yaw
        self.__pitch = pitch

        self.__input_handler = FirstPersonCamera.InputHandler()

        window.push_handlers(self.__input_handler)

        self.y_inv = y_inv
        self.key_map = key_map
        self.movement_speed = movement_speed
        self.mouse_sensitivity = mouse_sensitivity

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, value):
        # type: (List[float]) -> None
        self.__position = value

    @property
    def yaw(self):
        # type: () -> float
        return self.__yaw

    @yaw.setter
    def yaw(self, value):
        # type: (float) -> None
        """Turn above x-axis"""
        self.__yaw += value * self.mouse_sensitivity

    @property
    def pitch(self):
        # type: () -> float
        return self.__pitch

    @pitch.setter
    def pitch(self, value):
        # type: (float) -> None
        """Turn above y-axis"""
        self.__pitch += value * self.mouse_sensitivity * ((-1) if self.y_inv else 1)

    def move_forward(self, distance):
        """Move forward on distance"""
        self.__position[0] += distance * math.sin(math.radians(-self.__yaw))
        self.__position[1] -= distance * math.cos(math.radians(-self.__yaw))

    def move_backward(self, distance):
        """Move backward on distance"""
        self.__position[0] -= distance * math.sin(math.radians(-self.__yaw))
        self.__position[1] += distance * math.cos(math.radians(-self.__yaw))

    def move_left(self, distance):
        """Move left on distance"""
        self.__position[0] -= distance * math.sin(math.radians(-self.__yaw - 90))
        self.__position[1] += distance * math.cos(math.radians(-self.__yaw - 90))

    def move_right(self, distance):
        """Move right on distance"""
        self.__position[0] -= distance * math.sin(math.radians(-self.__yaw + 90))
        self.__position[1] += distance * math.cos(math.radians(-self.__yaw + 90))

    def move_up(self, distance):
        """Move up on distance"""
        self.__position[2] -= distance

    def move_down(self, distance):
        """Move down on distance"""
        self.__position[2] += distance

    def update(self, delta_time):
        """Update camera state"""
        self.yaw = self.__input_handler.dx
        self.__input_handler.dx = 0

        self.pitch = self.__input_handler.dy
        self.__input_handler.dy = 0

        if self.__input_handler.pressed[self.key_map['forward']]:
            self.move_forward(delta_time * self.movement_speed)

        if self.__input_handler.pressed[self.key_map['backward']]:
            self.move_backward(delta_time * self.movement_speed)

        if self.__input_handler.pressed[self.key_map['left']]:
            self.move_left(delta_time * self.movement_speed)

        if self.__input_handler.pressed[self.key_map['right']]:
            self.move_right(delta_time * self.movement_speed)

        if self.__input_handler.pressed[self.key_map['up']]:
            self.move_up(delta_time * self.movement_speed)

        if self.__input_handler.pressed[self.key_map['down']]:
            self.move_down(delta_time * self.movement_speed)

    def draw(self):
        """Apply transform"""
        pyglet.gl.glRotatef(self.__pitch, 1.0, 0.0, 0.0)
        pyglet.gl.glRotatef(self.__yaw, 0.0, 0.0, 1.0)
        pyglet.gl.glTranslatef(*self.__position)


class Visualizer(object):
    def __init__(self):
        # some vars
        #self.pressed_keys = []   # type: List[Any]
        self.program = None   # type: List[Any]
        self.window_closed = False
        self.mode = 'b'  # 'b' - blocking or 'c' - continous

        # keep a list of bodies
        self.bodies = []     # type: List[Dict[str, Any]]

        self._initWindow()
        self._initCamera()
        self._initGL()

        legend = '''<font face="Helvetica,Arial" size=15>wasd - move around <br>
        mouse drag - look <br>
        c - continous/blocking <br>
        b - add cube <br>
        q - close <br>
        </font>'''
        self.label = pyglet.text.HTMLLabel(legend,
                          x = 10, y = -10,
                          width = 200,
                          multiline = True,
                          anchor_x='left', anchor_y='bottom')

    def update(self, dt=None):
        self.camera.update(dt)

    def _initWindow(self):
        x = 100
        y = 100
        self.width = 800
        self.height = 600
        config = gl.Config(double_buffer=True, depth_size=24)
        self.window = pyglet.window.Window(self.width, self.height, resizable=True, visible=False, config=config)
        self.window_closed = False
        self.window.set_minimum_size(320, 200)
        self.window.set_location(x, y)
        self.window.set_caption('Model Visualization')
        self.window.on_draw = self.on_draw
        self.window.on_resize = self.on_resize
        self.window.on_key_press = self.on_key_press
        self.window.on_key_release = self.on_key_release
        self.window.on_close = self.on_close
        self.on_resize(self.width, self.height)

    def _initCamera(self):
        if 'camera' in self.__dict__:
            pos = self.camera.position
            pitch = self.camera.pitch
            yaw = self.camera.yaw
        else:
            pos = [3.577, 1.683, -0.860]
            pitch = -72.5
            yaw = 60
        self.camera = FirstPersonCamera(self.window, position=pos, pitch=pitch, yaw=yaw)
        self.fps = 50
        pyglet.clock.unschedule(self.update)
        pyglet.clock.schedule_interval(self.update, 1/self.fps)

    def _initGL(self):
        gl.glClearColor(0.8,0.8,0.9,0)
        gl.glClearDepth(1.0)                       # Enables Clearing Of The Depth Buffer
        gl.glDepthFunc(gl.GL_LESS)                 # The Type Of Depth Test To Do
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)   # make stuff look nice
        gl.glEnable(gl.GL_DEPTH_TEST)              # Enables Depth Testing
        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)   # Wireframe

        if not gl.glUseProgram:
            print( 'Missing Shader Objects!' )
            sys.exit(1)

        self.program = compileProgram(
            compileShader('''
                varying vec3 N;
                varying vec3 v;
                void main(void)
                {
                   v = vec3(gl_ModelViewMatrix * gl_Vertex);
                   N = normalize(gl_NormalMatrix * gl_Normal);
                   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                }

            ''', gl.GL_VERTEX_SHADER),
            compileShader('''
                varying vec3 N;
                varying vec3 v;
                void main (void)
                {
                   vec3 L = normalize(gl_LightSource[0].position.xyz - v);
                   vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0)
                   vec3 R = normalize(-reflect(L,N));

                   //calculate Ambient Term:
                   vec4 Iamb = gl_FrontLightProduct[0].ambient;

                   //calculate Diffuse Term:
                   vec4 Idiff = gl_FrontLightProduct[0].diffuse * max(dot(N,L), 0.0);
                   Idiff = clamp(Idiff, 0.0, 1.0);

                   // calculate Specular Term:
                   vec4 Ispec = gl_FrontLightProduct[0].specular
                                * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);
                   Ispec = clamp(Ispec, 0.0, 1.0);

                   // write Total Color:
                   //vec4 amb = vec4(Iamb[0]*1.0, Iamb[1]*0.5, Iamb[2]*0.5, Iamb[3]*1.0);
                   gl_FragColor = gl_FrontLightModelProduct.sceneColor + Iamb + Idiff + Ispec;
                }
            ''', gl.GL_FRAGMENT_SHADER),)

        mat_emission = [0.3, 0.3, 0.4, 1.0]
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, glvecf(mat_emission))
        mat_diffuse = [0.7, 0.5, 0.5, 1.0]
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, glvecf(mat_diffuse))

        cube = Cube()
        self.cube_list = pyglet.graphics.vertex_list_indexed(len(cube.vertices)//3, cube.indices,
                                                             ('v3f', cube.vertices), ('n3f', cube.normals))

    def init_ortho(self):
        # disable shaders
        gl.glUseProgram(0)

        # store the projection matrix to restore later
        gl.glMatrixMode(gl.GL_PROJECTION)

        # load orthographic projection matrix
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, 0, self.height, -1, 500)

        # reset modelview
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def init_perspective(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Init Projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        GLU.gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 100.0)
        # Initialize ModelView matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def on_close(self):
        self.window_closed = True
        self.window.close()
        self.window = None
        pyglet.app.exit()

    def on_key_press(self, symbol, modifiers):
        #print("Key pressed: {}".format(c))
        if symbol in [key.Q, key.ESCAPE]:
            print('leaving render')
            pyglet.app.exit()

        if symbol == key.B:
            self.addBody()

        if symbol == key.C:
            if self.mode == 'b':
                print('switching to continuous render')
                self.mode = 'c'
                pyglet.app.exit()
            else:
                print('switching to blocking render')
                self.mode = 'b'

        if symbol == key.I:
            print("Camera pos:{} pitch:{} yaw:{}".format(
                self.camera.position, self.camera.pitch, self.camera.yaw))

        if symbol == key.R:
            print("Reset camera")
            self._initCamera()

        '''
        if symbol in self.pressed_keys:
            return

        # remember pressed keys until released
        self.pressed_keys.append(symbol)
        '''

        return pyglet.event.EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        #if symbol in self.pressed_keys:
        #    self.pressed_keys.remove(symbol)
        pass

    def on_draw(self):
        self.init_perspective()
        # Redraw the scene
        gl.glClearColor(0.8,0.8,0.9,0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
        gl.glLoadIdentity()

        self.camera.draw()

        # run shaders
        if self.program:
            gl.glUseProgram(self.program)

        self.drawGrid()
        for b in self.bodies:
            self.drawBody(b)

        self.init_ortho()
        self.label.draw()

    def on_resize(self, width, height):
        """(Re-)Init drawing.
        """
        # Viewport
        gl.glViewport(0,0, width, height)
        self.width = width
        self.height = height
        self.init_perspective()

        return pyglet.event.EVENT_HANDLED


    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def addBody(self):
        print("Adding box to world")
        body = {}  # type: Dict[str, Any]
        body['geometry'] = 'box'
        body['boxsize'] = np.array([1.0, 1.0, 1.0])
        body['position'] = np.array([1.0, 1.0, 1.0])*np.random.rand(3)
        body['rotation'] = np.identity(3)
        self.bodies.append(body)
        print("Bodies: {}".format(self.bodies))

    def drawGrid(self):
        # dx, dy are thickness parameters of grid
        xmin = -50.0
        xmax = 50.0
        dx = 5.0
        ymin = -50.0
        ymax = 50.0
        dy = 5.0

        #TODO: put in vertices list
        gl.glBegin(gl.GL_LINES)
        for x in np.arange(xmin, xmax, dx):
            for y in np.arange(ymin, ymax, dy):
                gl.glVertex3f(x, ymin, 0.0)
                gl.glVertex3f(x, ymax, 0.0)
                gl.glVertex3f(xmin, y, 0.0)
                gl.glVertex3f(xmax, y, 0.0)
        gl.glEnd()

    def drawCube(self):
        #gl.glEnableVertexAttribArray(0)
        #gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, cube.vertices)
        #gl.glDrawElements(gl.GL_TRIANGLES, cube.indices)
        self.cube_list.draw(gl.GL_TRIANGLES)

    def drawBody(self, body):
        """Draw a body"""

        pos = body['position']
        R = body['rotation']
        r,p,y = self.rotationMatrixToEulerAngles(R)
        R = self.eulerAnglesToRotationMatrix([r,p,y])

        # homogenous transform
        trans = [R[0,0], R[0,1], R[0,2], 0.,
                 R[1,0], R[1,1], R[1,2], 0.,
                 R[2,0], R[2,1], R[2,2], 0.,
                 0,      0,      0,      1.0]

        gl.glPushMatrix()
        gl.glTranslatef(-pos[0], -pos[1], pos[2])
        gl.glRotatef(np.rad2deg(y), 0.0, 0.0, 1.0)
        gl.glRotatef(np.rad2deg(-r), 1.0, 0.0, 0.0)
        gl.glRotatef(np.rad2deg(-p), 0.0, 1.0, 0.0)
        #gl.glMultMatrixd(glvec(trans))
        if body['geometry'] is 'box':
            dim = body['boxsize']
            gl.glScalef(dim[0], dim[1], dim[2])
            self.drawCube()
        gl.glPopMatrix()

    def addIDynTreeModel(self,
                  model,          # type: iDynTree.DynamicsComputations
                  boxes,          # type: Dict                      # link hulls
                  real_links,     # type: List[str]                 # all the links that are not fake
                  ignore_links    # type: List[str]                 # links that will not be drawn
                  ):
        # type: (...) -> None
        #
        if self.window_closed:
            self._initWindow()
            self._initCamera()

        self.bodies = []
        for l in range(model.getNrOfLinks()):
            n_name = model.getFrameName(l)
            if n_name in ignore_links:
                continue
            if n_name not in real_links:
                continue
            body = {}  # type: Dict[str, Any]
            body['geometry'] = 'box'
            try:
                b = boxes[n_name]
                body['boxsize'] = np.array([b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0]])
            except KeyError:
                print('using cube for {}'.format(n_name))
                body['boxsize'] = np.array([0.1, 0.1, 0.1])
            t = model.getWorldTransform(l)
            body['position'] = t.getPosition().toNumPy()
            body['rotation'] = t.getRotation().toNumPy()
            self.bodies.append(body)

    def stop(self, dt):
        pyglet.app.exit()

    def run(self):
        self.window.set_visible()
        #from IPython import embed
        #embed()
        if self.mode == 'b':
            pyglet.app.run()
        else:
            #run one loop iterration only (draw one frame)
            pyglet.clock.tick()
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                window.flip()

        if self.mode == 'c':
            pyglet.clock.schedule_once(self.stop, 1/self.fps)


if __name__ == '__main__':

    import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile('model/walkman_measured.urdf')
    world_gravity = iDynTree.SpatialAcc.fromList([0,0,-9.81,0,0,0])

    # walkman
    link_cuboid_hulls = {
        'DWL':
        np.array([[-0.11982614, 0.06288684], [-0.11, 0.121],
                  [-0.10879193, 0.06298846]]),
        'DWS':
        np.array([[-0.07349821, 0.0735], [-0.0937, 0.0885],
                  [-0.07188258, 0.2072]]),
        'DWYTorso':
        np.array([[-0.10220008, 0.1113662], [-0.26770911, 0.2678313],
                  [-0.14542062, 0.07580003]]),
        'LElb':
        np.array([[-0.134, 0.05238984], [-0.096, 0.09307509],
                  [-0.087, 0.05239615]]),
        'LFoot':
        np.array([[-0.155, 0.165], [-0.067, 0.087], [-0.1435, 0.058]]),
        'LFootmot':
        np.array([[-0.125, 0.055], [-0.102, 0.072], [-0.069, 0.069]]),
        'LForearm':
        np.array([[-0.04249971, 0.04249971], [-0.0585, 0.0585],
                  [-0.03929598, 0.1215]]),
        'LHipMot':
        np.array([[-2.36900101e-01, 1.63424829e-16],
                  [-4.74997139e-02, 1.83532211e-01],
                  [-1.46772430e-01, 4.74721031e-02]]),
        'LLowLeg':
        np.array([[-0.11836108, 0.12498866], [-0.102, 0.127],
                  [-0.43789868, 0.09436713]]),
        'LShp':
        np.array([[-0.105, 0.104], [-0.05777887, 0.16954893],
                  [-0.06331461, 0.08662518]]),
        'LShr':
        np.array([[-0.0905, 0.0832], [-0.06499635, 0.06499484],
                  [-0.22170009, 0.06495815]]),
        'LShy':
        np.array([[-0.06052907, 0.09845684], [-0.08919501, 0.08377501],
                  [-0.21249971, 0.008]]),
        'LSoftHandLink':
        np.array([[-0.0980275, 0.1958], [-0.05189596, 0.0325],
                  [-0.34175, -0.0675]]),
        'LThighLowLeg':
        np.array([[-0.08745305, 0.11749961], [-0.1025, 0.127],
                  [-0.44349982, 0.06499506]]),
        'LThighUpLeg':
        np.array([[-0.082, 0.0655], [-0.086, 0.095], [-0.0655, 0.07949912]]),
        'LWrMot2':
        np.array([[-0.05528355, 0.0549846], [-0.05776, 0.0595],
                  [-0.13198608, 0.02437494]]),
        'LWrMot3':
        np.array([[-0.0542625, 0.063], [-0.02866348, 0.02866348],
                  [-0.0695, 0.025]]),
        'NeckPitch':
        np.array([[-0.06515144, 0.0225], [-0.07, 0.07], [-0.0225, 0.0985]]),
        'NeckYaw':
        np.array([[-0.0295, 0.0295], [-0.03449, 0.0295], [0.08, 0.142]]),
        'RElb':
        np.array([[-0.134, 0.05238984], [-0.09307509, 0.096],
                  [-0.087, 0.05239615]]),
        'RFoot':
        np.array([[-0.155, 0.165], [-0.087, 0.067], [-0.1435, 0.058]]),
        'RFootmot':
        np.array([[-0.125, 0.055], [-0.072, 0.102], [-0.069, 0.069]]),
        'RForearm':
        np.array([[-0.04249971, 0.04249971], [-0.0585, 0.0585],
                  [-0.03929598, 0.1215]]),
        'RHipMot':
        np.array([[-2.36900101e-01, 1.63424829e-16],
                  [-1.83532211e-01, 4.74997139e-02],
                  [-1.46772430e-01, 4.74721031e-02]]),
        'RLowLeg':
        np.array([[-0.11836108, 0.12498866], [-0.127, 0.102],
                  [-0.43789868, 0.09436713]]),
        'RShp':
        np.array([[-0.105, 0.104], [-0.16954893, 0.05777887],
                  [-0.06331461, 0.08662518]]),
        'RShr':
        np.array([[-0.0905, 0.0832], [-0.06499484, 0.06499635],
                  [-0.22170009, 0.06495815]]),
        'RShy':
        np.array([[-0.06052907, 0.09845684], [-0.08377501, 0.08919501],
                  [-0.21249971, 0.008]]),
        'RSoftHandLink':
        np.array([[-0.0980275, 0.1958], [-0.0325, 0.05189596],
                  [-0.34175, -0.0675]]),
        'RThighLowLeg':
        np.array([[-0.08745305, 0.11749961], [-0.127, 0.1025],
                  [-0.44349982, 0.06499506]]),
        'RThighUpLeg':
        np.array([[-0.082, 0.0655], [-0.095, 0.086], [-0.0655, 0.07949912]]),
        'RWrMot2':
        np.array([[-0.05528355, 0.0549846], [-0.0595, 0.05776],
                  [-0.13198608, 0.02437494]]),
        'RWrMot3':
        np.array([[-0.0542625, 0.063], [-0.02866348, 0.02866348],
                  [-0.0695, 0.025]]),
        'TorsoProtections':
        np.array([[-0.355, 0.25098981], [-0.21364139, 0.21364139],
                  [-0.1705, 0.3865]]),
        'Waist': np.array([[-0.23678584, 0.19874428], [-0.12499924, 0.12499924],
                           [-0.11371131, 0.20991649]]),
        'backpack': np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.165, 0.035]]),
        'imu_link': np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]),
        'imu_link2': np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]),
        'multisense/head':
        np.array([[-0.22704816, 0.03310263], [-0.09927974, 0.09928905],
                  [-0.06626804, 0.16240472]]),
        'multisense/head_imu_link':
        np.array([[-0.222993, -0.022993], [-0.0649666, 0.1350334],
                  [-0.07226, 0.12774]]),
        'multisense/hokuyo_link':
        np.array([[-0.03388125, 0.03853858], [-0.04238125, 0.05379988],
                  [-0.02115, 0.05885]])
    }

    link_names = ['Waist', 'LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot',
                'RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot', 'DWL', 'DWS',
                'DWYTorso', 'LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3',
                'LSoftHandLink', 'NeckYaw', 'NeckPitch', 'multisense/head', 'multisense/head_imu_link',
                'multisense/hokuyo_link', 'RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2',
                'RWrMot3', 'RSoftHandLink', 'backpack', 'TorsoProtections', 'imu_link', 'imu_link2']

    v = Visualizer()
    n_dof = 28
    #q0 = [0.0]*n_dof
    q0 = np.deg2rad([0., 0, -70., 90., -20.,   0.,
          0.,   0., -70.,  90., -20.,   0.,
          0.,   0.,
          0.,  10.,   0.,   0.,   0.,   0.,   0.,
          0., -10.,   0.,   0.,   0.,   0.,   0.])
    dq = iDynTree.VectorDynSize.fromList([0.0]*n_dof)

    ignore_links = ['imu_link', 'imu_link2', 'multisense/head_imu_link', 'TorsoProtections']
    for n in np.arange(0, np.pi, 0.01):
        q0[15] = n
        q = iDynTree.VectorDynSize.fromList(q0)
        dynComp.setRobotState(q, dq, dq, world_gravity)
        v.addIDynTreeModel(dynComp, link_cuboid_hulls, link_names, ignore_links)
        v.run()
