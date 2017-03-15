#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import Tuple, List, Dict, Callable, Any

import numpy as np
import math
import collections

import os, sys

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
        self.vertices = np.array([-0.5,  0.5, 0.5,
                                  -0.5, -0.5, 0.5,
                                   0.5, -0.5, 0.5,
                                   0.5,  0.5, 0.5,
                                  -0.5,  0.5, -0.5,
                                  -0.5, -0.5, -0.5,
                                   0.5, -0.5, -0.5,
                                   0.5,  0.5, -0.5], np.float32)
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
    def __init__(self, config):
        # type: (Dict[str]) -> None
        # some vars
        #self.pressed_keys = []   # type: List[Any]
        self.default_shader = None   # type: List[Any]
        self.window_closed = False
        self.mode = 'b'  # 'b' - blocking or 'c' - continous
        self.display_index = 0   # current index for displaying e.g. postures from file
        self.display_max = 1
        self.config = config

        # keep a list of bodies
        self.bodies = []     # type: List[Dict[str, Any]]

        # additional callback to be used with key handling
        self.event_callback = None  # type: Callable

        self._initWindow()
        self._initCamera()
        self._initGL()

        legend = '''<font face="Helvetica,Arial" size=15>wasd &#8679; &#x2423; - move around <br>
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
        #gl.glEnable(gl.GL_LIGHTING)
        #gl.glEnable(gl.GL_LIGHT0)
        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)   # Wireframe

        if not gl.glUseProgram:
            print("Can't run shaders!")
            sys.exit(1)

        self.default_shader = compileProgram(
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
            return pyglet.event.EVENT_HANDLED

        if symbol == key.B:
            self.addBody()

        if symbol == key.C:
            if self.mode == 'b':
                print('switching to continuous render')
                self.mode = 'c'
                pyglet.app.exit()
                return pyglet.event.EVENT_HANDLED
            else:
                print('switching to blocking render')
                self.mode = 'b'

        if symbol == key.I:
            print("Camera pos:{} pitch:{} yaw:{}".format(
                self.camera.position, self.camera.pitch, self.camera.yaw))

        if symbol == key.R:
            print("Reset camera")
            self._initCamera()

        if symbol == key.RIGHT:
            if self.display_index < self.display_max - 1:
                self.display_index +=1
                if self.event_callback:
                    self.event_callback()

        if symbol == key.LEFT:
            if self.display_index > 0:
                self.display_index -=1
                if self.event_callback:
                    self.event_callback()

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
        if self.default_shader:
            gl.glUseProgram(self.default_shader)

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
        body['rotation'] = np.array([1.0, 1.0, 1.0])*np.random.rand(3)
        self.bodies.append(body)
        print("Bodies: {}".format(self.bodies))

    def drawCoords(self):
        l = 0.2
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1.0, 0.0, 0.0);
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(l, 0.0, 0.0)

        gl.glColor3f(0.0, 1.0, 0.0);
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, l, 0.0)

        gl.glColor3f(0.0, 0.0, 1.0);
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, l)
        gl.glEnd()

    def drawGrid(self):
        # dx, dy are the width of grid
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
        rpy = body['rotation']
        r,p,y = rpy[0], rpy[1], rpy[2]
        #R = self.eulerAnglesToRotationMatrix([r,p,y])

        '''
        # homogenous transform
        trans = [R[0,0], R[0,1], R[0,2], 0.,
                 R[1,0], R[1,1], R[1,2], 0.,
                 R[2,0], R[2,1], R[2,2], 0.,
                 pos[0], pos[1], pos[2], 1.0]
        '''
        gl.glPushMatrix()
        gl.glTranslatef(-pos[0], -pos[1], pos[2])
        gl.glRotatef(np.rad2deg(y), 0.0, 0.0, 1.0)
        gl.glRotatef(np.rad2deg(-r), 1.0, 0.0, 0.0)
        gl.glRotatef(np.rad2deg(-p), 0.0, 1.0, 0.0)

        #gl.glMultMatrixd(glvec(trans))

        self.drawCoords()

        rel_pos = body['center']
        gl.glTranslatef(rel_pos[0], rel_pos[1], rel_pos[2])

        transparent = 'transparent' in body and body['transparent']
        if body['geometry'] is 'box':
            dim = body['boxsize']
            gl.glScalef(dim[0], dim[1], dim[2])
            if transparent:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)   # Wireframe
                #gl.glEnable(gl.GL_BLEND)
                #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            self.drawCube()
            if transparent:
                #gl.glDisable(gl.GL_BLEND)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glPopMatrix()

    def addWorld(self, boxes):
        # type: (Dict) -> None
        for linkName in boxes:
            body = {}  # type: Dict[str, Any]
            body['geometry'] = 'box'
            b = np.array(boxes[linkName][0])
            body['boxsize'] = np.array([b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0]])
            body['center'] = 0.5*np.array([np.abs(b[0][1])-np.abs(b[0][0]),
                                           np.abs(b[1][1])-np.abs(b[1][0]),
                                           np.abs(b[2][1])-np.abs(b[2][0])])
            body['position'] = boxes[linkName][1]
            body['rotation'] = boxes[linkName][2]
            self.bodies.append(body)


    def addIDynTreeModel(self,
                  model,          # type: iDynTree.DynamicsComputations
                  boxes,          # type: Dict[str, List]     # link hulls
                  real_links,     # type: List[str]           # all the links that are not fake
                  ignore_links    # type: List[str]           # links that will not be drawn
                  ):
        # type: (...) -> None
        ''' helper frunction that adds boxes for iDynTree model at position and rotations for
        given joint angles'''

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
                b = np.array(boxes[n_name][0]) * self.config['scaleCollisionHull']
                body['boxsize'] = np.array([b[0][1]-b[0][0], b[1][1]-b[1][0], b[2][1]-b[2][0]])
            except KeyError:
                print('using cube for {}'.format(n_name))
                body['boxsize'] = np.array([0.1, 0.1, 0.1])
            body['center'] = 0.5*np.array([np.abs(b[0][1])-np.abs(b[0][0]),
                                           np.abs(b[1][1])-np.abs(b[1][0]),
                                           np.abs(b[2][1])-np.abs(b[2][0])])
            t = model.getWorldTransform(l)
            body['position'] = t.getPosition().toNumPy()
            body['rotation'] = self.rotationMatrixToEulerAngles(t.getRotation().toNumPy())
            if n_name in self.config['transparentLinks']:
                body['transparent'] = True
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
    import argparse
    parser = argparse.ArgumentParser(description='Visualize postures or trajectories from file')
    parser.add_argument('--config', required=True, type=str, help="use options from given config file")
    parser.add_argument('-m', '--model', required=True, type=str, help='the file to load the robot model from')
    parser.add_argument('--trajectory', required=False, type=str, help='the file to load the trajectory from')
    parser.add_argument('--world', required=False, type=str, help='the file to load world links from')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    import iDynTree; iDynTree.init_helpers(); iDynTree.init_numpy_helpers()
    dynComp = iDynTree.DynamicsComputations()
    dynComp.loadRobotModelFromFile(args.model)
    world_gravity = iDynTree.SpatialAcc.fromList([0,0,-9.81,0,0,0])
    n_dof = dynComp.getNrOfDegreesOfFreedom()

    # TODO: get this from generator / model class (other order than dynComp)
    linkNames = ['Waist', 'LHipMot', 'LThighUpLeg', 'LThighLowLeg', 'LLowLeg', 'LFootmot', 'LFoot',
                'RHipMot', 'RThighUpLeg', 'RThighLowLeg', 'RLowLeg', 'RFootmot', 'RFoot', 'DWL', 'DWS',
                'DWYTorso', 'LShp', 'LShr', 'LShy', 'LElb', 'LForearm', 'LWrMot2', 'LWrMot3',
                'LSoftHandLink', 'NeckYaw', 'NeckPitch', 'multisense/head', 'multisense/head_imu_link',
                'multisense/hokuyo_link', 'RShp', 'RShr', 'RShy', 'RElb', 'RForearm', 'RWrMot2',
                'RWrMot3', 'RSoftHandLink', 'backpack', 'TorsoProtections', 'imu_link', 'imu_link2']
    '''
    # kuka
    linkNames = ['lwr_base_link', 'lwr_1_link', 'lwr_2_link', 'lwr_3_link', 'lwr_4_link',
                 'lwr_5_link', 'lwr_6_link', 'lwr_7_link']
    '''

    # get bounding boxes for model
    from identification.helpers import URDFHelpers, ParamHelpers
    paramHelpers = ParamHelpers(None, config)
    urdfHelpers = URDFHelpers(paramHelpers, None, config)

    link_cuboid_hulls = {}  # type: Dict[str, Tuple[List, List, List]]
    for i in range(len(linkNames)):
        link_name = linkNames[i]
        box, pos, rot = urdfHelpers.getBoundingBox(
                input_urdf = args.model,
                old_com = [0,0,0],  # TODO: get from params (not important if proper hulls exist, only used for fallback)
                link_name = link_name
        )
        link_cuboid_hulls[link_name] = (box, pos, rot)

    world_boxes = {} # type: Dict[str, Tuple[List, List, List]]
    if args.world:
        world_links = urdfHelpers.getLinkNames(args.world)
        for link_name in world_links:
            box, pos, rot = urdfHelpers.getBoundingBox(
                input_urdf = args.world,
                old_com = [0,0,0],
                link_name = link_name
            )
            world_boxes[link_name] = (box, pos, rot)

    v = Visualizer(config)

    if args.trajectory:
        # display trajectory
        data = np.load(args.trajectory, encoding='latin1')
        if 'angles' in data:
            data_is_static = True
        else:
            data_is_static = False
    else:
        # just diplay model
        data_is_static = False

    def draw_model():
        if not args.trajectory:
            # just displaying model, no data
            q0 = [0.0]*n_dof
        else:
            # take angles from data
            if data_is_static:
                print('posture {}'.format(v.display_index))
                q0 = data['angles'][v.display_index]['angles']
            else:
                #TODO: get data for trajectory
                pass

        dq = iDynTree.VectorDynSize.fromList([0.0]*n_dof)
        q = iDynTree.VectorDynSize.fromList(q0)
        dynComp.setRobotState(q, dq, dq, world_gravity)
        v.addIDynTreeModel(dynComp, link_cuboid_hulls, linkNames, config['ignoreLinksForCollision'])

        if args.world:
            v.addWorld(world_boxes)

    if data_is_static:
        v.display_max = len(data['angles'])  # number of postures
    v.event_callback = draw_model
    v.event_callback()
    v.run()
