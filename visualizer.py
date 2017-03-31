#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from typing import Tuple, List, Dict, Callable, Any
import math
import collections
import sys
import os

import numpy as np
from OpenGL import GLU
from OpenGL.GL.shaders import compileShader, compileProgram
import pyglet
from pyglet import gl
from pyglet.window import key

from identification.model import Model
from excitation.trajectoryGenerator import PulsedTrajectory, Trajectory

# convert python list to gldouble array
def glvec(v):
    return (gl.GLdouble * len(v))(*v)

def glvecf(v):
    return (gl.GLfloat * len(v))(*v)

# define some geometries

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
        c = np.array([0.0,0.0,0.0])
        self.normals = ((self.vertices.reshape((8,3)) - c) / np.sqrt(3)).flatten()
        '''self.normals = np.array([-0.28867513,  0.28867513, -0.28867513,
                                -0.28867513, -0.28867513, -0.28867513,
                                0.28867513, -0.28867513, -0.28867513,
                                0.28867513, 0.28867513, -0.28867513,
                                -0.28867513,  0.28867513,  0.28867513,
                               -0.28867513, -0.28867513,  0.28867513,
                               0.28867513, -0.28867513, 0.28867513,
                               0.28867513,  0.28867513,  0.28867513])
        '''

    def getVerticeList(self):
        return pyglet.graphics.vertex_list_indexed(len(self.vertices)//3, self.indices,
                                                   ('v3f', self.vertices), ('n3f', self.normals))

class Coord(object):
    ''' vertices for 3-axis coordinate system arrows '''
    def __init__(self):
        l = 0.2
        self.vertices = np.array([0.0, 0.0, 0.0,
                                  l, 0.0, 0.0,
                                  0.0, l, 0.0,
                                  0.0, 0.0, l], np.float32)
        self.indices = np.array([0,1, 0,2, 0,3], np.ushort)

    def getVerticeList(self):
        return pyglet.graphics.vertex_list_indexed(len(self.vertices)//3, self.indices,
                                                   ('v3f', self.vertices))

class Grid(object):
    '''vertices for the coordinate grid'''
    def __init__(self):
        # dx, dy are the width of grid
        xmin = -50.0
        xmax = 50.0
        dx = 5.0

        self.vertices = []     # type: np.ndarray
        self.indices = []      # type: np.ndarray
        idx = 0

        for x in np.arange(xmin, xmax+dx, dx):
            for y in np.arange(xmin, xmax+dx, dx):
                self.vertices.append((x, xmin, 0.0))
                self.vertices.append((x, xmax, 0.0))
                self.indices.append((idx+0,idx+1))
                idx += 2
                self.vertices.append((xmin, y, 0.0))
                self.vertices.append((xmax, y, 0.0))
                self.indices.append((idx+0,idx+1))
                idx += 2

        self.vertices = np.array(self.vertices, np.float32).flatten()
        self.indices = np.array(self.indices, np.ushort).flatten()

    def getVerticeList(self):
        return pyglet.graphics.vertex_list_indexed(len(self.vertices)//3, self.indices, ('v3f', self.vertices))


class Mesh(object):
    def __init__(self, mesh_file, scaling):
        # type: (str, np._ArrayLike) -> None
        import trimesh
        self.mesh = trimesh.load_mesh(mesh_file)
        self.num_vertices = np.size(self.mesh.vertices)

        self.normals = self.mesh.vertex_normals.reshape(-1).tolist() #.flatten()
        self.faces = self.mesh.faces.reshape(-1).tolist()
        self.vertices = self.mesh.vertices
        self.vertices[:, 0] *= scaling[0]
        self.vertices[:, 1] *= scaling[1]
        self.vertices[:, 2] *= scaling[2]
        self.vertices = self.vertices.reshape(-1).tolist()

    def getVerticeList(self):
        return pyglet.graphics.vertex_list_indexed(self.num_vertices//3, self.faces, ('v3f/static', self.vertices), ('n3f/static', self.normals))


class FirstPersonCamera(object):
    DEFAULT_MOVEMENT_SPEED = 2.0
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
        # type: (Dict[str, Any]) -> None
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

        self.show_meshes = False

        self.angles = None  # type: List[float]
        self.trajectory = None  # type: Trajectory
        self.playing_traj = False  # currently playing or not
        self.playable = False   # can the trajectory be "played"
        self.freq = 1   # frequency in Hz of position / angle data

        # additional callbacks to be used with key handling
        self.event_callback = None  # type: Callable
        self.timer_callback = self.next_frame
        self.info_label = None

        self._initWindow()
        self._initCamera()
        self._initGL()

        move_keys = "lshift, space"   # &#8679; &#x2423;
        enter_key = "enter"   # &#x2324;
        legend = '''<font face="Helvetica,Arial" size=15>wasd, {} - move around <br/>
        mouse drag - look <br/>
        {} - play/stop trajectory <br/>
        &#x2190; &#x2192; - prev/next frame <br/>
        m - show mesh/bounding boxes <br/>
        c - continous/blocking (for optimizer) <br/>
        q - close <br/>
        </font>'''.format(move_keys, enter_key)
        self.help_label = pyglet.text.HTMLLabel(legend,
                          x = 10, y = -10,
                          width = 300,
                          multiline = True,
                          anchor_x='left', anchor_y='bottom')
        self.info_label = pyglet.text.HTMLLabel('',
                          x = 10, y = self.height - 10,
                          width = 50,
                          multiline = False,
                          anchor_x='left', anchor_y='top')
        self.updateLabels()

    def updateLabels(self):
        self.info_label.text = '<font face="Helvetica,Arial" size=15>Index: {}</font>'.format(self.display_index)

    def update(self, dt=None):
        self.camera.update(dt)

    def _initWindow(self):
        x = 100
        y = 100
        self.width = 800
        self.height = 600
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        try:
            config_temp = gl.Config(double_buffer=True, depth_size=32, sample_buffers=1, samples=4)
            config = screen.get_best_config(config_temp)
            self.anti_alias = True
        except pyglet.window.NoSuchConfigException:
            config_temp = gl.Config(double_buffer=True, depth_size=24)
            config = screen.get_best_config(config_temp)
            self.anti_alias = False

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
            pos = [-3.272, -0.710, -1.094]
            pitch = -70.5
            yaw = -103.4
        self.camera = FirstPersonCamera(self.window, position=pos, pitch=pitch, yaw=yaw)
        self.fps = 50
        pyglet.clock.unschedule(self.update)
        pyglet.clock.schedule_interval(self.update, 1/self.fps)

    def setLights(self):
        pos = [1.0, 0.0, 2.0, 1.0]
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, glvecf(pos))
        gl.glEnable(gl.GL_LIGHT0)
        #self.addBox(0.1, pos, [0,0,0])

    def setMaterial(self, name):
        if name == 'neutral':
            # 'lines'
            mat_ambient = [0.6, 0.6, 0.6, 1.0]    #[0.3, 0.3, 0.4, 1.0]
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, glvecf(mat_ambient));
            mat_diffuse = [0.1, 0.1, 0.1]  #[0.7, 0.5, 0.5, 1.0]
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, glvecf(mat_diffuse))
            mat_specular = [0.2, 0.2, 0.2]
            gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, glvecf(mat_specular));
            shine = 0.0
            gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, shine * 128.0);
        elif name == 'metal':
            # 'chrome'
            mat_ambient = [0.25, 0.25, 0.25, 1.0]    #[0.3, 0.3, 0.4, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, glvecf(mat_ambient));
            mat_diffuse = [0.4, 0.4, 0.4]  #[0.7, 0.5, 0.5, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, glvecf(mat_diffuse))
            mat_specular = [0.774597, 0.774597, 0.774597]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, glvecf(mat_specular));
            shine = 0.6
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, shine * 128.0);
            mat_emission = [0.1, 0.1, 0.1, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, glvecf(mat_emission))
        elif name == 'green rubber':
            mat_ambient = [0.01, 0.1, 0.01, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, glvecf(mat_ambient));
            mat_diffuse = [0.5, 0.6, 0.5]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, glvecf(mat_diffuse))
            mat_specular = [0.05, 0.1, 0.05]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, glvecf(mat_specular));
            shine = 0.03
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, shine * 128.0);
            #mat_emission = [0.1, 0.1, 0.15, 1.0]
            #gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, glvecf(mat_emission))
        elif name == 'white rubber':
            mat_ambient = [0.7, 0.7, 0.7, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, glvecf(mat_ambient));
            mat_diffuse = [0.5, 0.5, 0.5]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, glvecf(mat_diffuse))
            mat_specular = [0.01, 0.01, 0.01]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, glvecf(mat_specular));
            shine = 0.03
            gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, shine * 128.0);
            mat_emission = [0.2, 0.2, 0.2, 1.0]
            gl.glMaterialfv(gl.GL_FRONT, gl.GL_EMISSION, glvecf(mat_emission))
        else:
            print('Undefined material {}'.format(name))

    def _initGL(self):
        gl.glClearColor(0.8,0.8,0.9,0)
        gl.glClearDepth(1.0)                       # Enables Clearing Of The Depth Buffer
        gl.glDepthFunc(gl.GL_LESS)                 # The Type Of Depth Test To Do
        gl.glHint(gl.GL_PERSPECTIVE_CORRECTION_HINT, gl.GL_NICEST)   # make stuff look nice
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        if self.anti_alias:
            gl.glLineWidth(0.1)
        else:
            gl.glLineWidth(1.0)

        #gl.glEnable(gl.GL_BLEND)
        #gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_DEPTH_TEST)              # Enables Depth Testing
        gl.glEnable(gl.GL_LIGHTING)
        #gl.glEnable(gl.GL_NORMALIZE)
        #gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_FALSE)

        if not gl.glUseProgram:
            print("Can't run shaders!")
            sys.exit(1)

        self.default_shader = compileProgram(
            compileShader('''
                varying vec3 vN;
                varying vec3 v;
                void main(void)
                {
                   v = vec3(gl_ModelViewMatrix * gl_Vertex);
                   vN = normalize(gl_NormalMatrix * gl_Normal);
                   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                }

            ''', gl.GL_VERTEX_SHADER),
            compileShader('''
                varying vec3 vN;
                varying vec3 v;
                #define MAX_LIGHTS 1

                void main (void)
                {
                   vec3 N = normalize(vN);
                   vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);

                   for (int i=0;i<MAX_LIGHTS;i++)
                   {
                      vec3 L = normalize(gl_LightSource[i].position.xyz - v);
                      vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0)
                      vec3 R = normalize(-reflect(L,N));

                      //calculate Ambient Term:
                      vec4 Iamb = gl_FrontLightProduct[i].ambient;
                      //calculate Diffuse Term:
                      vec4 Idiff = gl_FrontLightProduct[i].diffuse * max(dot(N,L), 0.0);
                      Idiff = clamp(Idiff, 0.0, 1.0);

                      // calculate Specular Term:
                      vec4 Ispec = gl_FrontLightProduct[i].specular
                             * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);
                      Ispec = clamp(Ispec, 0.0, 1.0);

                      finalColor += Iamb + Idiff + Ispec;
                   }

                   // write Total Color:
                   gl_FragColor = gl_FrontLightModelProduct.sceneColor + finalColor;
                }
            ''', gl.GL_FRAGMENT_SHADER),)

        self.cube_list = Cube().getVerticeList()
        self.coord_list = Coord().getVerticeList()
        self.grid_list = Grid().getVerticeList()

        # fill later
        self.mesh_lists = {}   # type: Dict[str, Any]

    def init_ortho(self):
        # disable shaders
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_LIGHTING)

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
        gl.glEnable(gl.GL_LIGHTING)
        # Init Projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        GLU.gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 100.0)
        # Initialize ModelView matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

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

        if symbol == key.M:
            self.show_meshes = not self.show_meshes
            if self.event_callback:
                self.event_callback()

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

        if symbol == key.ENTER:
            if not self.playing_traj and self.playable:
                self.playing_traj = True
                pyglet.clock.schedule_interval(self.timer_callback, 1/self.fps)
            else:
                self.playing_traj = False
                pyglet.clock.unschedule(self.timer_callback)

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

        self.camera.draw()
        self.setLights()

        self.drawGrid()

        # run shaders
        if self.default_shader:
            gl.glUseProgram(self.default_shader)

        for b in self.bodies:
            self.drawBody(b)

        self.init_ortho()
        self.help_label.draw()
        self.info_label.draw()

    def on_resize(self, width, height):
        """(Re-)Init drawing.
        """
        # Viewport
        gl.glViewport(0,0, width, height)
        self.width = width
        self.height = height
        self.init_perspective()
        if self.info_label:
            self.info_label.y = self.height - 10

        return pyglet.event.EVENT_HANDLED

    def drawCoords(self):
        self.setMaterial('neutral')
        self.coord_list.draw(gl.GL_LINES)

    def drawGrid(self):
        self.setMaterial('neutral')
        self.grid_list.draw(gl.GL_LINES)

    def drawCube(self):
        #gl.glEnableVertexAttribArray(0)
        #gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, cube.vertices)
        #gl.glDrawElements(gl.GL_TRIANGLES, cube.indices)
        self.cube_list.draw(gl.GL_TRIANGLES)

    def drawMesh(self, linkName):
        self.mesh_lists[linkName].draw(gl.GL_TRIANGLES)

    def drawBody(self, body):
        # type: (Dict[str, Any]) -> None
        """Draw a body"""

        pos = body['position']
        rpy = body['rotation']
        r,p,y = rpy[0], rpy[1], rpy[2]

        gl.glPushMatrix()
        gl.glTranslatef(pos[0], pos[1], pos[2])
        gl.glRotatef(np.rad2deg(y), 0.0, 0.0, 1.0)
        gl.glRotatef(np.rad2deg(p), 0.0, 1.0, 0.0)
        gl.glRotatef(np.rad2deg(r), 1.0, 0.0, 0.0)

        self.drawCoords()

        rel_pos = body['center']
        gl.glTranslatef(rel_pos[0], rel_pos[1], rel_pos[2])

        transparent = 'transparent' in body and body['transparent']
        dim = body['size3']
        gl.glScalef(dim[0], dim[1], dim[2])
        self.setMaterial(body['material'])
        if body['geometry'] is 'box':
            if transparent:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)   # Wireframe
            self.drawCube()
            if transparent:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif body['geometry'] is 'mesh':
            self.drawMesh(body['name'])

        gl.glPopMatrix()

    def addBox(self, size, pos, rpy):
        body = {}  # type: Dict[str, Any]
        body['geometry'] = 'box'
        body['material'] = 'white rubber'
        body['size3'] = np.array([size, size, size])
        body['center'] = body['size3'] * 0.5
        body['position'] = pos
        body['rotation'] = rpy
        self.bodies.append(body)

    def addWorld(self, boxes):
        # type: (Dict) -> None
        for linkName in boxes:
            body = {}  # type: Dict[str, Any]
            body['geometry'] = 'box'
            body['material'] = 'white rubber'
            b = np.array(boxes[linkName][0])
            body['size3'] = np.array([b[1][0]-b[0][0], b[1][1]-b[0][1], b[1][2]-b[0][2]])
            body['center'] = 0.5*np.array([np.abs(b[1][0])-np.abs(b[0][0]),
                                           np.abs(b[1][1])-np.abs(b[0][1]),
                                           np.abs(b[1][2])-np.abs(b[0][2])])
            body['position'] = boxes[linkName][1]
            body['rotation'] = boxes[linkName][2]
            self.bodies.append(body)

    def setModelTrajectory(self, trajectory):
        self.trajectory = trajectory

    def next_frame(self, dt):
        if self.display_index >= self.display_max:
            self.display_index = 0
        self.display_index += 1
        self.event_callback()

    def loadMeshes(self, urdfpath, linkNames, urdfHelpers):
        # load meshes
        if not len(self.mesh_lists):
            for i in range(0, len(linkNames)):
                filename = urdfHelpers.getMeshPath(urdfpath, linkNames[i])
                if filename and os.path.exists(filename):
                    # use last mesh scale (from getMeshPath)
                    scale = urdfHelpers.mesh_scaling.split(' ')
                    scale = [float(scale[0]), float(scale[1]), float(scale[2])]
                    self.mesh_lists[linkNames[i]] = Mesh(filename, scale).getVerticeList()
            if len(self.mesh_lists):
                self.show_meshes = True

    def addIDynTreeModel(self,
                  model,          # type: iDynTree.DynamicsComputations
                  boxes,          # type: Dict[str, Tuple[List, List, List]]     # link hulls
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
            body['name'] = n_name
            if self.show_meshes and n_name in self.mesh_lists:
                body['geometry'] = 'mesh'
                body['size3'] = [1.0, 1.0, 1.0]
                body['center'] = [0.0, 0.0, 0.0]
                body['material'] = 'metal'
            else:
                body['geometry'] = 'box'
                body['material'] = 'white rubber'
                try:
                    b = np.array(boxes[n_name][0]) * self.config['scaleCollisionHull']
                    p = np.array(boxes[n_name][1])
                    body['size3'] = np.array([b[1][0]-b[0][0], b[1][1]-b[0][1], b[1][2]-b[0][2]])
                    body['center'] = 0.5*np.array([np.abs(b[1][0])-np.abs(b[0][0]) + p[0],
                                                   np.abs(b[1][1])-np.abs(b[0][1]) + p[1],
                                                   np.abs(b[1][2])-np.abs(b[0][2])] + p[2])
                except KeyError:
                    print('using cube for {}'.format(n_name))
                    body['size3'] = np.array([0.1, 0.1, 0.1])
                    body['center'] = [0.0, 0.0, 0.0]

            t = model.getWorldTransform(l)
            body['position'] = t.getPosition().toNumPy()
            rpy = t.getRotation().asRPY()
            body['rotation'] = [rpy.getVal(0), rpy.getVal(1), rpy.getVal(2)]

            if 'transparentLinks' in self.config and n_name in self.config['transparentLinks']:
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
            #run one loop iteration only (draw one frame)
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
    config['num_dofs'] = n_dof
    config['urdf'] = args.model

    g_model = Model(config, args.model, regressor_file=None, regressor_init=False)
    linkNames = g_model.linkNames

    # get bounding boxes for model
    from identification.helpers import URDFHelpers, ParamHelpers
    paramHelpers = ParamHelpers(None, config)
    urdfHelpers = URDFHelpers(paramHelpers, None, config)

    link_cuboid_hulls = {}  # type: Dict[str, Tuple[List, List, List]]
    for i in range(len(linkNames)):
        link_name = linkNames[i]
        box, pos, rot = urdfHelpers.getBoundingBox(
                input_urdf = args.model,
                old_com = [0,0,0],
                link_name = link_name,
                scaling = False
        )
        link_cuboid_hulls[link_name] = (box, pos, rot)

    world_boxes = {} # type: Dict[str, Tuple[List, List, List]]
    if args.world:
        world_links = urdfHelpers.getLinkNames(args.world)
        for link_name in world_links:
            box, pos, rot = urdfHelpers.getBoundingBox(
                input_urdf = args.world,
                old_com = [0,0,0],
                link_name = link_name,
                scaling = False
            )
            world_boxes[link_name] = (box, pos, rot)

    v = Visualizer(config)

    v.loadMeshes(args.model, linkNames, urdfHelpers)

    if args.trajectory:
        # display trajectory
        data = np.load(args.trajectory, encoding='latin1')
        if 'angles' in data:
            data_type = 'static'
        elif 'positions' in data:
            data_type = 'measurements'
            v.playable = True
        else:
            data_type = 'trajectory'
            v.playable = True
    else:
        # just diplay model
        data_type = 'none'

    def draw_model():
        if not args.trajectory:
            # just displaying model, no data
            q0 = [0.0]*n_dof
        else:
            # take angles from data
            if data_type == 'static':
                q0 = data['angles'][v.display_index]['angles']
            elif data_type == 'trajectory':
                # get data of trajectory
                v.trajectory.setTime(v.display_index/v.fps)
                q0 = [v.trajectory.getAngle(d) for d in range(config['num_dofs'])]
            elif data_type == 'measurements':
                idx = int(v.display_index*v.freq/v.fps)
                if idx > data['positions'].shape[0]-1:
                    v.display_index = 0
                    idx = 0
                q0 = data['positions'][idx, :]

        q = iDynTree.VectorDynSize.fromList(q0)
        dq = iDynTree.VectorDynSize.fromList([0.0]*n_dof)
        dynComp.setRobotState(q, dq, dq, world_gravity)
        v.addIDynTreeModel(dynComp, link_cuboid_hulls, linkNames, config['ignoreLinksForCollision'])

        if args.world:
            v.addWorld(world_boxes)

        v.updateLabels()

    if args.trajectory:
        if data_type == 'static':
            v.display_max = len(data['angles'])  # number of postures
        elif data_type == 'trajectory':
            trajectory = PulsedTrajectory(n_dof, use_deg=data['use_deg'])
            trajectory.initWithParams(data['a'], data['b'], data['q'], data['nf'], data['wf'])
            v.setModelTrajectory(trajectory)

            v.freq = config['excitationFrequency']
            v.display_max = trajectory.getPeriodLength()*v.fps # length of trajectory
        elif data_type == 'measurements':
            v.freq = config['excitationFrequency']
            v.display_max = data['positions'].shape[0]

    v.event_callback = draw_model
    v.event_callback()
    v.run()
