#!/usr/bin/env python

from __future__ import annotations

import collections
import math
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from excitation.trajectoryGenerator import Trajectory

import numpy as np
import pyglet
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader
from pyglet.window import key

from excitation.trajectoryGenerator import PulsedTrajectory
from identification.model import Model

# ── Matrix utility functions (replace legacy fixed-function pipeline) ──────────


def perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Build a perspective projection matrix (replaces gluPerspective)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = near - far
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / nf, (2.0 * far * near) / nf],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def ortho_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    """Build an orthographic projection matrix (replaces glOrtho)."""
    rl = right - left
    tb = top - bottom
    fn = far - near
    return np.array(
        [
            [2.0 / rl, 0.0, 0.0, -(right + left) / rl],
            [0.0, 2.0 / tb, 0.0, -(top + bottom) / tb],
            [0.0, 0.0, -2.0 / fn, -(far + near) / fn],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Build a 4x4 translation matrix (replaces glTranslatef)."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def rotation_matrix(angle_deg: float, ax: float, ay: float, az: float) -> np.ndarray:
    """Build a 4x4 rotation matrix around an arbitrary axis (replaces glRotatef)."""
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-12:
        return np.eye(4, dtype=np.float32)
    ax, ay, az = ax / length, ay / length, az / length
    t = 1.0 - c
    return np.array(
        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay, 0.0],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax, 0.0],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """Build a 4x4 scale matrix (replaces glScalef)."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def shadow_matrix(light: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Build a 4x4 planar shadow projection matrix.

    Projects geometry onto the plane defined by the 4-component plane equation
    (nx, ny, nz, d) as if cast from a point light at (lx, ly, lz, lw).
    Formula: M = dot(plane, light) * I - outer(light, plane).
    """
    d = float(np.dot(plane, light))
    m = -np.outer(light, plane)
    m[0, 0] += d
    m[1, 1] += d
    m[2, 2] += d
    m[3, 3] += d
    return m.astype(np.float32)


# ── VAO/VBO mesh wrapper (replaces pyglet.graphics.vertex_list_indexed) ───────


class VAOMesh:
    """Raw OpenGL VAO/VBO wrapper for core-profile rendering."""

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint16)

        self.index_count = len(indices)

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # position VBO — attribute 0
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # normals VBO — attribute 1
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            nbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, nbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # element buffer
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

    def draw(self, mode: int) -> None:
        """Draw with the given GL primitive mode."""
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(mode, self.index_count, gl.GL_UNSIGNED_SHORT, None)
        gl.glBindVertexArray(0)


# ── Geometry classes ──────────────────────────────────────────────────────────


class Cube:
    """vertices for a cube of size 1"""

    def __init__(self):
        self.vertices = np.array(
            [
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
            ],
            np.float32,
        )
        self.indices = np.array(
            [
                0,
                1,
                2,
                0,
                2,
                3,
                0,
                3,
                7,
                0,
                7,
                4,
                0,
                4,
                5,
                0,
                5,
                1,
                3,
                2,
                6,
                3,
                6,
                7,
                1,
                2,
                5,
                2,
                5,
                6,
                5,
                4,
                7,
                7,
                6,
                5,
            ],
            np.uint16,
        )

        # normals are unit vector from center to vertex
        c = np.array([0.0, 0.0, 0.0])
        self.normals = ((self.vertices.reshape((8, 3)) - c) / np.sqrt(3)).flatten().astype(np.float32)

    def getVerticeList(self) -> VAOMesh:
        return VAOMesh(self.vertices, self.indices, self.normals)


class Coord:
    """vertices for 3-axis coordinate system arrows"""

    def __init__(self):
        l = 0.2
        self.vertices = np.array([0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l], np.float32)
        self.indices = np.array([0, 1, 0, 2, 0, 3], np.uint16)

    def getVerticeList(self) -> VAOMesh:
        return VAOMesh(self.vertices, self.indices)


class Grid:
    """vertices for the coordinate grid"""

    def __init__(self):
        # dx, dy are the width of grid
        xmin = -50.0
        xmax = 50.0
        dx = 5.0

        self.vertices: list[tuple[float, float, float]] = []
        self.indices: list[tuple[int, int]] = []
        idx = 0

        for x in np.arange(xmin, xmax + dx, dx):
            for y in np.arange(xmin, xmax + dx, dx):
                self.vertices.append((x, xmin, 0.0))
                self.vertices.append((x, xmax, 0.0))
                self.indices.append((idx + 0, idx + 1))
                idx += 2
                self.vertices.append((xmin, y, 0.0))
                self.vertices.append((xmax, y, 0.0))
                self.indices.append((idx + 0, idx + 1))
                idx += 2

        self.vertices_flat = np.array(self.vertices, np.float32).flatten()
        self.indices_flat = np.array(self.indices, np.uint16).flatten()

    def getVerticeList(self) -> VAOMesh:
        return VAOMesh(self.vertices_flat, self.indices_flat)


class GroundQuad:
    """A large quad at z=0 used as a stencil mask for planar shadows."""

    def __init__(self, half_extent: float = 50.0) -> None:
        e = half_extent
        self.vertices = np.array(
            [-e, -e, 0.0, e, -e, 0.0, e, e, 0.0, -e, e, 0.0],
            dtype=np.float32,
        )
        self.indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    def getVerticeList(self) -> VAOMesh:
        return VAOMesh(self.vertices, self.indices)


class Mesh:
    def __init__(self, mesh_file: str, scaling: np.ndarray) -> None:
        import trimesh

        self.mesh = trimesh.load_mesh(mesh_file)
        self.num_vertices = np.size(self.mesh.vertices)

        self.normals = np.asarray(self.mesh.vertex_normals.reshape(-1), dtype=np.float32)
        self.faces = np.asarray(self.mesh.faces.reshape(-1), dtype=np.uint16)
        self.vertices: np.ndarray = self.mesh.vertices.copy()
        self.vertices[:, 0] *= scaling[0]
        self.vertices[:, 1] *= scaling[1]
        self.vertices[:, 2] *= scaling[2]
        self.vertices = np.asarray(self.vertices.reshape(-1), dtype=np.float32)

    def getVerticeList(self) -> VAOMesh:
        return VAOMesh(self.vertices, self.faces, self.normals)


# ── Camera ────────────────────────────────────────────────────────────────────


class FirstPersonCamera:
    DEFAULT_MOVEMENT_SPEED = 2.0
    DEFAULT_MOUSE_SENSITIVITY = 0.4
    DEFAULT_KEY_MAP = {
        "forward": key.W,
        "backward": key.S,
        "left": key.A,
        "right": key.D,
        "up": key.SPACE,
        "down": key.LCTRL,
    }

    class InputHandler:
        def __init__(self, window):
            self.pressed = collections.defaultdict(bool)
            self.dx = 0
            self.dy = 0
            self._window = window
            # skip the first drag event after a press: pyglet may fire one
            # spurious drag with a large delta from the cursor warp on lock
            self._skip_next_drag = False

        def on_key_press(self, symbol, modifiers):
            self.pressed[symbol] = True

        def on_key_release(self, symbol, modifiers):
            self.pressed[symbol] = False

        def on_mouse_press(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self._skip_next_drag = True
                self._window.set_exclusive_mouse(True)

        def on_mouse_release(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self._window.set_exclusive_mouse(False)

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                if self._skip_next_drag:
                    self._skip_next_drag = False
                    return
                self.dx = dx
                self.dy = dy

    def __init__(
        self,
        window,
        position=(0, 0, 0),
        pitch=-90.0,
        yaw=0.0,
        key_map=DEFAULT_KEY_MAP,
        movement_speed=DEFAULT_MOVEMENT_SPEED,
        mouse_sensitivity=DEFAULT_MOUSE_SENSITIVITY,
        y_inv=True,
    ):
        """Create camera object

        Arguments:
            window -- pyglet window with camera attached
            position -- position of camera
            key_map -- dict like FirstPersonCamera.DEFAULT_KEY_MAP
            movement_speed -- speed of camera move (scalar)
            mouse_sensitivity -- sensitivity of mouse (scalar)
            y_inv -- inversion turn above y-axis
        """

        self.__position = list(position)

        self.__yaw = yaw
        self.__pitch = pitch

        self.__input_handler = FirstPersonCamera.InputHandler(window)

        window.push_handlers(self.__input_handler)

        self.y_inv = y_inv
        self.key_map = key_map
        self.movement_speed = movement_speed
        self.mouse_sensitivity = mouse_sensitivity

        # orbit mode state (activated while shift is held)
        self._orbit_mode = False
        self._orbit_az = 0.0  # azimuth around Z in degrees
        self._orbit_el = 30.0  # elevation above XY plane in degrees
        self._orbit_r = 3.0  # distance from origin
        self._shift_prev = False

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, value: list[float]) -> None:
        self.__position = value

    @property
    def yaw(self) -> float:
        return self.__yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        """Turn above x-axis"""
        self.__yaw += value * self.mouse_sensitivity

    @property
    def pitch(self) -> float:
        return self.__pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Turn above y-axis"""
        self.__pitch += value * self.mouse_sensitivity * ((-1) if self.y_inv else 1)

    def move_forward(self, distance):
        """Fly forward along the full 3D look direction (includes pitch)."""
        p_rad = math.radians(self.__pitch)
        y_rad = math.radians(self.__yaw)
        # forward_world = (-sin(y)*sin(p), -cos(y)*sin(p), -cos(p))
        # stored position = -world_eye, so position -= forward * distance
        self.__position[0] += math.sin(y_rad) * math.sin(p_rad) * distance
        self.__position[1] += math.cos(y_rad) * math.sin(p_rad) * distance
        self.__position[2] += math.cos(p_rad) * distance

    def move_backward(self, distance):
        """Fly backward along the full 3D look direction (includes pitch)."""
        self.move_forward(-distance)

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
        dx = self.__input_handler.dx
        dy = self.__input_handler.dy
        self.__input_handler.dx = 0
        self.__input_handler.dy = 0

        shift_held = self.__input_handler.pressed[key.LSHIFT]

        # on shift press: initialise orbit parameters from the current camera position
        if shift_held and not self._shift_prev:
            # __position stores the negative of the world-space eye position
            wx, wy, wz = -self.__position[0], -self.__position[1], -self.__position[2]
            self._orbit_r = math.sqrt(wx * wx + wy * wy + wz * wz) or 3.0
            self._orbit_el = math.degrees(math.asin(max(-1.0, min(1.0, wz / self._orbit_r))))
            horiz = math.cos(math.radians(self._orbit_el))
            self._orbit_az = math.degrees(math.atan2(wy, wx)) if abs(horiz) > 1e-6 else 0.0

        # on shift release: sync yaw/pitch to the current look direction so the
        # free-cam view is continuous with the orbit view (no visual jump)
        if not shift_held and self._shift_prev:
            r = math.sqrt(sum(p * p for p in self.__position))
            if r > 1e-10:
                lx, ly, lz = self.__position[0] / r, self.__position[1] / r, self.__position[2] / r
                self.__yaw = math.degrees(math.atan2(lx, ly))
                self.__pitch = -math.degrees(math.acos(max(-1.0, min(1.0, -lz))))

        self._shift_prev = shift_held
        self._orbit_mode = shift_held

        if shift_held:
            # orbit mode: drag rotates the camera around the world origin
            if dx != 0 or dy != 0:
                self._orbit_az -= dx * self.mouse_sensitivity
                self._orbit_el += dy * self.mouse_sensitivity * ((-1) if self.y_inv else 1)
                self._orbit_el = max(-89.0, min(89.0, self._orbit_el))
            # w/s adjust the orbit radius (zoom in/out)
            if self.__input_handler.pressed[self.key_map["forward"]]:
                self._orbit_r = max(0.1, self._orbit_r - delta_time * self.movement_speed)
            if self.__input_handler.pressed[self.key_map["backward"]]:
                self._orbit_r += delta_time * self.movement_speed
            el_r = math.radians(self._orbit_el)
            az_r = math.radians(self._orbit_az)
            r = self._orbit_r
            # store negated world position (convention used by the free-cam view matrix)
            self.__position = [
                -(r * math.cos(el_r) * math.cos(az_r)),
                -(r * math.cos(el_r) * math.sin(az_r)),
                -(r * math.sin(el_r)),
            ]
        else:
            # normal first-person mode
            self.__yaw += dx * self.mouse_sensitivity
            self.__pitch += dy * self.mouse_sensitivity * ((-1) if self.y_inv else 1)

            if self.__input_handler.pressed[self.key_map["forward"]]:
                self.move_forward(delta_time * self.movement_speed)

            if self.__input_handler.pressed[self.key_map["backward"]]:
                self.move_backward(delta_time * self.movement_speed)

            if self.__input_handler.pressed[self.key_map["left"]]:
                self.move_left(delta_time * self.movement_speed)

            if self.__input_handler.pressed[self.key_map["right"]]:
                self.move_right(delta_time * self.movement_speed)

            if self.__input_handler.pressed[self.key_map["up"]]:
                self.move_up(delta_time * self.movement_speed)

            if self.__input_handler.pressed[self.key_map["down"]]:
                self.move_down(delta_time * self.movement_speed)

    def _look_at_origin(self) -> np.ndarray:
        """Build a view matrix looking from the current position toward the world origin."""
        # __position stores the negative of the world eye, so negate to get world eye
        eye = np.array([-self.__position[0], -self.__position[1], -self.__position[2]], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        f = -eye  # look toward origin
        norm_f = float(np.linalg.norm(f))
        if norm_f < 1e-10:
            return np.eye(4, dtype=np.float32)
        f = f / norm_f
        r = np.cross(f, up)
        if float(np.linalg.norm(r)) < 1e-6:
            # camera pointing straight up/down, fall back to Y-up
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            r = np.cross(f, up)
        r = r / float(np.linalg.norm(r))
        u = np.cross(r, f)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -float(np.dot(r, eye))
        m[1, 3] = -float(np.dot(u, eye))
        m[2, 3] = float(np.dot(f, eye))
        return m

    def get_view_matrix(self) -> np.ndarray:
        """Compute and return the 4x4 view matrix (replaces legacy draw())."""
        if self._orbit_mode:
            return self._look_at_origin()
        return (
            rotation_matrix(self.__pitch, 1.0, 0.0, 0.0)
            @ rotation_matrix(self.__yaw, 0.0, 0.0, 1.0)
            @ translation_matrix(*self.__position)
        )


# ── GLSL 410 core shaders ────────────────────────────────────────────────────

LIT_VERTEX_SHADER = """
#version 410 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform mat3 uNormalMat;

out vec3 vN;
out vec3 vPos;

void main() {
    vPos = vec3(uMV * vec4(aPos, 1.0));
    vN = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

LIT_FRAGMENT_SHADER = """
#version 410 core
in vec3 vN;
in vec3 vPos;

uniform vec4 uLightPos;
uniform vec4 uMatAmbient;
uniform vec4 uMatDiffuse;
uniform vec4 uMatSpecular;
uniform vec4 uMatEmission;
uniform float uMatShininess;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vN);
    vec3 L = normalize(uLightPos.xyz - vPos);
    vec3 E = normalize(-vPos);
    vec3 R = normalize(-reflect(L, N));

    vec4 Iamb = uMatAmbient * 0.3;
    vec4 Idiff = uMatDiffuse * max(dot(N, L), 0.0);
    Idiff = clamp(Idiff, 0.0, 1.0);

    vec4 Ispec = uMatSpecular * pow(max(dot(R, E), 0.0), max(uMatShininess, 1.0));
    Ispec = clamp(Ispec, 0.0, 1.0);

    fragColor = uMatEmission + Iamb + Idiff + Ispec;
}
"""

UNLIT_VERTEX_SHADER = """
#version 410 core
layout(location=0) in vec3 aPos;

uniform mat4 uMVP;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

UNLIT_FRAGMENT_SHADER = """
#version 410 core
uniform vec4 uColor;

out vec4 fragColor;

void main() {
    fragColor = uColor;
}
"""


# ── Material definitions ─────────────────────────────────────────────────────


def _f32(v: list[float]) -> np.ndarray:
    return np.array(v, dtype=np.float32)


# Numpy arrays are pre-built so _upload_material makes zero allocations per frame.
MATERIALS: dict[str, dict[str, Any]] = {
    "neutral": {
        "ambient": _f32([0.6, 0.6, 0.6, 1.0]),
        "diffuse": _f32([0.1, 0.1, 0.1, 1.0]),
        "specular": _f32([0.2, 0.2, 0.2, 1.0]),
        "shininess": 0.0,
        "emission": _f32([0.0, 0.0, 0.0, 1.0]),
    },
    "metal": {
        "ambient": _f32([0.25, 0.25, 0.25, 1.0]),
        "diffuse": _f32([0.4, 0.4, 0.4, 1.0]),
        "specular": _f32([0.774597, 0.774597, 0.774597, 1.0]),
        "shininess": 0.6 * 128.0,
        "emission": _f32([0.1, 0.1, 0.1, 1.0]),
    },
    "green rubber": {
        "ambient": _f32([0.01, 0.1, 0.01, 1.0]),
        "diffuse": _f32([0.5, 0.6, 0.5, 1.0]),
        "specular": _f32([0.05, 0.1, 0.05, 1.0]),
        "shininess": 0.03 * 128.0,
        "emission": _f32([0.0, 0.0, 0.0, 1.0]),
    },
    "white rubber": {
        "ambient": _f32([0.7, 0.7, 0.7, 1.0]),
        "diffuse": _f32([0.5, 0.5, 0.5, 1.0]),
        "specular": _f32([0.01, 0.01, 0.01, 1.0]),
        "shininess": 0.03 * 128.0,
        "emission": _f32([0.2, 0.2, 0.2, 1.0]),
    },
}


# ── Visualizer ────────────────────────────────────────────────────────────────


class Visualizer:
    def __init__(self, config: dict[str, Any]) -> None:
        # some vars
        self.window_closed = False
        self.mode = "b"  # 'b' - blocking or 'c' - continous
        self.display_index = 0  # current index for displaying e.g. postures from file
        self.display_max = 1
        self.config = config

        # keep a list of bodies
        self.bodies: list[dict[str, Any]] = []

        self.show_meshes = False

        self.angles: list[float] | None = None
        self.trajectory: Trajectory | None = None
        self.playing_traj = False  # currently playing or not
        self.playable = False  # can the trajectory be "played"
        self.freq = 1  # frequency in Hz of position / angle data

        # additional callbacks to be used with key handling
        self.event_callback: Callable | None = None
        self.timer_callback = self.next_frame
        self.info_label = None

        # shader programs and uniform caches
        self.lit_shader: int = 0
        self.unlit_shader: int = 0
        self.lit_uniforms: dict[str, int] = {}
        self.unlit_uniforms: dict[str, int] = {}

        # projection matrix (updated on resize)
        self.proj_matrix: np.ndarray = np.eye(4, dtype=np.float32)

        # light position and ground plane for shadow projection
        self.light_pos = np.array([1.0, 0.0, 2.0, 1.0], dtype=np.float32)
        # ground plane equation: z = 0  →  normal (0, 0, 1), d = 0
        self.ground_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # declared here; initialized by _initWindow / _initCamera
        self.window: pyglet.window.Window | None = None
        self.camera: FirstPersonCamera
        self.playback_rate: int = 50

        self._initWindow()
        self._initCamera()
        self._initGL()

        move_keys = "ctrl, space"  # &#8679; &#x2423;
        enter_key = "enter"  # &#x2324;
        legend = f"""<font face="Helvetica,Arial" size=15>wasd, {move_keys} - move around <br/>
        mouse drag - look / shift+drag - orbit <br/>
        {enter_key} - play/stop trajectory <br/>
        &#x2190; &#x2192; - prev/next frame <br/>
        m - show mesh/bounding boxes <br/>
        c - continous/blocking (for optimizer) <br/>
        q - close <br/>
        </font>"""
        self.help_label = pyglet.text.HTMLLabel(
            legend,
            x=10,
            y=-10,
            width=300,
            multiline=True,
            anchor_x="left",
            anchor_y="bottom",
        )
        self.info_label = pyglet.text.Label(
            "",
            font_name="Helvetica",
            font_size=11,
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=(0, 0, 0, 220),
        )
        self._last_frame_time: float = 0.0
        self._frame_times: collections.deque[float] = collections.deque(maxlen=30)
        # cache pyglet label matrices so we don't rebuild them every frame
        self._label_projection = pyglet.math.Mat4.orthogonal_projection(0, self.width, 0, self.height, -1, 500)
        self._label_view = pyglet.math.Mat4()
        self.updateLabels()

    def updateLabels(self):
        self.info_label.text = f"Index: {self.display_index}"

    def _update_fps_label(self) -> None:
        """Update FPS label using a 30-frame rolling average, throttled to 2x/sec."""
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._frame_times.append(dt)
        avg_dt = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.0
        fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        self.info_label.text = f"Index: {self.display_index}   {fps:.0f} fps"

    def update(self, dt=None):
        self.camera.update(dt)

    def _initWindow(self):
        x = 100
        y = 100
        self.width = 800
        self.height = 600
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        try:
            config_temp = pyglet.gl.Config(  # type: ignore[abstract]  # pyglet stubs incorrectly mark Config as abstract
                double_buffer=True,
                depth_size=32,
                stencil_size=8,
                sample_buffers=1,
                samples=4,
            )
            config = screen.get_best_config(config_temp)
            self.anti_alias = True
        except pyglet.window.NoSuchConfigException:
            config_temp = pyglet.gl.Config(  # type: ignore[abstract]  # pyglet stubs incorrectly mark Config as abstract
                double_buffer=True,
                depth_size=24,
                stencil_size=8,
            )
            config = screen.get_best_config(config_temp)
            self.anti_alias = False

        w = pyglet.window.Window(  # type: ignore[abstract]  # pyglet stubs incorrectly mark Window as abstract
            self.width, self.height, resizable=True, visible=False, config=config
        )
        self.window = w
        self.window_closed = False
        # Disable vsync so macOS ProMotion adaptive-sync doesn't lock us to 40 Hz.
        # Without this, CGLFlushDrawable blocks until the OS-chosen vsync tick (often
        # 40 Hz = 120/3 on ProMotion displays), wasting 15-20 ms per frame even though
        # rendering only takes ~5 ms.
        w.set_vsync(False)
        w.set_minimum_size(320, 200)
        w.set_location(x, y)
        w.set_caption("Model Visualization")
        w.push_handlers(
            on_draw=self.on_draw,
            on_resize=self.on_resize,
            on_key_press=self.on_key_press,
            on_key_release=self.on_key_release,
            on_close=self.on_close,
        )
        self.on_resize(self.width, self.height)

    def _initCamera(self):
        prev_camera: FirstPersonCamera | None = getattr(self, "camera", None)
        if prev_camera is not None:
            pos = prev_camera.position
            pitch = prev_camera.pitch
            yaw = prev_camera.yaw
        else:
            pos = [-3.272, -0.710, -1.094]
            pitch = -70.5
            yaw = -103.4
        self.camera = FirstPersonCamera(self.window, position=pos, pitch=pitch, yaw=yaw)
        self.playback_rate = 50  # trajectory playback rate (Hz), updated to match data frequency
        self.render_fps = 120  # render loop cap
        pyglet.clock.unschedule(self.update)
        # Schedule at a high rate so the event loop doesn't sleep longer than one vsync
        # interval. playback_rate is only used for the trajectory playback timer below.
        pyglet.clock.schedule_interval(self.update, 1 / 120)

    def _compile_shader(self, vs_src: str, fs_src: str) -> int:
        """Compile and link a shader program from vertex and fragment source.

        We avoid compileProgram() because it calls glValidateProgram, which
        fails on macOS core profile when no VAO is bound yet.
        """
        vs = compileShader(vs_src, gl.GL_VERTEX_SHADER)
        fs = compileShader(fs_src, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        # We intentionally skip glValidateProgram here. Validation checks whether
        # the program can execute given the *current* GL state — it is designed to
        # be called just before a draw call, not during setup. PyOpenGL's
        # compileProgram() calls it immediately after linking, which is wrong:
        # with no VAO bound yet, macOS core profile 4.1 reports "Validation Failed:
        # No vertex array object bound" even for a perfectly valid shader. Linking
        # (GL_LINK_STATUS below) already catches all real GLSL errors.
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            log = gl.glGetProgramInfoLog(program)
            raise RuntimeError(f"Shader link error: {log}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    def _cache_uniforms(self, program: int, names: list[str]) -> dict[str, int]:
        """Get uniform locations for a shader program."""
        return {name: gl.glGetUniformLocation(program, name) for name in names}

    def _upload_material(self, mat_name: str) -> None:
        """Upload material uniforms for the lit shader."""
        mat = MATERIALS.get(mat_name)
        if mat is None:
            print(f"Undefined material {mat_name}")
            return
        u = self.lit_uniforms
        gl.glUniform4fv(u["uMatAmbient"], 1, np.array(mat["ambient"], dtype=np.float32))
        gl.glUniform4fv(u["uMatDiffuse"], 1, np.array(mat["diffuse"], dtype=np.float32))
        gl.glUniform4fv(u["uMatSpecular"], 1, np.array(mat["specular"], dtype=np.float32))
        gl.glUniform4fv(u["uMatEmission"], 1, np.array(mat["emission"], dtype=np.float32))
        gl.glUniform1f(u["uMatShininess"], mat["shininess"])

    def _initGL(self):
        gl.glClearColor(0.8, 0.8, 0.9, 0)
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)
        if self.anti_alias:
            gl.glLineWidth(0.1)
        else:
            gl.glLineWidth(1.0)

        # compile shader programs
        self.lit_shader = self._compile_shader(LIT_VERTEX_SHADER, LIT_FRAGMENT_SHADER)
        self.lit_uniforms = self._cache_uniforms(
            self.lit_shader,
            [
                "uMVP",
                "uMV",
                "uNormalMat",
                "uLightPos",
                "uMatAmbient",
                "uMatDiffuse",
                "uMatSpecular",
                "uMatEmission",
                "uMatShininess",
            ],
        )

        self.unlit_shader = self._compile_shader(UNLIT_VERTEX_SHADER, UNLIT_FRAGMENT_SHADER)
        self.unlit_uniforms = self._cache_uniforms(self.unlit_shader, ["uMVP", "uColor"])

        # create VAOs for geometry
        self.cube_vao = Cube().getVerticeList()
        self.coord_vao = Coord().getVerticeList()
        self.grid_vao = Grid().getVerticeList()
        self.ground_quad_vao = GroundQuad().getVerticeList()

        # precompute shadow projection matrix (light → z=0 plane)
        self.shadow_proj = shadow_matrix(self.light_pos, self.ground_plane)

        # fill later
        self.mesh_vaos: dict[str, VAOMesh] = {}

    def init_perspective(self):
        """Compute perspective projection matrix."""
        aspect = float(self.width) / float(self.height) if self.height > 0 else 1.0
        self.proj_matrix = perspective_matrix(45.0, aspect, 0.1, 100.0)

    def on_close(self):
        self.window_closed = True
        if self.window is not None:
            self.window.close()
        self.window = None
        pyglet.app.exit()

    def on_key_press(self, symbol, modifiers):
        # print("Key pressed: {}".format(c))
        if symbol in [key.Q, key.ESCAPE]:
            print("leaving render")
            self.on_close()
            return pyglet.event.EVENT_HANDLED

        if symbol == key.C:
            if self.mode == "b":
                print("switching to continuous render")
                self.mode = "c"
                pyglet.app.exit()
                return pyglet.event.EVENT_HANDLED
            else:
                print("switching to blocking render")
                self.mode = "b"

        if symbol == key.I:
            print(f"Camera pos:{self.camera.position} pitch:{self.camera.pitch} yaw:{self.camera.yaw}")

        if symbol == key.R:
            print("Reset camera")
            self._initCamera()

        if symbol == key.M:
            self.show_meshes = not self.show_meshes
            if self.event_callback:
                self.event_callback()

        if symbol == key.RIGHT:
            if self.display_index < self.display_max - 1:
                self.display_index += 1
                if self.event_callback:
                    self.event_callback()

        if symbol == key.LEFT:
            if self.display_index > 0:
                self.display_index -= 1
                if self.event_callback:
                    self.event_callback()

        if symbol == key.ENTER:
            if not self.playing_traj and self.playable:
                self.playing_traj = True
                pyglet.clock.schedule_interval(self.timer_callback, 1 / self.playback_rate)
            else:
                self.playing_traj = False
                pyglet.clock.unschedule(self.timer_callback)

        """
        if symbol in self.pressed_keys:
            return

        # remember pressed keys until released
        self.pressed_keys.append(symbol)
        """

        return pyglet.event.EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        # if symbol in self.pressed_keys:
        #    self.pressed_keys.remove(symbol)
        pass

    def on_draw(self):
        # compute view and projection matrices once per frame
        view = self.camera.get_view_matrix()
        proj = self.proj_matrix
        vp = proj @ view

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)

        # ── Unlit pass: grid + all coordinate axes (single glUseProgram) ──────
        gl.glUseProgram(self.unlit_shader)
        u_unlit = self.unlit_uniforms

        # grid (model = identity, so MVP = VP)
        gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, vp)
        gl.glUniform4f(u_unlit["uColor"], 0.6, 0.6, 0.6, 1.0)
        self.grid_vao.draw(gl.GL_LINES)

        # coordinate axes for each body
        for b in self.bodies:
            coord_mvp = vp @ b["base_model"]
            gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, coord_mvp)
            gl.glUniform4f(u_unlit["uColor"], 0.6, 0.6, 0.6, 1.0)
            self.coord_vao.draw(gl.GL_LINES)

        # ── Lit pass: all body geometry (single glUseProgram) ─────────────────
        gl.glUseProgram(self.lit_shader)
        u_lit = self.lit_uniforms
        # transform light to view space once
        gl.glUniform4fv(
            u_lit["uLightPos"],
            1,
            (view @ self.light_pos.reshape(4, 1)).flatten(),
        )

        for b in self.bodies:
            self._draw_body_lit(b, view, proj, u_lit)

        # ── Shadow pass: project body silhouettes onto z=0 ground plane ───────
        # Step 1: stamp the ground quad into the stencil buffer (stencil = 1
        # wherever the ground plane is visible). This prevents shadows from
        # drawing outside the ground area and avoids double-darkening.
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
        gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUseProgram(self.unlit_shader)
        gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, vp)
        self.ground_quad_vao.draw(gl.GL_TRIANGLES)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)

        # Step 2: draw each body's shadow where stencil == 1, then set
        # stencil to 0 so overlapping shadows don't darken twice.
        gl.glStencilFunc(gl.GL_EQUAL, 1, 0xFF)
        gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_ZERO)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # slight z-offset to avoid z-fighting with the grid
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-1.0, -1.0)
        gl.glUniform4f(u_unlit["uColor"], 0.0, 0.0, 0.0, 0.3)
        for b in self.bodies:
            if b["geometry"] == "box":
                shadow_model = self.shadow_proj @ b["model"]
            else:
                shadow_model = self.shadow_proj @ b["base_model"]
            shadow_mvp = vp @ shadow_model
            gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, shadow_mvp)
            if b["geometry"] == "box":
                self.cube_vao.draw(gl.GL_TRIANGLES)
            elif b["geometry"] == "mesh" and b["name"] in self.mesh_vaos:
                self.mesh_vaos[b["name"]].draw(gl.GL_TRIANGLES)

        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_STENCIL_TEST)
        gl.glDepthMask(gl.GL_TRUE)

        # ── Labels: disable depth test, use pyglet's built-in rendering ───────
        self._update_fps_label()
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)

        # set up orthographic projection for pyglet labels (matrices cached, updated on resize)
        if self.window is not None:
            self.window.projection = self._label_projection
            self.window.view = self._label_view

        self.help_label.draw()
        if self.info_label is not None:
            self.info_label.draw()

        # restore depth test
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_resize(self, width, height):
        """(Re-)Init drawing."""
        gl.glViewport(0, 0, width, height)
        self.width = width
        self.height = height
        self.init_perspective()
        self._label_projection = pyglet.math.Mat4.orthogonal_projection(0, width, 0, height, -1, 500)
        if self.info_label:
            self.info_label.y = self.height - 10

        return pyglet.event.EVENT_HANDLED

    def _precompute_body_model(self, body: dict[str, Any]) -> None:
        """Precompute and cache model matrices in the body dict.

        Called once when a body is added/updated, not per frame.
        Caches base_model, model (with scale), and the 3x3 rotation submatrix
        so drawBody only needs to compute the view-dependent matrices per frame.
        """
        pos = body["position"]
        rpy = body["rotation"]
        r_deg = float(np.rad2deg(rpy[0]))
        p_deg = float(np.rad2deg(rpy[1]))
        y_deg = float(np.rad2deg(rpy[2]))

        base_model = (
            translation_matrix(float(pos[0]), float(pos[1]), float(pos[2]))
            @ rotation_matrix(y_deg, 0.0, 0.0, 1.0)
            @ rotation_matrix(p_deg, 0.0, 1.0, 0.0)
            @ rotation_matrix(r_deg, 1.0, 0.0, 0.0)
        )
        body["base_model"] = base_model
        # 3x3 rotation submatrix used for the normal matrix (inv-transpose of scale+rot)
        body["rot3"] = base_model[:3, :3].copy()

        rel_pos = body["center"]
        dim = body["size3"]
        body["model"] = (
            base_model
            @ translation_matrix(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2]))
            @ scale_matrix(float(dim[0]), float(dim[1]), float(dim[2]))
        )
        body["scale"] = np.array([float(dim[0]), float(dim[1]), float(dim[2])], dtype=np.float32)

    def drawBody(self, body: dict[str, Any], view: np.ndarray, proj: np.ndarray) -> None:
        """Draw a body (legacy entry point; delegates to the two-pass helpers)."""
        self._draw_body_lit(body, view, proj, self.lit_uniforms)

    def _draw_body_lit(self, body: dict[str, Any], view: np.ndarray, proj: np.ndarray, u: dict[str, int]) -> None:
        """Draw a body's geometry with the lit shader (lit pass only).

        Assumes the lit shader is already bound and the light uniform has been uploaded.
        The coordinate axes are drawn separately in the unlit pass (see on_draw).
        """
        if body["geometry"] == "box":
            mv = view @ body["model"]
            mvp = proj @ mv
            # normal matrix: inv(MV)^T = V_rot * R * S^{-1}
            # avoids np.linalg.inv — valid because model = T*R*S with R orthogonal, S diagonal
            normal_mat = (view[:3, :3] @ body["rot3"]) / body["scale"]
            gl.glUniformMatrix4fv(u["uMVP"], 1, gl.GL_TRUE, mvp)
            gl.glUniformMatrix4fv(u["uMV"], 1, gl.GL_TRUE, mv)
            gl.glUniformMatrix3fv(u["uNormalMat"], 1, gl.GL_TRUE, np.ascontiguousarray(normal_mat))
            self._upload_material(body["material"])
            transparent = body.get("transparent", False)
            if transparent:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            self.cube_vao.draw(gl.GL_TRIANGLES)
            if transparent:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif body["geometry"] == "mesh":
            # meshes use base_model (already scaled at load time, no extra S)
            mv = view @ body["base_model"]
            mvp = proj @ mv
            normal_mat = view[:3, :3] @ body["rot3"]  # S=I for meshes
            gl.glUniformMatrix4fv(u["uMVP"], 1, gl.GL_TRUE, mvp)
            gl.glUniformMatrix4fv(u["uMV"], 1, gl.GL_TRUE, mv)
            gl.glUniformMatrix3fv(u["uNormalMat"], 1, gl.GL_TRUE, np.ascontiguousarray(normal_mat))
            self._upload_material(body["material"])
            self.mesh_vaos[body["name"]].draw(gl.GL_TRIANGLES)

    def addBox(self, size, pos, rpy):
        body: dict[str, Any] = {}
        body["geometry"] = "box"
        body["material"] = "white rubber"
        body["size3"] = np.array([size, size, size])
        body["center"] = body["size3"] * 0.5
        body["position"] = pos
        body["rotation"] = rpy
        self._precompute_body_model(body)
        self.bodies.append(body)

    def addWorld(self, boxes: dict) -> None:
        for linkName in boxes:
            body: dict[str, Any] = {}
            body["geometry"] = "box"
            body["material"] = "white rubber"
            b = np.array(boxes[linkName][0])
            body["size3"] = np.array([b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]])
            body["center"] = 0.5 * np.array(
                [
                    b[1][0] + b[0][0],
                    b[1][1] + b[0][1],
                    b[1][2] + b[0][2],
                ]
            )
            body["position"] = boxes[linkName][1]
            body["rotation"] = boxes[linkName][2]
            self._precompute_body_model(body)
            self.bodies.append(body)

    def setModelTrajectory(self, trajectory):
        self.trajectory = trajectory

    def next_frame(self, dt):
        if self.display_index >= self.display_max:
            self.display_index = 0
        self.display_index += 1
        if self.event_callback is not None:
            self.event_callback()

    def loadMeshes(self, urdfpath, linkNames, urdfHelpers):
        # load meshes
        if not len(self.mesh_vaos):
            for i in range(0, len(linkNames)):
                filename = urdfHelpers.getMeshPath(urdfpath, linkNames[i])
                if filename and os.path.exists(filename):
                    # use last mesh scale (from getMeshPath)
                    scale_parts = urdfHelpers.mesh_scaling.split(" ")
                    scale = np.array([float(scale_parts[0]), float(scale_parts[1]), float(scale_parts[2])])
                    self.mesh_vaos[linkNames[i]] = Mesh(filename, scale).getVerticeList()
            if len(self.mesh_vaos):
                self.show_meshes = True

    def addIDynTreeModel(self, kinDyn, boxes, real_links, ignore_links):
        """helper function that adds boxes for iDynTree model at position and rotations for
        given joint angles"""

        if self.window_closed:
            self._initWindow()
            self._initCamera()

        self.bodies = []
        for l in range(kinDyn.getNrOfLinks()):
            n_name = kinDyn.getFrameName(l)
            if n_name in ignore_links:
                continue
            if n_name not in real_links:
                continue
            body: dict[str, Any] = {}
            body["name"] = n_name
            if self.show_meshes and n_name in self.mesh_vaos:
                body["geometry"] = "mesh"
                body["size3"] = [1.0, 1.0, 1.0]
                body["center"] = [0.0, 0.0, 0.0]
                body["material"] = "metal"
            else:
                body["geometry"] = "box"
                body["material"] = "white rubber"
                try:
                    b = np.array(boxes[n_name][0]) * self.config["scaleCollisionHull"]
                    p = np.array(boxes[n_name][1])
                    body["size3"] = np.array([b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]])
                    body["center"] = 0.5 * (b[0] + b[1]) + p
                except KeyError:
                    print(f"using cube for {n_name}")
                    body["size3"] = np.array([0.1, 0.1, 0.1])
                    body["center"] = [0.0, 0.0, 0.0]

            t = kinDyn.getWorldTransform(l)
            body["position"] = t.getPosition().toNumPy()
            rpy = t.getRotation().asRPY()
            body["rotation"] = [rpy.getVal(0), rpy.getVal(1), rpy.getVal(2)]

            if "transparentLinks" in self.config and n_name in self.config["transparentLinks"]:
                body["transparent"] = True

            self._precompute_body_model(body)
            self.bodies.append(body)

    def stop(self, dt):
        pyglet.app.exit()

    def run(self):
        if self.window is not None:
            self.window.set_visible()
        # from IPython import embed
        # embed()
        if self.mode == "b":
            # pyglet.app.run() on macOS only triggers on_draw on ~2/3 of event loop
            # iterations due to Cocoa window-invalidation timing, capping us at ~40fps.
            # An explicit render loop bypasses this and draws every iteration.
            _frame_budget = 1.0 / self.render_fps
            _frame_start = time.perf_counter()
            while not self.window_closed and self.window is not None:
                pyglet.clock.tick()
                self.window.switch_to()
                self.window.dispatch_events()
                if self.window_closed or self.window is None:
                    break
                self.window.dispatch_event("on_draw")
                self.window.flip()
                # sleep for the remainder of the 120fps frame budget
                _elapsed = time.perf_counter() - _frame_start
                _sleep = _frame_budget - _elapsed
                if _sleep > 0:
                    time.sleep(_sleep)
                _frame_start = time.perf_counter()
        else:
            # run one loop iteration only (draw one frame)
            pyglet.clock.tick()
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event("on_draw")
                window.flip()

        if self.mode == "c":
            pyglet.clock.schedule_once(self.stop, 1 / self.playback_rate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize postures or trajectories from file")
    parser.add_argument("--config", required=True, type=str, help="use options from given config file")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="the file to load the robot model from",
    )
    parser.add_argument(
        "--trajectory",
        required=False,
        type=str,
        help="the file to load the trajectory from",
    )
    parser.add_argument("--world", required=False, type=str, help="the file to load world links from")
    args = parser.parse_args()

    import yaml

    with open(args.config) as stream:
        try:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)

    from idyntree import bindings as iDynTree

    loader = iDynTree.ModelLoader()
    loader.loadModelFromFile(args.model)
    kinDyn = iDynTree.KinDynComputations()
    kinDyn.loadRobotModel(loader.model())
    gravity = iDynTree.Vector3()
    gravity.setVal(2, -9.81)
    n_dof = kinDyn.getNrOfDegreesOfFreedom()
    config["num_dofs"] = n_dof
    config["urdf"] = args.model

    g_model = Model(config, args.model, regressor_file=None, regressor_init=False)
    linkNames = g_model.linkNames

    # get bounding boxes for model
    from identification.helpers import ParamHelpers, URDFHelpers

    paramHelpers = ParamHelpers(g_model, config)
    urdfHelpers = URDFHelpers(paramHelpers, g_model, config)

    BBoxEntry = tuple[list[list[float]], list[float], np.ndarray | list[float]]
    link_cuboid_hulls: dict[str, BBoxEntry] = {}
    for i in range(len(linkNames)):
        link_name = linkNames[i]
        box, pos, rot = urdfHelpers.getBoundingBox(
            input_urdf=args.model, old_com=[0, 0, 0], link_name=link_name, scaling=False
        )
        link_cuboid_hulls[link_name] = (box, pos, rot)

    world_boxes: dict[str, BBoxEntry] = {}
    if args.world:
        world_links = urdfHelpers.getLinkNames(args.world)
        for link_name in world_links:
            box, pos, rot = urdfHelpers.getBoundingBox(
                input_urdf=args.world,
                old_com=[0, 0, 0],
                link_name=link_name,
                scaling=False,
            )
            world_boxes[link_name] = (box, pos, rot)

    v = Visualizer(config)

    v.loadMeshes(args.model, linkNames, urdfHelpers)

    if args.trajectory:
        # display trajectory
        data = np.load(args.trajectory, encoding="latin1")
        if "angles" in data:
            data_type = "static"
        elif "positions" in data:
            data_type = "measurements"
            v.playable = True
        else:
            data_type = "trajectory"
            v.playable = True
    else:
        # just diplay model
        data_type = "none"

    def draw_model():
        if not args.trajectory:
            # just displaying model, no data
            q0 = [0.0] * n_dof
        else:
            # take angles from data
            if data_type == "static":
                q0 = data["angles"][v.display_index]["angles"]
            elif data_type == "trajectory":
                # get data of trajectory
                if v.trajectory is not None:
                    v.trajectory.setTime(v.display_index / v.playback_rate)
                    q0 = [v.trajectory.getAngle(d) for d in range(config["num_dofs"])]
            elif data_type == "measurements":
                idx = int(v.display_index * v.freq / v.playback_rate)
                if idx > data["positions"].shape[0] - 1:
                    v.display_index = 0
                    idx = 0
                q0 = data["positions"][idx, :]

        s = iDynTree.JointPosDoubleArray(n_dof)
        ds = iDynTree.JointDOFsDoubleArray(n_dof)
        for _i in range(n_dof):
            s.setVal(_i, float(q0[_i]))
        kinDyn.setRobotState(s, ds, gravity)
        v.addIDynTreeModel(kinDyn, link_cuboid_hulls, linkNames, config["ignoreLinksForCollision"])

        if args.world:
            v.addWorld(world_boxes)

        v.updateLabels()

    if args.trajectory:
        if data_type == "static":
            v.display_max = len(data["angles"])  # number of postures
        elif data_type == "trajectory":
            trajectory = PulsedTrajectory(n_dof, use_deg=data["use_deg"])
            trajectory.initWithParams(data["a"], data["b"], data["q"], data["nf"], data["wf"])
            v.setModelTrajectory(trajectory)

            v.freq = config["excitationFrequency"]
            v.playback_rate = v.freq
            v.display_max = trajectory.getPeriodLength() * v.playback_rate  # length of trajectory
        elif data_type == "measurements":
            v.freq = config["excitationFrequency"]
            v.playback_rate = v.freq
            v.display_max = data["positions"].shape[0]

    v.event_callback = draw_model
    v.event_callback()
    v.run()
