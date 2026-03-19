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

import platform
import subprocess

import numpy as np
import pyglet
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader
from pyglet.window import key

from excitation.trajectoryGenerator import PulsedTrajectory
from identification.model import Model


def _is_dark_mode() -> bool:
    """Detect if the OS is in dark mode (macOS, GNOME, KDE)."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip().lower() == "dark"
        elif system == "Linux":
            # GNOME
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True,
                text=True,
            )
            if "dark" in result.stdout.lower():
                return True
            # KDE
            result = subprocess.run(
                ["kreadconfig5", "--group", "General", "--key", "ColorScheme"],
                capture_output=True,
                text=True,
            )
            if "dark" in result.stdout.lower():
                return True
    except Exception:
        pass
    return False


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


class CylinderGeom:
    """Unit open-ended cylinder along Z from -0.5 to 0.5, radius 1.

    Used as the shaft of a capsule. Hemisphere caps are rendered separately as
    SphereGeom instances at each endpoint to avoid non-uniform scale distortion.
    """

    def __init__(self, n_segments: int = 16) -> None:
        """Generate cylinder side vertices, normals, and triangle indices."""
        verts: list[list[float]] = []
        norms: list[list[float]] = []
        idxs: list[int] = []

        for z in [-0.5, 0.5]:
            for i in range(n_segments + 1):
                theta = 2.0 * np.pi * i / n_segments
                x, y = np.cos(theta), np.sin(theta)
                verts.append([x, y, z])
                norms.append([x, y, 0.0])

        for i in range(n_segments):
            b = 0
            t = n_segments + 1
            idxs.extend([b + i, b + i + 1, t + i])
            idxs.extend([b + i + 1, t + i + 1, t + i])

        self.vertices = np.array(verts, dtype=np.float32).flatten()
        self.normals = np.array(norms, dtype=np.float32).flatten()
        self.indices = np.array(idxs, dtype=np.uint16)

    def getVerticeList(self) -> VAOMesh:
        """Create a VAOMesh for this cylinder geometry."""
        return VAOMesh(self.vertices, self.indices, self.normals)


class SphereGeom:
    """Unit sphere of radius 1 centered at origin.

    Used for hemisphere caps of capsule rendering.
    """

    def __init__(self, n_segments: int = 16, n_rings: int = 12) -> None:
        """Generate sphere vertices, normals, and triangle indices."""
        verts: list[list[float]] = []
        norms: list[list[float]] = []
        idxs: list[int] = []

        # generate vertices ring by ring from bottom pole to top pole
        for ring in range(n_rings + 1):
            phi = np.pi * ring / n_rings  # 0 (top) to pi (bottom)
            y = float(np.cos(phi))
            r = float(np.sin(phi))
            for seg in range(n_segments + 1):
                theta = 2.0 * np.pi * seg / n_segments
                x = float(r * np.cos(theta))
                z = float(r * np.sin(theta))
                verts.append([x, y, z])
                norms.append([x, y, z])  # unit sphere: normal = position

        # triangulate adjacent rings
        for ring in range(n_rings):
            for seg in range(n_segments):
                r0 = ring * (n_segments + 1) + seg
                r1 = r0 + n_segments + 1
                idxs.extend([r0, r1, r0 + 1])
                idxs.extend([r0 + 1, r1, r1 + 1])

        self.vertices = np.array(verts, dtype=np.float32).flatten()
        self.normals = np.array(norms, dtype=np.float32).flatten()
        self.indices = np.array(idxs, dtype=np.uint16)

    def getVerticeList(self) -> VAOMesh:
        """Create a VAOMesh for this sphere geometry."""
        return VAOMesh(self.vertices, self.indices, self.normals)


def capsule_render_matrices(
    p0_world: np.ndarray, p1_world: np.ndarray, radius: float
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Compute model matrices for rendering a capsule as cylinder + two sphere caps.

    Returns:
        (cylinder_model, sphere0_model, sphere1_model) — three 4x4 matrices.
        The cylinder is scaled by (radius, radius, length) along the capsule axis.
        Each sphere is placed at an endpoint with uniform radius scaling.
    """
    mid = 0.5 * (p0_world + p1_world)
    axis = p1_world - p0_world
    length = float(np.linalg.norm(axis))

    # sphere matrices: uniform scale at each endpoint
    S_sphere = np.eye(4, dtype=np.float32)
    S_sphere[0, 0] = S_sphere[1, 1] = S_sphere[2, 2] = radius
    T0 = np.eye(4, dtype=np.float32)
    T0[:3, 3] = p0_world
    T1 = np.eye(4, dtype=np.float32)
    T1[:3, 3] = p1_world
    sphere0 = (T0 @ S_sphere).astype(np.float32)
    sphere1 = (T1 @ S_sphere).astype(np.float32)

    if length < 1e-10:
        # degenerate: both spheres at same point, no cylinder needed
        return None, sphere0, sphere1

    # rotation that maps Z-hat to capsule axis direction
    z_hat = axis / length
    if abs(z_hat[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])
    x_hat = np.cross(up, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)

    R = np.eye(4, dtype=np.float32)
    R[:3, 0] = x_hat
    R[:3, 1] = y_hat
    R[:3, 2] = z_hat

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = mid

    S = np.eye(4, dtype=np.float32)
    S[0, 0] = radius
    S[1, 1] = radius
    S[2, 2] = length

    cylinder = (T @ R @ S).astype(np.float32)
    return cylinder, sphere0, sphere1


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


class TorqueArc:
    """A filled arc (annular sector) in the XY plane for visualizing joint torques.
    The arc spans from angle 0 to `sweep` radians, with inner radius `r_inner`
    and outer radius `r_outer`. The geometry is a triangle strip."""

    def __init__(self, sweep: float, r_inner: float = 0.06, r_outer: float = 0.09, segments: int = 32) -> None:
        if abs(sweep) < 1e-6:
            # degenerate arc — create empty geometry
            self.vertices = np.zeros(6, dtype=np.float32)
            self.indices = np.array([0, 1, 0], dtype=np.uint16)
            return
        n = max(2, int(abs(sweep) / (2 * np.pi) * segments) + 1)
        verts: list[float] = []
        idxs: list[int] = []
        for i in range(n + 1):
            angle = sweep * i / n
            c, s = float(np.cos(angle)), float(np.sin(angle))
            # inner vertex
            verts.extend([r_inner * c, r_inner * s, 0.0])
            # outer vertex
            verts.extend([r_outer * c, r_outer * s, 0.0])
            if i < n:
                base = i * 2
                # two triangles forming a quad
                idxs.extend([base, base + 1, base + 2])
                idxs.extend([base + 1, base + 3, base + 2])
        self.vertices = np.array(verts, dtype=np.float32)
        self.indices = np.array(idxs, dtype=np.uint16)

    def getVerticeList(self) -> VAOMesh:
        """Return a VAOMesh for this arc."""
        return VAOMesh(self.vertices, self.indices)


class GroundQuad:
    """A large quad used as a stencil mask for planar shadows."""

    def __init__(self, half_extent: float = 50.0, z: float = 0.0) -> None:
        e = half_extent
        self.vertices = np.array(
            [-e, -e, z, e, -e, z, e, e, z, -e, e, z],
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

        self.vertices: np.ndarray = self.mesh.vertices.copy()
        self.vertices[:, 0] *= scaling[0]
        self.vertices[:, 1] *= scaling[1]
        self.vertices[:, 2] *= scaling[2]

        normals = np.array(self.mesh.vertex_normals)
        faces = np.array(self.mesh.faces)

        # fix face winding and normals when scale has negative components (mirroring)
        if np.prod(scaling) < 0:
            faces = np.asarray(faces[:, ::-1])
            # flip normals for mirrored axes
            for ax in range(3):
                if scaling[ax] < 0:
                    normals[:, ax] *= -1

        self.normals = np.asarray(normals.reshape(-1), dtype=np.float32)
        self.faces = np.asarray(faces.reshape(-1), dtype=np.uint16)
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
            # skip drag events after a press: pyglet may fire spurious drags
            # with large deltas from the cursor warp on exclusive mouse lock
            self._skip_drag_count = 0

        def on_key_press(self, symbol, modifiers):
            self.pressed[symbol] = True

        def on_key_release(self, symbol, modifiers):
            self.pressed[symbol] = False

        def on_mouse_press(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self._skip_drag_count = 2
                self._window.set_exclusive_mouse(True)

        def on_mouse_release(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self._window.set_exclusive_mouse(False)

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                if self._skip_drag_count > 0:
                    self._skip_drag_count -= 1
                    return
                # clamp to reject spurious large deltas from cursor warps
                max_delta = 40
                if abs(dx) > max_delta or abs(dy) > max_delta:
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

    @property
    def keys_pressed(self) -> collections.defaultdict:
        """Expose held-key state for use outside the camera (e.g. frame stepping)."""
        return self.__input_handler.pressed

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
    "dark world": {
        "ambient": _f32([0.12, 0.12, 0.12, 1.0]),
        "diffuse": _f32([0.15, 0.15, 0.15, 1.0]),
        "specular": _f32([0.02, 0.02, 0.02, 1.0]),
        "shininess": 0.03 * 128.0,
        "emission": _f32([0.02, 0.02, 0.02, 1.0]),
    },
    "collision red": {
        "ambient": _f32([0.3, 0.02, 0.02, 1.0]),
        "diffuse": _f32([0.8, 0.1, 0.1, 1.0]),
        "specular": _f32([0.3, 0.1, 0.1, 1.0]),
        "shininess": 0.1 * 128.0,
        "emission": _f32([0.3, 0.0, 0.0, 1.0]),
    },
}


# ── Visualizer ────────────────────────────────────────────────────────────────


class Visualizer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.dark_mode = _is_dark_mode()

        # theme colors
        if self.dark_mode:
            self.grid_color = (0.25, 0.25, 0.25, 1.0)
            self.coord_color = (0.4, 0.4, 0.4, 1.0)
            self.label_color = (200, 200, 200, 220)
        else:
            self.grid_color = (0.6, 0.6, 0.6, 1.0)
            self.coord_color = (0.6, 0.6, 0.6, 1.0)
            self.label_color = (0, 0, 0, 220)

        # some vars
        self.window_closed = False
        self.mode = "b"  # 'b' - blocking or 'c' - continous
        self.display_index = 0  # current index for displaying e.g. postures from file
        self.display_max = 1
        self.config = config

        # keep a list of bodies
        self.bodies: list[dict[str, Any]] = []

        # mesh display mode: "visual", "collision", or "boxes"
        self.mesh_mode = "boxes"
        self.show_meshes = False  # legacy compat, updated by mesh_mode

        self.angles: list[float] | None = None
        self.trajectory: Trajectory | None = None
        self.playing_traj = False  # currently playing or not
        self.playable = False  # can the trajectory be "played"
        self.freq = 1  # frequency in Hz of position / angle data

        # torque visualization
        self.torque_rings: list[dict[str, Any]] = []  # per-frame torque ring data
        self.show_torque_rings = False
        self.has_torque_data = False  # set to True when torque data is available

        # collision highlighting
        self.colliding_links: set[str] = set()
        self.check_collisions = True  # toggled with C key

        # additional callbacks to be used with key handling
        self.event_callback: Callable | None = None
        self.timer_callback = self.next_frame
        self._playback_start_time: float = 0.0
        self._playback_start_index: int = 0
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
        font_color = "color='#cccccc'" if self.dark_mode else ""
        self._help_font_color = font_color
        self._help_move_keys = move_keys
        self._help_enter_key = enter_key
        self.help_label: pyglet.text.HTMLLabel | None = None
        self._update_help_label()
        self.info_label = pyglet.text.Label(
            "",
            font_name="Helvetica",
            font_size=11,
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            color=self.label_color,
        )
        self._last_frame_time: float = 0.0
        self._frame_times: collections.deque[float] = collections.deque(maxlen=30)
        # cache pyglet label matrices so we don't rebuild them every frame
        self._label_projection = pyglet.math.Mat4.orthogonal_projection(0, self.width, 0, self.height, -1, 500)
        self._label_view = pyglet.math.Mat4()
        self.updateLabels()

    def updateLabels(self):
        self.info_label.text = f"Index: {self.display_index}"

    def _update_help_label(self) -> None:
        """Rebuild help label with current toggle states."""
        fc = self._help_font_color
        chk_coll = "[on]" if self.check_collisions else "[off]"
        chk_torq = "[on]" if self.show_torque_rings else "[off]"
        html = (
            f'<font face="Helvetica,Arial" size=15 {fc}>'
            f"wasd, {self._help_move_keys} - move around <br/>"
            f"mouse drag - look / shift+drag - orbit <br/>"
            f"{self._help_enter_key} - play/stop trajectory <br/>"
            f"&#x2190; &#x2192; - prev/next frame <br/>"
            f"m - cycle visual/collision/capsules/boxes <br/>"
            f"c - collision checking {chk_coll} <br/>"
            f"t - show joint torques {chk_torq} <br/>"
            f"b - continuous/blocking (for optimizer) <br/>"
            f"q - close <br/>"
            f"</font>"
        )
        self.help_label = pyglet.text.HTMLLabel(
            html,
            x=10,
            y=-10,
            width=300,
            multiline=True,
            anchor_x="left",
            anchor_y="bottom",
        )

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

        # frame stepping while arrow keys are held
        if not hasattr(self, "_step_accum"):
            self._step_accum = 0.0
        step_interval = 1.0 / 50.0  # frames per second when holding arrow keys
        pressed = self.camera.keys_pressed
        stepping = False
        if pressed[key.RIGHT]:
            stepping = True
            self._step_accum += dt if dt else 0.0
            while self._step_accum >= step_interval:
                self._step_accum -= step_interval
                self.display_index += 1
                if self.display_index >= self.display_max:
                    self.display_index = 0
        elif pressed[key.LEFT]:
            stepping = True
            self._step_accum += dt if dt else 0.0
            while self._step_accum >= step_interval:
                self._step_accum -= step_interval
                self.display_index -= 1
                if self.display_index < 0:
                    self.display_index = self.display_max - 1
        if not stepping:
            self._step_accum = 0.0
        elif self.event_callback:
            self.event_callback()

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
        if self.dark_mode:
            gl.glClearColor(0.12, 0.12, 0.14, 0)
        else:
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
        self.cylinder_vao = CylinderGeom().getVerticeList()
        self.sphere_vao = SphereGeom().getVerticeList()
        self.coord_vao = Coord().getVerticeList()
        self.grid_vao = Grid().getVerticeList()
        self.ground_quad_vao = GroundQuad().getVerticeList()

        # precompute shadow projection matrix (light → z=0 plane)
        self.shadow_proj = shadow_matrix(self.light_pos, self.ground_plane)

        # fill later
        self.mesh_vaos: dict[str, VAOMesh] = {}
        self.collision_mesh_vaos: dict[str, VAOMesh] = {}

        # capsule collision data (set via loadCapsules)
        self._capsule_data: dict[str, Any] = {}  # link_name -> Capsule

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

        if symbol == key.B:
            if self.mode == "b":
                print("switching to continuous render")
                self.mode = "c"
                pyglet.app.exit()
                return pyglet.event.EVENT_HANDLED
            else:
                print("switching to blocking render")
                self.mode = "b"

        if symbol == key.C:
            self.check_collisions = not self.check_collisions
            if not self.check_collisions:
                self.colliding_links = set()
            self._update_help_label()
            if self.event_callback:
                self.event_callback()

        if symbol == key.I:
            print(f"Camera pos:{self.camera.position} pitch:{self.camera.pitch} yaw:{self.camera.yaw}")

        if symbol == key.R:
            print("Reset camera")
            self._initCamera()

        if symbol == key.M:
            # cycle: visual → collision → capsules → boxes → visual ...
            if self.mesh_mode == "visual":
                if len(self.collision_mesh_vaos):
                    self.mesh_mode = "collision"
                    self.show_meshes = True
                elif self._capsule_data:
                    self.mesh_mode = "capsules"
                    self.show_meshes = False
                else:
                    self.mesh_mode = "boxes"
                    self.show_meshes = False
            elif self.mesh_mode == "collision":
                if self._capsule_data:
                    self.mesh_mode = "capsules"
                    self.show_meshes = False
                else:
                    self.mesh_mode = "boxes"
                    self.show_meshes = False
            elif self.mesh_mode == "capsules":
                self.mesh_mode = "boxes"
                self.show_meshes = False
            else:
                if len(self.mesh_vaos):
                    self.mesh_mode = "visual"
                    self.show_meshes = True
                elif len(self.collision_mesh_vaos):
                    self.mesh_mode = "collision"
                    self.show_meshes = True
                elif self._capsule_data:
                    self.mesh_mode = "capsules"
                    self.show_meshes = False
            if self.event_callback:
                self.event_callback()

        if symbol == key.T:
            if self.has_torque_data:
                self.show_torque_rings = not self.show_torque_rings
                self._update_help_label()
                if self.event_callback:
                    self.event_callback()

        if symbol == key.RIGHT:
            if self.display_index < self.display_max - 1:
                self.display_index += 1
                self._step_accum = 0.0  # reset so holding starts fresh
                if self.event_callback:
                    self.event_callback()

        if symbol == key.LEFT:
            if self.display_index > 0:
                self.display_index -= 1
                self._step_accum = 0.0
                if self.event_callback:
                    self.event_callback()

        if symbol == key.ENTER:
            if not self.playing_traj and self.playable:
                self.playing_traj = True
                self._playback_start_time = time.perf_counter()
                self._playback_start_index = self.display_index
                # schedule at render rate; next_frame uses wall-clock time to
                # compute the correct display_index regardless of timer frequency
                pyglet.clock.schedule_interval(self.timer_callback, 1 / self.render_fps)
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
        gl.glUniform4f(u_unlit["uColor"], *self.grid_color)
        self.grid_vao.draw(gl.GL_LINES)

        # coordinate axes for each body
        for b in self.bodies:
            coord_mvp = vp @ b["base_model"]
            gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, coord_mvp)
            gl.glUniform4f(u_unlit["uColor"], *self.coord_color)
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

        # ── Torque rings (unlit, blended, always on top) ────────────────────
        if self.show_torque_rings and self.torque_rings:
            gl.glUseProgram(self.unlit_shader)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            for ring in self.torque_rings:
                mvp = proj @ view @ ring["model"]
                gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, mvp)
                gl.glUniform4f(u_unlit["uColor"], *ring["color"])
                ring["vao"].draw(gl.GL_TRIANGLES)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_BLEND)

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
        # z-offset to avoid z-fighting between shadows and ground geometry
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(-2.0, -2.0)
        gl.glUniform4f(u_unlit["uColor"], 0.0, 0.0, 0.0, 0.3)
        for b in self.bodies:
            # world objects (ground, etc.) don't cast shadows
            if b.get("world", False):
                continue
            if b["geometry"] == "box":
                shadow_model = self.shadow_proj @ b["model"]
            else:
                shadow_model = self.shadow_proj @ b["base_model"]
            shadow_mvp = vp @ shadow_model
            gl.glUniformMatrix4fv(u_unlit["uMVP"], 1, gl.GL_TRUE, shadow_mvp)
            if b["geometry"] == "box":
                self.cube_vao.draw(gl.GL_TRIANGLES)
            elif b["geometry"] == "mesh":
                shadow_mesh_dict = self.collision_mesh_vaos if b.get("mesh_source") == "collision" else self.mesh_vaos
                if b["name"] in shadow_mesh_dict:
                    shadow_mesh_dict[b["name"]].draw(gl.GL_TRIANGLES)

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

        if self.help_label is not None:
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
        elif body["geometry"] == "capsule":
            self._upload_material(body["material"])
            # draw cylinder shaft + two sphere caps as three draw calls
            for m_key, vao in [
                ("capsule_cylinder", self.cylinder_vao),
                ("capsule_sphere0", self.sphere_vao),
                ("capsule_sphere1", self.sphere_vao),
            ]:
                model = body[m_key]
                if model is None:
                    continue  # degenerate capsule (sphere only), skip cylinder
                mv = view @ model
                mvp = proj @ mv
                # normal matrix from T*R*S: columns of R*S normalized = R * S^{-1}
                rs = model[:3, :3]
                scale_sq = np.sum(rs**2, axis=0)
                normal_mat = view[:3, :3] @ (rs / scale_sq)
                gl.glUniformMatrix4fv(u["uMVP"], 1, gl.GL_TRUE, mvp)
                gl.glUniformMatrix4fv(u["uMV"], 1, gl.GL_TRUE, mv)
                gl.glUniformMatrix3fv(u["uNormalMat"], 1, gl.GL_TRUE, np.ascontiguousarray(normal_mat))
                vao.draw(gl.GL_TRIANGLES)
        elif body["geometry"] == "mesh":
            # meshes use base_model (already scaled at load time, no extra S)
            mv = view @ body["base_model"]
            mvp = proj @ mv
            normal_mat = view[:3, :3] @ body["rot3"]  # S=I for meshes
            gl.glUniformMatrix4fv(u["uMVP"], 1, gl.GL_TRUE, mvp)
            gl.glUniformMatrix4fv(u["uMV"], 1, gl.GL_TRUE, mv)
            gl.glUniformMatrix3fv(u["uNormalMat"], 1, gl.GL_TRUE, np.ascontiguousarray(normal_mat))
            self._upload_material(body["material"])
            mesh_dict = self.collision_mesh_vaos if body.get("mesh_source") == "collision" else self.mesh_vaos
            mesh_dict[body["name"]].draw(gl.GL_TRIANGLES)

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
        ground_z = 0.0
        world_material = "dark world" if self.dark_mode else "white rubber"
        for linkName in boxes:
            body: dict[str, Any] = {}
            body["geometry"] = "box"
            body["material"] = world_material
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
            body["world"] = True
            self._precompute_body_model(body)
            self.bodies.append(body)
            # track the top surface of the lowest ground-like box
            top_z = float(body["position"][2]) + float(body["center"][2]) + 0.5 * float(body["size3"][2])
            ground_z = min(ground_z, top_z)
        self.setGroundHeight(ground_z)

    def setGroundHeight(self, z: float) -> None:
        """Update shadow ground plane and stencil quad to the given z height.
        A small offset places shadows just below the ground surface to avoid z-fighting."""
        shadow_z = z + 0.001
        self.ground_plane = np.array([0.0, 0.0, 1.0, -shadow_z], dtype=np.float32)
        self.shadow_proj = shadow_matrix(self.light_pos, self.ground_plane)
        self.ground_quad_vao = GroundQuad(z=shadow_z).getVerticeList()

    def setModelTrajectory(self, trajectory):
        self.trajectory = trajectory

    def next_frame(self, dt):
        # advance based on wall-clock time so playback matches real time
        now = time.perf_counter()
        elapsed = now - self._playback_start_time
        new_index = self._playback_start_index + int(elapsed * self.playback_rate)
        if new_index >= self.display_max:
            new_index = 0
            self._playback_start_time = now
            self._playback_start_index = 0
        self.display_index = new_index
        if self.event_callback is not None:
            self.event_callback()

    def loadCapsules(self, capsules: dict[str, Any]) -> None:
        """Store capsule collision data for visualization.

        Args:
            capsules: Dictionary mapping link name to Capsule objects.
        """
        self._capsule_data = capsules

    def loadMeshes(self, urdfpath, linkNames, urdfHelpers, use_convex_hull: bool = False):
        """Load visual and collision meshes for all links.

        Collision meshes match what the optimizer uses: either the raw collision
        mesh (when useConvexHullCollision is off) or its convex hull (when on).
        Falls back to visual mesh (always convex-hulled) when no collision mesh exists.
        """
        import trimesh

        if not len(self.mesh_vaos):
            for i in range(0, len(linkNames)):
                # visual meshes
                filename = urdfHelpers.getMeshPath(urdfpath, linkNames[i])
                if filename and os.path.exists(filename):
                    scale_parts = urdfHelpers.mesh_scaling.split(" ")
                    scale = np.array([float(scale_parts[0]), float(scale_parts[1]), float(scale_parts[2])])
                    self.mesh_vaos[linkNames[i]] = Mesh(filename, scale).getVerticeList()

                # collision meshes: try collision mesh first, fall back to visual mesh
                coll_filename = urdfHelpers.getCollisionMeshPath(urdfpath, linkNames[i])
                is_collision_mesh = coll_filename is not None and os.path.exists(coll_filename)
                if not is_collision_mesh:
                    coll_filename = urdfHelpers.getMeshPath(urdfpath, linkNames[i])
                if coll_filename and os.path.exists(coll_filename):
                    scale_parts = urdfHelpers.mesh_scaling.split(" ")
                    scale = np.array([float(scale_parts[0]), float(scale_parts[1]), float(scale_parts[2])])
                    mesh = trimesh.load_mesh(coll_filename)
                    verts = np.array(mesh.vertices) * scale
                    faces = np.array(mesh.faces)
                    if np.prod(scale) < 0:
                        faces = np.asarray(faces[:, ::-1])
                    # convex hull when configured, or for visual mesh fallback
                    if use_convex_hull or not is_collision_mesh:
                        hull = trimesh.Trimesh(vertices=verts, faces=faces).convex_hull
                        verts = np.array(hull.vertices)
                        faces = np.array(hull.faces)
                        normals = np.array(hull.vertex_normals)
                    else:
                        normals = np.array(mesh.vertex_normals)
                        # apply same scale to normals for mirrored axes
                        for ax in range(3):
                            if scale[ax] < 0:
                                normals[:, ax] *= -1
                    render_verts = np.asarray(verts.reshape(-1), dtype=np.float32)
                    render_normals = np.asarray(normals.reshape(-1), dtype=np.float32)
                    render_faces = np.asarray(faces.reshape(-1), dtype=np.uint16)
                    self.collision_mesh_vaos[linkNames[i]] = VAOMesh(render_verts, render_faces, render_normals)
            if len(self.mesh_vaos):
                self.mesh_mode = "visual"
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
            is_colliding = n_name in self.colliding_links
            active_meshes = self.mesh_vaos if self.mesh_mode == "visual" else self.collision_mesh_vaos
            if self.mesh_mode == "capsules" and n_name in self._capsule_data:
                # capsule mode: render capsule primitives
                cap = self._capsule_data[n_name]
                t = kinDyn.getWorldTransform(l)
                rot = t.getRotation().toNumPy()
                pos = t.getPosition().toNumPy()
                p0_world = rot @ cap.p0_local + pos
                p1_world = rot @ cap.p1_local + pos
                body["geometry"] = "capsule"
                body["material"] = "collision red" if is_colliding else "white rubber"
                cyl_m, sph0_m, sph1_m = capsule_render_matrices(p0_world, p1_world, cap.radius)
                body["capsule_cylinder"] = cyl_m
                body["capsule_sphere0"] = sph0_m
                body["capsule_sphere1"] = sph1_m
                # set dummy fields expected by _precompute_body_model
                body["size3"] = [1.0, 1.0, 1.0]
                body["center"] = [0.0, 0.0, 0.0]
                body["position"] = pos
                rpy = t.getRotation().asRPY()
                body["rotation"] = [rpy.getVal(0), rpy.getVal(1), rpy.getVal(2)]
            elif self.show_meshes and n_name in active_meshes:
                body["geometry"] = "mesh"
                body["mesh_source"] = self.mesh_mode
                body["size3"] = [1.0, 1.0, 1.0]
                body["center"] = [0.0, 0.0, 0.0]
                body["material"] = "collision red" if is_colliding else "metal"
            elif n_name in boxes:
                body["geometry"] = "box"
                body["material"] = "collision red" if is_colliding else "white rubber"
                b = np.array(boxes[n_name][0]) * self.config["scaleCollisionHull"]
                p = np.array(boxes[n_name][1])
                body["size3"] = np.array([b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]])
                body["center"] = 0.5 * (b[0] + b[1]) + p
            else:
                # no mesh and no bounding box (e.g. sensor frames), skip this link
                continue

            if "position" not in body:
                t = kinDyn.getWorldTransform(l)
                body["position"] = t.getPosition().toNumPy()
                rpy = t.getRotation().asRPY()
                body["rotation"] = [rpy.getVal(0), rpy.getVal(1), rpy.getVal(2)]

            if "transparentLinks" in self.config and n_name in self.config["transparentLinks"]:
                body["transparent"] = True

            self._precompute_body_model(body)
            self.bodies.append(body)

    def setTorqueRings(
        self,
        kinDyn: Any,
        torques: np.ndarray,
        joint_names: list[str],
        joint_axes: dict[str, list[float]],
        torque_limits: dict[str, float],
    ) -> None:
        """Build torque ring data for the current frame.

        For each joint, creates up to one arc: green filling clockwise for positive
        torque, orange filling counter-clockwise for negative. A full circle corresponds
        to the joint's maximum torque."""
        self.torque_rings = []
        for i, jname in enumerate(joint_names):
            if jname not in torque_limits or torque_limits[jname] == 0:
                continue
            tau = float(torques[i])
            tau_max = torque_limits[jname]
            ratio = np.clip(tau / tau_max, -1.0, 1.0)
            if abs(ratio) < 0.01:
                continue

            # sweep angle: positive torque → positive sweep (green),
            # negative → negative sweep (orange)
            sweep = float(ratio * 2 * np.pi)

            arc_vao = TorqueArc(sweep).getVerticeList()

            # get joint world transform (use child link frame)
            j_idx = kinDyn.model().getJointIndex(jname)
            j = kinDyn.model().getJoint(j_idx)
            child_link = j.getSecondAttachedLink()
            t = kinDyn.getWorldTransform(child_link)
            pos = t.getPosition().toNumPy()
            rot = t.getRotation().toNumPy()

            # build rotation to align arc's Z axis with the joint axis (in local frame)
            axis = np.array(joint_axes.get(jname, [0.0, 0.0, 1.0]), dtype=np.float64)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            # rotation from Z to axis: R such that R @ [0,0,1] = axis
            z = np.array([0.0, 0.0, 1.0])
            if np.allclose(axis, z):
                axis_rot = np.eye(3)
            elif np.allclose(axis, -z):
                axis_rot = np.diag([1.0, -1.0, -1.0])
            else:
                v = np.cross(z, axis)
                s = np.linalg.norm(v)
                c = np.dot(z, axis)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                axis_rot = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

            # combine: world rotation of link * local axis alignment
            full_rot = rot @ axis_rot

            # build model matrix (translation + rotation, no scale — arc is pre-sized)
            model = np.eye(4, dtype=np.float32)
            model[:3, :3] = full_rot
            model[:3, 3] = pos

            color = [0.0, 0.8, 0.1, 0.7] if tau > 0 else [1.0, 0.5, 0.0, 0.7]
            self.torque_rings.append({"vao": arc_vao, "model": model, "color": color})

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
        # skip links without any visual geometry (e.g. sensor frames)
        if not urdfHelpers.hasVisualGeometry(args.model, link_name):
            continue
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

    # set up collision checker
    from identification.collision import CollisionChecker

    collision_checker = CollisionChecker(
        urdf_helpers=urdfHelpers,
        urdf_file=args.model,
        link_cuboid_hulls=link_cuboid_hulls,
        link_names=linkNames,
        scale_collision_hull=config.get("scaleCollisionHull", 1.0),
        use_convex_hull=config.get("useConvexHullCollision", False),
    )
    neighbors = URDFHelpers.getNeighbors(loader.model())

    v = Visualizer(config)

    v.loadMeshes(
        args.model,
        linkNames,
        urdfHelpers,
        use_convex_hull=config.get("useConvexHullCollision", False),
    )

    # fit capsule collision primitives for visualization
    from excitation.capsule import fit_capsules_from_urdf

    capsules = fit_capsules_from_urdf(
        args.model,
        linkNames,
        urdfHelpers,
        radius_scale=config.get("scaleCapsuleRadius", 1.0),
    )
    if capsules:
        v.loadCapsules(capsules)

    # prepare torque visualization data
    joint_axes = URDFHelpers.getJointAxes(args.model)
    joint_names = g_model.jointNames
    torque_limits = {jn: g_model.limits[jn]["torque"] for jn in joint_names if jn in g_model.limits}
    torque_data: np.ndarray | None = None

    if args.trajectory:
        # display trajectory
        data = np.load(args.trajectory, encoding="latin1", allow_pickle=True)
        if "angles" in data:
            data_type = "static"
        elif "positions" in data:
            data_type = "measurements"
            v.playable = True
        else:
            data_type = "trajectory"
            v.playable = True

        # check if torque data is available
        if "torques" in data:
            torque_data = data["torques"]
            # floating-base: first 6 columns are base wrench, skip them
            if torque_data.shape[1] > n_dof:
                torque_data = torque_data[:, torque_data.shape[1] - n_dof :]
            v.has_torque_data = True
            print("Torque data found")
        else:
            print("No torque data in file")
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

        # check collisions using the same code path the optimizer would use:
        # capsule distance when viewing capsules, FCL when viewing meshes/boxes
        if not v.check_collisions:
            pass  # collision checking disabled (toggle with C key)
        elif v.mesh_mode == "capsules" and capsules:
            from excitation.capsule import find_colliding_links_capsule

            v.colliding_links = find_colliding_links_capsule(
                capsules,
                kinDyn,
                linkNames,
                ignore_links=set(config["ignoreLinksForCollision"]),
                ignore_pairs=config.get("ignoreLinkPairsForCollision", []),
                neighbors=neighbors,
                max_kin_distance=config.get("collisionMaxKinematicDistance", 0),
            )
        else:
            v.colliding_links = collision_checker.find_colliding_links(
                kinDyn,
                linkNames,
                ignore_links=set(config["ignoreLinksForCollision"]),
                ignore_pairs=config.get("ignoreLinkPairsForCollision", []),
                neighbors=neighbors,
                max_kin_distance=config.get("collisionMaxKinematicDistance", 0),
                use_visual_mesh=(v.mesh_mode == "visual"),
            )
        v.addIDynTreeModel(kinDyn, link_cuboid_hulls, linkNames, ignore_links=[])

        if args.world:
            v.addWorld(world_boxes)

        # update torque rings if enabled and data is available
        if v.show_torque_rings and torque_data is not None:
            if data_type == "measurements":
                idx = int(v.display_index * v.freq / v.playback_rate)
                idx = min(idx, torque_data.shape[0] - 1)
                tau = torque_data[idx, :]
            else:
                tau = np.zeros(n_dof)
            v.setTorqueRings(kinDyn, tau, joint_names, joint_axes, torque_limits)
        else:
            v.torque_rings = []

        v.updateLabels()

    if args.trajectory:
        if data_type == "static":
            v.display_max = len(data["angles"])  # number of postures
        elif data_type == "trajectory":
            trajectory = PulsedTrajectory(n_dof, use_deg=data["use_deg"])
            jl = [tuple(row) for row in data["joint_limits"]] if "joint_limits" in data else None
            trajectory.initWithParams(data["a"], data["b"], data["q"], data["nf"], data["wf"], joint_limits=jl)
            v.setModelTrajectory(trajectory)

            v.freq = config["excitationFrequency"]
            v.playback_rate = v.freq
            v.display_max = int(trajectory.getPeriodLength() * v.playback_rate)  # length of trajectory
        elif data_type == "measurements":
            v.freq = config["excitationFrequency"]
            v.playback_rate = v.freq
            v.display_max = data["positions"].shape[0]

    v.event_callback = draw_model
    v.event_callback()
    v.run()
