#!/usr/bin/env python

"""Generate the FloBaRoID application icon (icon.png).

Draws a friendly robot face: rounded-square head, thick-ring eyes, smile arc,
antenna with green tip, and ear panels on a blue circular background.
Anti-aliased using signed distance fields. Uses only stdlib.

Usage:
  uv run tools/generate_icon.py [size]
"""

import math
import struct
import sys
import zlib
from pathlib import Path


def generate_icon(size: int = 128) -> bytes:
    """Generate a robot face icon and return raw PNG bytes."""
    pixels: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * size for _ in range(size)]
    s = size
    cx, cy = s / 2, s / 2

    # color palette
    bg = (52, 120, 246)  # blue background circle
    head_c = (210, 218, 228)  # silver-blue head
    head_dark = (180, 190, 205)  # slightly darker for ears
    eye_ring = (40, 45, 55)  # dark eye outline
    eye_fill = head_c  # eye interior matches head
    mouth_c = (60, 65, 80)  # dark mouth
    antenna_c = (180, 190, 205)  # antenna stick
    antenna_tip = (100, 210, 130)  # green antenna light

    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def aa(d: float, edge: float) -> float:
        """Anti-aliased coverage: smooth 1px transition at boundary."""
        return max(0.0, min(1.0, edge - d + 0.5))

    def rrect_d(px: float, py: float, rx: float, ry: float, hw: float, hh: float, cr: float) -> float:
        """Signed distance to a rounded rectangle centered at (rx, ry)."""
        qx = max(abs(px - rx) - (hw - cr), 0.0)
        qy = max(abs(py - ry) - (hh - cr), 0.0)
        return math.sqrt(qx * qx + qy * qy) - cr

    def blend(base_c: tuple[int, int, int], top_c: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
        return (
            int(base_c[0] + (top_c[0] - base_c[0]) * alpha),
            int(base_c[1] + (top_c[1] - base_c[1]) * alpha),
            int(base_c[2] + (top_c[2] - base_c[2]) * alpha),
        )

    # geometry — head fills ~70% of icon
    head_cy = cy + s * 0.06
    head_hw, head_hh = s * 0.33, s * 0.28  # half-width, half-height
    head_cr = s * 0.10  # corner radius

    # ears (small rounded rects on sides)
    ear_w, ear_h, ear_cr = s * 0.06, s * 0.14, s * 0.03

    # eyes (thick dark ring, small head-colored interior)
    eye_y = head_cy - s * 0.02
    eye_sep = s * 0.28  # wide apart
    eye_outer_r = s * 0.07  # large outer ring
    eye_inner_r = s * 0.03  # small inner hole

    # mouth: smile arc fitting between the eyes
    mouth_y = head_cy + s * 0.12
    mouth_arc_r = s * 0.16
    mouth_arc_cy = mouth_y - mouth_arc_r
    mouth_hw = s * 0.07  # narrow, fits between eyes
    mouth_thick = s * 0.02

    # antenna
    ant_base_y = head_cy - head_hh
    ant_top_y = ant_base_y - s * 0.14
    ant_w = s * 0.025
    ant_tip_r = s * 0.045

    for y in range(s):
        for x in range(s):
            px, py = x + 0.5, y + 0.5

            # background circle
            bg_a = aa(dist(px, py, cx, cy), s * 0.47)
            if bg_a <= 0:
                continue
            r, g, b = bg

            # antenna stick
            if abs(px - cx) < ant_w and ant_top_y - ant_tip_r < py < ant_base_y + s * 0.01:
                a_stick = aa(abs(px - cx), ant_w)
                r, g, b = blend((r, g, b), antenna_c, a_stick)

            # antenna tip (green light)
            a_tip = aa(dist(px, py, cx, ant_top_y), ant_tip_r)
            if a_tip > 0:
                r, g, b = blend((r, g, b), antenna_tip, a_tip)

            # ears (left and right)
            for ear_side in (-1, 1):
                ear_x = cx + ear_side * (head_hw + ear_w * 0.4)
                d_ear = rrect_d(px, py, ear_x, head_cy, ear_w / 2, ear_h / 2, ear_cr)
                ear_a = aa(d_ear, 0)
                if ear_a > 0:
                    r, g, b = blend((r, g, b), head_dark, ear_a)

            # head
            d_head = rrect_d(px, py, cx, head_cy, head_hw, head_hh, head_cr)
            ha = aa(d_head, 0)
            if ha > 0:
                r, g, b = blend((r, g, b), head_c, ha)

                # eyes (thick dark ring, head-colored interior)
                for eye_side in (-1, 1):
                    ex = cx + eye_side * eye_sep / 2
                    d_eye = dist(px, py, ex, eye_y)
                    eo_a = aa(d_eye, eye_outer_r)
                    if eo_a > 0:
                        r, g, b = blend((r, g, b), eye_ring, eo_a)
                        ei_a = aa(d_eye, eye_inner_r)
                        if ei_a > 0:
                            r, g, b = blend((r, g, b), eye_fill, ei_a)

                # mouth: arc stroke with rounded endcaps
                mouth_end_angle = math.asin(min(1.0, mouth_hw / mouth_arc_r))
                angle_to_pt = math.atan2(px - cx, py - mouth_arc_cy)
                if abs(angle_to_pt) <= mouth_end_angle:
                    d_arc = abs(dist(px, py, cx, mouth_arc_cy) - mouth_arc_r)
                else:
                    sign = 1.0 if angle_to_pt > 0 else -1.0
                    end_x = cx + sign * mouth_arc_r * math.sin(mouth_end_angle)
                    end_y = mouth_arc_cy + mouth_arc_r * math.cos(mouth_end_angle)
                    d_arc = dist(px, py, end_x, end_y)
                if py > mouth_arc_cy:
                    m_a = aa(d_arc, mouth_thick)
                    if m_a > 0:
                        r, g, b = blend((r, g, b), mouth_c, m_a)

            out_a = int(bg_a * 255)
            pixels[y][x] = (r, g, b, out_a)

    # encode as PNG
    raw_rows = b""
    for row in pixels:
        raw_rows += b"\x00"  # filter byte: none
        for rgba in row:
            raw_rows += struct.pack("BBBB", *rgba)

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw_rows))
    png += _png_chunk(b"IEND", b"")

    return png


if __name__ == "__main__":
    icon_size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    out_path = Path(__file__).parent.parent / "output" / "icon.png"
    out_path.write_bytes(generate_icon(icon_size))
    print(f"Saved {icon_size}x{icon_size} icon to {out_path}")
