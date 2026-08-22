#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Debug Visualizer
========================
Generates diagnostic maps for inspecting pipeline stages:
- Line Segments & Perspective Vectors Map
- Planar Structural Zones Map
- Correspondence Deformation Flow Field Map
- Confidence & Seam Map
"""

import math
import array

def render_structure_zone_map(structure_geometry, width, height):
    """
    Renders planar structural zones as distinct false-color regions.
    """
    zone_map = structure_geometry.zone_map
    out_img = bytearray(width * height * 4)

    colors = [
        (40, 40, 40),    # Zone 0: Background
        (220, 100, 40),  # Zone 1: Left/Right Frame (Orange)
        (40, 140, 220),  # Zone 2: Shelf Bar (Blue)
        (50, 180, 80),   # Zone 3: Upper Books (Green)
        (180, 60, 200),  # Zone 4: Middle Books (Purple)
        (220, 200, 40),  # Zone 5: Lower Shelf / Floor (Yellow)
    ]

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            z = zone_map[idx] % len(colors)
            c = colors[z]
            pix = idx * 4
            out_img[pix] = c[0]
            out_img[pix + 1] = c[1]
            out_img[pix + 2] = c[2]
            out_img[pix + 3] = 255

    return out_img
