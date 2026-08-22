#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 6: Geometric Patch Transformations
=========================================
Implements geometric patch adaptation:
- Discrete candidate rotation angles (-30 deg to +30 deg)
- Discrete candidate scale factors (0.8x to 1.25x)
- Horizontal and vertical mirroring
- Precomputes transformed sampling grids
"""

import math

class PatchTransformation:
    def __init__(self, angle_deg=0.0, scale=1.0, mirror_h=False, mirror_v=False):
        self.angle_deg = angle_deg
        self.scale = scale
        self.mirror_h = mirror_h
        self.mirror_v = mirror_v
        self.rad = math.radians(angle_deg)
        self.cos_a = math.cos(self.rad)
        self.sin_a = math.sin(self.rad)

    def transform_offset(self, dx, dy):
        """Applies scale, mirroring, and rotation to a 2D patch offset."""
        mx = -dx if self.mirror_h else dx
        my = -dy if self.mirror_v else dy

        sx = mx * self.scale
        sy = my * self.scale

        rx = sx * self.cos_a - sy * self.sin_a
        ry = sx * self.sin_a + sy * self.cos_a

        return int(round(rx)), int(round(ry))


def get_standard_transformations(adaptation_mode="none"):
    """
    Returns candidate transformations based on user settings:
    - 'none': identity only
    - 'rotation': [-30, -15, 0, 15, 30]
    - 'scale': [0.8, 1.0, 1.25]
    - 'full': rotation + scale + mirroring
    """
    transforms = [PatchTransformation(0.0, 1.0, False, False)]

    if adaptation_mode == "none":
        return transforms

    if adaptation_mode in ("rotation", "full"):
        for ang in (-30.0, -15.0, 15.0, 30.0):
            transforms.append(PatchTransformation(ang, 1.0, False, False))

    if adaptation_mode in ("scale", "full"):
        for sc in (0.8, 1.25):
            transforms.append(PatchTransformation(0.0, sc, False, False))

    if adaptation_mode == "full":
        # Mirroring
        transforms.append(PatchTransformation(0.0, 1.0, True, False))
        transforms.append(PatchTransformation(0.0, 1.0, False, True))

    return transforms
