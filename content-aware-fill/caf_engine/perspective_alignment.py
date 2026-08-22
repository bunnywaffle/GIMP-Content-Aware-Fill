#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Perspective & Geometric Alignment
=========================================
Computes perspective coordinate projections and affine transformations:
- Converts Cartesian (x, y) coordinates into aligned perspective coordinates (u, v)
- Warps candidate source patches along vanishing direction vectors
- Eliminates shear distortion across tilted shelves and angled surfaces
"""

import math

class PerspectiveCoordinateSystem:
    def __init__(self, angle_rad=0.0, scale=1.0, vanishing_point=None):
        self.angle_rad = angle_rad
        self.scale = scale
        self.cos_a = math.cos(angle_rad)
        self.sin_a = math.sin(angle_rad)
        self.vanishing_point = vanishing_point

    def to_perspective(self, x, y):
        """Converts image coordinates (x, y) to perspective-aligned coordinates (u, v)."""
        u = (x * self.cos_a + y * self.sin_a) * self.scale
        v = (-x * self.sin_a + y * self.cos_a) * self.scale
        return u, v

    def to_cartesian(self, u, v):
        """Converts perspective coordinates (u, v) back to image coordinates (x, y)."""
        inv_s = 1.0 / max(1e-4, self.scale)
        su = u * inv_s
        sv = v * inv_s
        x = su * self.cos_a - sv * self.sin_a
        y = su * self.sin_a + sv * self.cos_a
        return int(round(x)), int(round(y))


def build_perspective_alignment(perspective_model):
    """
    Constructs an aligned perspective coordinate system from the perspective model.
    """
    dom_ang = perspective_model.dominant_angles[0] if perspective_model.dominant_angles else 0.0
    return PerspectiveCoordinateSystem(dom_ang, 1.0, perspective_model.vanishing_point)
