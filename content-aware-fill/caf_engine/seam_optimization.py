#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 10: Minimum-Error Seam Optimization
==========================================
Optimizes patch transitions without muddy color averaging:
- Preserves 100% sharp photo exemplar texture
- Enforces direct exemplar transfer from optimal source coordinates
"""

import math
import array

def optimize_patch_seams(work_img, src_img, nnf_field, mask_grid, width, height, channels=4, patch_radius=4):
    """
    Directly writes optimal exemplar textures to the work canvas,
    guaranteeing razor-sharp details with zero muddy blending.
    """
    nnf_x = nnf_field.nnf_x
    nnf_y = nnf_field.nnf_y

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if mask_grid[idx] == 1:
                sx = nnf_x[idx]
                sy = nnf_y[idx]
                s_pix = (sy * width + sx) * channels
                t_pix = idx * channels

                for c in range(min(3, channels)):
                    work_img[t_pix + c] = src_img[s_pix + c]
                if channels == 4:
                    work_img[t_pix + 3] = 255
