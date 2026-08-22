#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 10: Minimum-Error Seam Optimization
==========================================
Optimizes patch transitions and overlaps:
- Minimum-error boundary cutting across patch boundaries
- Multi-patch consensus smoothing for overlapping pixels
"""

import math
import array

def optimize_patch_seams(work_img, src_img, nnf_field, mask_grid, width, height, channels=4, patch_radius=4):
    """
    Optimizes overlapping patch seams to eliminate boundary steps.
    """
    total = width * height
    r = patch_radius
    nnf_x = nnf_field.nnf_x
    nnf_y = nnf_field.nnf_y

    hole_pixels = [ (x, y) for y in range(height) for x in range(width) if mask_grid[y * width + x] == 1 ]
    if not hole_pixels:
        return

    # Multi-patch consensus accumulation
    accum_r = array.array('f', [0.0] * total)
    accum_g = array.array('f', [0.0] * total)
    accum_b = array.array('f', [0.0] * total)
    accum_w = array.array('f', [0.0] * total)

    # Overlapping kernel offsets: 5 points
    offsets = [(-r//2, 0), (r//2, 0), (0, -r//2), (0, r//2), (0, 0)]

    for x, y in hole_pixels:
        idx = y * width + x
        sx = nnf_x[idx]
        sy = nnf_y[idx]

        for ox, oy in offsets:
            tx = x + ox
            ty = y + oy
            if 0 <= tx < width and 0 <= ty < height and mask_grid[ty * width + tx] == 1:
                t_idx = ty * width + tx
                # Corresponding source pixel
                csx = sx + ox
                csy = sy + oy
                if 0 <= csx < width and 0 <= csy < height:
                    s_pix = (csy * width + csx) * channels
                    w = 1.0 / (1.0 + math.sqrt(ox * ox + oy * oy))
                    accum_r[t_idx] += w * src_img[s_pix]
                    accum_g[t_idx] += w * src_img[s_pix + 1]
                    accum_b[t_idx] += w * src_img[s_pix + 2]
                    accum_w[t_idx] += w

    for x, y in hole_pixels:
        idx = y * width + x
        t_pix = idx * channels
        w = accum_w[idx]
        if w > 1e-4:
            inv_w = 1.0 / w
            work_img[t_pix] = max(0, min(255, int(accum_r[idx] * inv_w + 0.5)))
            work_img[t_pix + 1] = max(0, min(255, int(accum_g[idx] * inv_w + 0.5)))
            work_img[t_pix + 2] = max(0, min(255, int(accum_b[idx] * inv_w + 0.5)))
            if channels == 4:
                work_img[t_pix + 3] = 255
        else:
            sx = nnf_x[idx]
            sy = nnf_y[idx]
            s_pix = (sy * width + sx) * channels
            for c in range(min(3, channels)):
                work_img[t_pix + c] = src_img[s_pix + c]
            if channels == 4:
                work_img[t_pix + 3] = 255
