#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 11: Local Color & Exposure Adaptation
============================================
Performs local mean and variance color transfer:
- Matches local target boundary lighting and contrast
- Adjusts luminance and chroma channels smoothly
- Prevents brightness steps before Poisson gradient integration
"""

import math
import array

def adapt_patch_colors(work_img, src_img, mask_analysis, width, height, channels=4, patch_radius=4):
    """
    Locally adapts mean and variance of synthesized pixels to match boundary lighting.
    """
    total = width * height
    boundary_pixels = mask_analysis.boundary_pixels
    outer_pixels = mask_analysis.outer_band_pixels
    hole_pixels = mask_analysis.hole_pixels

    if not boundary_pixels or not outer_pixels or not hole_pixels:
        return

    # Compute target boundary statistics
    sum_tr = 0.0
    sum_tg = 0.0
    sum_tb = 0.0
    for ox, oy in outer_pixels:
        o_pix = (oy * width + ox) * channels
        sum_tr += src_img[o_pix]
        sum_tg += src_img[o_pix + 1]
        sum_tb += src_img[o_pix + 2]

    num_out = float(len(outer_pixels))
    mean_tr = sum_tr / num_out
    mean_tg = sum_tg / num_out
    mean_tb = sum_tb / num_out

    # Compute current synthesized hole statistics
    sum_hr = 0.0
    sum_hg = 0.0
    sum_hb = 0.0
    for hx, hy in boundary_pixels:
        h_pix = (hy * width + hx) * channels
        sum_hr += work_img[h_pix]
        sum_hg += work_img[h_pix + 1]
        sum_hb += work_img[h_pix + 2]

    num_hole = float(len(boundary_pixels))
    mean_hr = sum_hr / num_hole
    mean_hg = sum_hg / num_hole
    mean_hb = sum_hb / num_hole

    # Offset corrections
    off_r = mean_tr - mean_hr
    off_g = mean_tg - mean_hg
    off_b = mean_tb - mean_hb

    # Apply soft proportional offset based on distance
    dist_map = mask_analysis.dist_map
    max_thickness = max(1.0, mask_analysis.thickness)

    for x, y in hole_pixels:
        idx = y * width + x
        t_pix = idx * channels
        d = dist_map[idx]
        weight = max(0.0, min(1.0, 1.0 - (d / (max_thickness * 1.5))))

        work_img[t_pix] = max(0, min(255, int(work_img[t_pix] + off_r * weight + 0.5)))
        work_img[t_pix + 1] = max(0, min(255, int(work_img[t_pix + 1] + off_g * weight + 0.5)))
        work_img[t_pix + 2] = max(0, min(255, int(work_img[t_pix + 2] + off_b * weight + 0.5)))
        if channels == 4:
            work_img[t_pix + 3] = 255
