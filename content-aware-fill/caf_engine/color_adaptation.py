#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 11: Local Color & Exposure Adaptation
============================================
Performs local mean and variance color transfer:
- Preserves 100% of fine synthesized photo texture
- Normalizes global gain and bias without introducing artificial distance-ramp artifacts
"""

import math
import array

def adapt_patch_colors(work_img, src_img, mask_analysis, width, height, channels=4, patch_radius=4):
    """
    Subtly harmonizes average illumination across the synthesized hole.
    Preserves 100% of underlying photo texture contrast.
    """
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

    # Compute current synthesized hole boundary statistics
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

    # Subtle constant offset (no distance-ramp tent gradients)
    off_r = (mean_tr - mean_hr) * 0.5
    off_g = (mean_tg - mean_hg) * 0.5
    off_b = (mean_tb - mean_hb) * 0.5

    if abs(off_r) > 1.0 or abs(off_g) > 1.0 or abs(off_b) > 1.0:
        for x, y in hole_pixels:
            idx = y * width + x
            t_pix = idx * channels
            work_img[t_pix] = max(0, min(255, int(work_img[t_pix] + off_r + 0.5)))
            work_img[t_pix + 1] = max(0, min(255, int(work_img[t_pix + 1] + off_g + 0.5)))
            work_img[t_pix + 2] = max(0, min(255, int(work_img[t_pix + 2] + off_b + 0.5)))
            if channels == 4:
                work_img[t_pix + 3] = 255
