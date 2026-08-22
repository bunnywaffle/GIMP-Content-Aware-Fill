#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 12: Gradient-Domain / Screened Poisson Blending
======================================================
Solves the harmonic Laplace/Poisson equation:
  nabla^2 Delta = 0, subject to Delta|_boundary = clamp(I_orig - I_synth)
- Seamlessly blends global lighting and exposure gradients
- Prevents cross-structural color bleeding between distinct image regions
- Preserves 100% of fine high-frequency synthesized texture
"""

import math
import array

def solve_poisson_residual_blending(
    work_img,
    src_img,
    mask_analysis,
    width,
    height,
    channels=4,
    num_iterations=8,
    progress_callback=None
):
    """
    Executes robust harmonic Poisson residual diffusion across the hole.
    Clamps extreme structural jumps to prevent cross-border color bleeding.
    """
    total = width * height
    boundary_pixels = mask_analysis.boundary_pixels
    hole_pixels = mask_analysis.hole_pixels
    hole_set = mask_analysis.hole_set

    if not boundary_pixels or not hole_pixels:
        return

    residual_r = array.array('f', [0.0] * total)
    residual_g = array.array('f', [0.0] * total)
    residual_b = array.array('f', [0.0] * total)

    # 1. Boundary Residual Conditions: Delta = clamp(I_orig - I_synth, -35, +35)
    for bx, by in boundary_pixels:
        b_idx = by * width + bx
        b_pix = b_idx * channels
        for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = bx + ndx
            ny = by + ndy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in hole_set:
                n_pix = (ny * width + nx) * channels
                diff_r = float(src_img[n_pix] - work_img[b_pix])
                diff_g = float(src_img[n_pix + 1] - work_img[b_pix + 1])
                diff_b = float(src_img[n_pix + 2] - work_img[b_pix + 2])

                # Clamp residuals to prevent cross-structural bleeds
                residual_r[b_idx] = max(-35.0, min(35.0, diff_r))
                residual_g[b_idx] = max(-35.0, min(35.0, diff_g))
                residual_b[b_idx] = max(-35.0, min(35.0, diff_b))
                break

    # 2. Multi-Pass Jacobi Relaxation: Delta(x, y) = 1/4 * sum_{neighbors} Delta(nx, ny)
    for it in range(num_iterations):
        for x, y in hole_pixels:
            idx = y * width + x
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0
            cnt = 0
            for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = x + ndx
                ny = y + ndy
                if 0 <= nx < width and 0 <= ny < height:
                    n_idx = ny * width + nx
                    sum_r += residual_r[n_idx]
                    sum_g += residual_g[n_idx]
                    sum_b += residual_b[n_idx]
                    cnt += 1
            if cnt > 0:
                residual_r[idx] = sum_r / cnt
                residual_g[idx] = sum_g / cnt
                residual_b[idx] = sum_b / cnt

    # 3. Additive Integration: I_final = I_synth + Delta
    for x, y in hole_pixels:
        idx = y * width + x
        t_pix = idx * channels
        work_img[t_pix] = max(0, min(255, int(work_img[t_pix] + residual_r[idx] + 0.5)))
        work_img[t_pix + 1] = max(0, min(255, int(work_img[t_pix + 1] + residual_g[idx] + 0.5)))
        work_img[t_pix + 2] = max(0, min(255, int(work_img[t_pix + 2] + residual_b[idx] + 0.5)))
        if channels == 4:
            work_img[t_pix + 3] = 255
