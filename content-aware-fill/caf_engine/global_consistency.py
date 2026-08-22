#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 9: Global Patch Consistency & Label Optimization
=======================================================
Enforces spatial coherence across adjacent target patches:
- Minimizes pairwise displacement jumps between neighbors
- Iterative local energy minimization (Markov Random Field relaxation)
- Suppresses isolated discordant patch selections
"""

import math
import array

def optimize_global_consistency(
    nnf_field,
    work_img,
    src_img,
    mask_grid,
    width,
    height,
    channels=4,
    patch_radius=4,
    source_selection=None,
    num_passes=2
):
    """
    Iteratively refines the NNF to minimize total energy:
    E(NNF) = sum_p D(T_p, S_p) + lambda * sum_{(p,q)} || (S_p - p) - (S_q - q) ||^2
    """
    total = width * height
    r = patch_radius
    nnf_x = nnf_field.nnf_x
    nnf_y = nnf_field.nnf_y
    valid_mask = source_selection.valid_mask if source_selection else bytearray([1]*total)

    hole_pixels = [ (x, y) for y in range(height) for x in range(width) if mask_grid[y * width + x] == 1 ]
    if not hole_pixels:
        return

    # 1D byte offsets
    grid_offsets = [
        (-r * width - r) * channels, (0) * channels, (r * width + r) * channels,
        (-r) * channels, (r) * channels
    ]

    def patch_dist(t_byte, s_byte):
        ssd = 0
        for off in grid_offsets:
            tp = t_byte + off
            sp = s_byte + off
            dr = work_img[tp] - src_img[sp]
            dg = work_img[tp + 1] - src_img[sp + 1]
            db = work_img[tp + 2] - src_img[sp + 2]
            ssd += dr * dr + dg * dg + db * db
        return ssd

    smoothness_weight = 0.35

    for _ in range(num_passes):
        for x, y in hole_pixels:
            if not (r <= x < width - r and r <= y < height - r):
                continue
            idx = y * width + x
            t_byte = idx * channels
            curr_sx = nnf_x[idx]
            curr_sy = nnf_y[idx]

            curr_data_cost = patch_dist(t_byte, (curr_sy * width + curr_sx) * channels)

            # Test candidate sources from 4-neighbors
            best_sx, best_sy = curr_sx, curr_sy
            best_energy = curr_data_cost

            # Compute neighbor smoothness for current
            smooth_curr = 0.0
            for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = x + ndx
                ny = y + ndy
                if 0 <= nx < width and 0 <= ny < height and mask_grid[ny * width + nx] == 1:
                    n_idx = ny * width + nx
                    nsx = nnf_x[n_idx]
                    nsy = nnf_y[n_idx]
                    # Displacement error
                    disp_x = (curr_sx - x) - (nsx - nx)
                    disp_y = (curr_sy - y) - (nsy - ny)
                    smooth_curr += math.sqrt(disp_x * disp_x + disp_y * disp_y)

            best_energy += smoothness_weight * smooth_curr

            # Evaluate neighbor proposals
            for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = x + ndx
                ny = y + ndy
                if 0 <= nx < width and 0 <= ny < height and mask_grid[ny * width + nx] == 1:
                    n_idx = ny * width + nx
                    cand_sx = nnf_x[n_idx] - ndx
                    cand_sy = nnf_y[n_idx] - ndy

                    if r <= cand_sx < width - r and r <= cand_sy < height - r and valid_mask[cand_sy * width + cand_sx] == 1:
                        data_cost = patch_dist(t_byte, (cand_sy * width + cand_sx) * channels)
                        smooth_cost = 0.0
                        for odx, ody in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ox = x + odx
                            oy = y + ody
                            if 0 <= ox < width and 0 <= oy < height and mask_grid[oy * width + ox] == 1:
                                o_idx = oy * width + ox
                                osx = nnf_x[o_idx]
                                osy = nnf_y[o_idx]
                                disp_x = (cand_sx - x) - (osx - ox)
                                disp_y = (cand_sy - y) - (osy - oy)
                                smooth_cost += math.sqrt(disp_x * disp_x + disp_y * disp_y)

                        total_cand_energy = data_cost + smoothness_weight * smooth_cost
                        if total_cand_energy < best_energy:
                            best_energy = total_cand_energy
                            best_sx = cand_sx
                            best_sy = cand_sy

            nnf_x[idx] = best_sx
            nnf_y[idx] = best_sy
