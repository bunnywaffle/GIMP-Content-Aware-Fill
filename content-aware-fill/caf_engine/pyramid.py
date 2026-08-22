#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4: Multi-Scale Image Pyramid
===================================
Constructs dynamic Gaussian image and mask pyramids:
- Dynamic scale level selection based on hole size
- Robust Gaussian downsampling with boundary support
- Multi-scale NNF projection from coarse to fine resolutions
"""

import math
import array

class PyramidLevel:
    def __init__(self, img_bytes, mask_bytes, width, height, channels, scale_factor):
        self.img_bytes = img_bytes
        self.mask_bytes = mask_bytes
        self.width = width
        self.height = height
        self.channels = channels
        self.scale_factor = scale_factor  # e.g. 1.0, 0.5, 0.25


def build_image_pyramid(img_bytes, mask_bytes, width, height, channels=4, max_hole_dim=100):
    """
    Constructs an image pyramid from coarse to fine.
    Returns list of PyramidLevel objects ordered [coarsest, ..., finest (Level 0)].
    """
    # Determine number of levels dynamically
    if max_hole_dim < 64 or width < 80 or height < 80:
        num_levels = 1
    elif max_hole_dim < 256 or width < 160 or height < 160:
        num_levels = 2
    else:
        num_levels = 3

    levels = []
    curr_img = bytearray(img_bytes)
    curr_mask = bytearray(mask_bytes)
    curr_w = width
    curr_h = height
    curr_scale = 1.0

    # Level 0 (Full Resolution)
    levels.append(PyramidLevel(curr_img, curr_mask, curr_w, curr_h, channels, 1.0))

    # Downsample for subsequent levels
    for lvl in range(1, num_levels):
        next_w = max(4, curr_w // 2)
        next_h = max(4, curr_h // 2)
        next_img = bytearray(next_w * next_h * channels)
        next_mask = bytearray(next_w * next_h)

        for dy in range(next_h):
            sy0 = dy * 2
            row_d = dy * next_w
            for dx in range(next_w):
                sx0 = dx * 2
                sum_c = [0] * channels
                mask_val = 0
                count = 0

                for sy_off in range(2):
                    sy = min(curr_h - 1, sy0 + sy_off)
                    row_s = sy * curr_w
                    for sx_off in range(2):
                        sx = min(curr_w - 1, sx0 + sx_off)
                        s_idx = row_s + sx
                        if curr_mask[s_idx] > 10:
                            mask_val = 255
                        pix_s = s_idx * channels
                        for c in range(channels):
                            sum_c[c] += curr_img[pix_s + c]
                        count += 1

                idx_d = row_d + dx
                pix_d = idx_d * channels
                for c in range(channels):
                    next_img[pix_d + c] = sum_c[c] // count
                next_mask[idx_d] = mask_val

        curr_img = next_img
        curr_mask = next_mask
        curr_w = next_w
        curr_h = next_h
        curr_scale *= 0.5
        levels.append(PyramidLevel(curr_img, curr_mask, curr_w, curr_h, channels, curr_scale))

    # Return ordered from coarsest to finest
    return list(reversed(levels))


def project_nnf_to_finer_scale(coarse_nnf, coarse_w, coarse_h, fine_w, fine_h):
    """
    Upsamples a nearest-neighbor field (NNF) from coarse scale to fine scale (2x).
    Preserves exact relative displacement vectors.
    """
    total_fine = fine_w * fine_h
    fine_nnf_x = array.array('h', [0] * total_fine)
    fine_nnf_y = array.array('h', [0] * total_fine)
    coarse_nnf_x, coarse_nnf_y = coarse_nnf

    for fy in range(fine_h):
        cy = min(coarse_h - 1, fy // 2)
        row_f = fy * fine_w
        row_c = cy * coarse_w
        y_mod = fy % 2
        for fx in range(fine_w):
            cx = min(coarse_w - 1, fx // 2)
            idx_f = row_f + fx
            idx_c = row_c + cx
            x_mod = fx % 2

            # Source coordinates scaled by 2 + sub-pixel offset
            sx = min(fine_w - 1, max(0, int(coarse_nnf_x[idx_c]) * 2 + x_mod))
            sy = min(fine_h - 1, max(0, int(coarse_nnf_y[idx_c]) * 2 + y_mod))

            fine_nnf_x[idx_f] = sx
            fine_nnf_y[idx_f] = sy

    return fine_nnf_x, fine_nnf_y
