#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2: Structure Detection
=============================
Computes advanced classical image structure metrics:
- Scharr gradient filters (Gx, Gy, magnitude, orientation)
- Structure Tensor Matrix J = [Gx^2, GxGy; GxGy, Gy^2] smoothed with Gaussian kernel
- Local coherence / anisotropy map C = ((lambda1 - lambda2) / (lambda1 + lambda2 + eps))^2
- Isophote direction field orthogonal to gradient
- High-confidence structural edge detection along the hole boundary
"""

import math
import array

class StructureMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.total = width * height
        self.gray = array.array('f', [0.0] * self.total)
        self.gx = array.array('f', [0.0] * self.total)
        self.gy = array.array('f', [0.0] * self.total)
        self.grad_mag = array.array('f', [0.0] * self.total)
        self.grad_angle = array.array('f', [0.0] * self.total)
        self.coherence = array.array('f', [0.0] * self.total)      # [0.0, 1.0] (1.0 = strong linear edge)
        self.dominant_angle = array.array('f', [0.0] * self.total) # Edge trajectory angle


def compute_structure_maps(img_bytes, width, height, channels=4, mask_grid=None):
    """
    Computes Scharr gradients, structure tensor, and coherence field.
    """
    total = width * height
    smap = StructureMap(width, height)

    # 1. Luminance Conversion
    gray = smap.gray
    for i in range(total):
        pix = i * channels
        gray[i] = 0.299 * img_bytes[pix] + 0.587 * img_bytes[pix + 1] + 0.114 * img_bytes[pix + 2]

    gx = smap.gx
    gy = smap.gy
    gmag = smap.grad_mag
    gang = smap.grad_angle

    # 2. Scharr Operator for Rotationally Invariant Derivatives
    # Gx = [-3 0 3; -10 0 10; -3 0 3] / 32
    # Gy = [-3 -10 -3; 0 0 0; 3 10 3] / 32
    inv_32 = 1.0 / 32.0

    for y in range(1, height - 1):
        row_prev = (y - 1) * width
        row_curr = y * width
        row_next = (y + 1) * width
        for x in range(1, width - 1):
            idx = row_curr + x
            if mask_grid is not None and mask_grid[idx] == 1:
                continue

            tl = gray[row_prev + x - 1]
            tc = gray[row_prev + x]
            tr = gray[row_prev + x + 1]
            cl = gray[row_curr + x - 1]
            cr = gray[row_curr + x + 1]
            bl = gray[row_next + x - 1]
            bc = gray[row_next + x]
            br = gray[row_next + x + 1]

            val_gx = (3.0 * (tr - tl + br - bl) + 10.0 * (cr - cl)) * inv_32
            val_gy = (3.0 * (bl - tl + br - tr) + 10.0 * (bc - tc)) * inv_32

            gx[idx] = val_gx
            gy[idx] = val_gy
            mag = math.sqrt(val_gx * val_gx + val_gy * val_gy)
            gmag[idx] = mag
            gang[idx] = math.atan2(val_gy, val_gx)

    # 3. Structure Tensor J0 = [Gx^2, GxGy; GxGy, Gy^2]
    jxx = array.array('f', [0.0] * total)
    jxy = array.array('f', [0.0] * total)
    jyy = array.array('f', [0.0] * total)

    for i in range(total):
        vx = gx[i]
        vy = gy[i]
        jxx[i] = vx * vx
        jxy[i] = vx * vy
        jyy[i] = vy * vy

    # Gaussian Smoothing of Tensor Elements (3x3 Kernel: [1 2 1; 2 4 2; 1 2 1] / 16)
    def smooth_tensor_channel(src):
        dst = array.array('f', [0.0] * total)
        for y in range(1, height - 1):
            row_p = (y - 1) * width
            row_c = y * width
            row_n = (y + 1) * width
            for x in range(1, width - 1):
                idx = row_c + x
                val = (src[row_p + x - 1] + 2.0 * src[row_p + x] + src[row_p + x + 1] +
                       2.0 * src[row_c + x - 1] + 4.0 * src[row_c + x] + 2.0 * src[row_c + x + 1] +
                       src[row_n + x - 1] + 2.0 * src[row_n + x] + src[row_n + x + 1]) * (1.0 / 16.0)
                dst[idx] = val
        return dst

    s_jxx = smooth_tensor_channel(jxx)
    s_jxy = smooth_tensor_channel(jxy)
    s_jyy = smooth_tensor_channel(jyy)

    # 4. Eigenvalue Decomposition & Coherence
    coherence = smap.coherence
    dominant_angle = smap.dominant_angle

    for i in range(total):
        a = s_jxx[i]
        b = s_jxy[i]
        c = s_jyy[i]

        # Eigenvalues of 2x2 symmetric matrix
        trace = a + c
        det = a * c - b * b
        disc = math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
        lambda1 = (trace + disc) * 0.5
        lambda2 = max(0.0, (trace - disc) * 0.5)

        denom = lambda1 + lambda2 + 1e-4
        coh = ((lambda1 - lambda2) / denom) ** 2
        coherence[i] = min(1.0, coh)

        # Dominant orientation
        dominant_angle[i] = 0.5 * math.atan2(2.0 * b, a - c)

    return smap
