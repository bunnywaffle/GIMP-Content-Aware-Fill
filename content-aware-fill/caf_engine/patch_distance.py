#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 5: Composite Patch Distance Metric
=========================================
Evaluates perceptual patch similarity combining:
- Perceptual YCbCr color distance (luminance vs chroma weighting)
- Scharr gradient magnitude & orientation alignment
- Structure tensor coherence penalty
- Fast 1D byte offsets with immediate early-exit bounding
"""

import math
import array

class PatchDistanceEvaluator:
    def __init__(self, width, height, channels=4, patch_radius=4, structure_maps=None):
        self.width = width
        self.height = height
        self.channels = channels
        self.r = patch_radius
        self.structure_maps = structure_maps

        # Precompute flat 1D grid byte offsets for fast inner-loop cache access
        # Multi-point sparse star + grid pattern
        r = self.r
        self.grid_offsets_2d = [
            (-r, -r), (0, -r), (r, -r),
            (-r, 0),  (0, 0),  (r, 0),
            (-r, r),  (0, r),  (r, r)
        ]
        self.grid_byte_offsets = [
            (dy * width + dx) * channels for dx, dy in self.grid_offsets_2d
        ]
        self.grid_pixel_offsets = [
            dy * width + dx for dx, dy in self.grid_offsets_2d
        ]

    def compute_patch_distance_fast(self, work_img, src_img, t_byte, s_byte, best_limit=float('inf')):
        """
        Fast early-exit RGB/RGBA patch distance.
        """
        ssd = 0
        for off in self.grid_byte_offsets:
            tp = t_byte + off
            sp = s_byte + off
            dr = work_img[tp] - src_img[sp]
            dg = work_img[tp + 1] - src_img[sp + 1]
            db = work_img[tp + 2] - src_img[sp + 2]
            ssd += dr * dr + dg * dg + db * db
            if ssd >= best_limit:
                return ssd
        return ssd

    def compute_composite_distance(self, work_img, src_img, t_x, t_y, s_x, s_y, best_limit=float('inf'), struct_weight=0.5):
        """
        Full composite distance including YCbCr luminance/chroma and gradient tensor alignment.
        """
        width = self.width
        channels = self.channels
        t_center_idx = t_y * width + t_x
        s_center_idx = s_y * width + s_x
        t_center_byte = t_center_idx * channels
        s_center_byte = s_center_idx * channels

        # 1. Color Distance in YCbCr
        color_cost = 0.0
        for off in self.grid_byte_offsets:
            tp = t_center_byte + off
            sp = s_center_byte + off

            # Target RGB
            tr = work_img[tp]
            tg = work_img[tp + 1]
            tb = work_img[tp + 2]

            # Source RGB
            sr = src_img[sp]
            sg = src_img[sp + 1]
            sb = src_img[sp + 2]

            # Luminance Y
            ty = 0.299 * tr + 0.587 * tg + 0.114 * tb
            sy = 0.299 * sr + 0.587 * sg + 0.114 * sb
            dy = ty - sy

            # Chroma Cb, Cr
            t_cb = 128.0 - 0.1687 * tr - 0.3313 * tg + 0.5 * tb
            s_cb = 128.0 - 0.1687 * sr - 0.3313 * sg + 0.5 * sb
            d_cb = t_cb - s_cb

            t_cr = 128.0 + 0.5 * tr - 0.4187 * tg - 0.0813 * tb
            s_cr = 128.0 + 0.5 * sr - 0.4187 * sg - 0.0813 * sb
            d_cr = t_cr - s_cr

            color_cost += 2.0 * (dy * dy) + 0.7 * (d_cb * d_cb + d_cr * d_cr)
            if color_cost >= best_limit:
                return color_cost

        # 2. Gradient / Structure Alignment
        grad_cost = 0.0
        if self.structure_maps is not None and struct_weight > 0.0:
            smap = self.structure_maps
            t_coh = smap.coherence[t_center_idx]
            s_coh = smap.coherence[s_center_idx]

            # Coherence mismatch penalty
            coh_diff = abs(t_coh - s_coh)
            grad_cost += coh_diff * 5000.0

            # Orientation alignment when coherence is high
            if t_coh > 0.3 and s_coh > 0.3:
                t_ang = smap.dominant_angle[t_center_idx]
                s_ang = smap.dominant_angle[s_center_idx]
                ang_diff = abs(math.atan2(math.sin(t_ang - s_ang), math.cos(t_ang - s_ang)))
                if ang_diff > math.pi * 0.5:
                    ang_diff = abs(math.pi - ang_diff)
                grad_cost += (ang_diff / (math.pi * 0.5)) * 10000.0 * (t_coh * s_coh)

        total_cost = color_cost + struct_weight * grad_cost
        return total_cost
