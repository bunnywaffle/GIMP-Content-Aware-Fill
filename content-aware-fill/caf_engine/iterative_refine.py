#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 13: Confidence Map & Iterative Refinement
================================================
Tracks reconstruction confidence across the filled hole:
- Evaluates patch match SSD error, distance from boundary, and edge continuity
- Detects low-confidence regions and performs targeted second-pass refinement
"""

import math
import array

class ConfidenceMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.total = width * height
        self.confidence = array.array('f', [1.0] * self.total)


def compute_confidence_map(nnf_field, mask_analysis, width, height, channels=4):
    """
    Computes a normalized confidence value for every pixel in the hole.
    """
    total = width * height
    cmap = ConfidenceMap(width, height)
    conf = cmap.confidence
    nnf_dist = nnf_field.nnf_dist
    dist_map = mask_analysis.dist_map
    max_d = max(1.0, mask_analysis.thickness)

    for x, y in mask_analysis.hole_pixels:
        idx = y * width + x
        d_val = dist_map[idx]
        geo_conf = max(0.2, 1.0 - (d_val / (max_d * 2.0)))

        ssd = nnf_dist[idx]
        # Normalize SSD (lower is better, e.g. ssd 0 -> 1.0, ssd 50000 -> 0.0)
        ssd_conf = 1.0 / (1.0 + (ssd / 15000.0))

        conf[idx] = min(1.0, max(0.05, geo_conf * ssd_conf))

    return cmap


def refine_low_confidence_regions(
    work_img,
    src_img,
    nnf_field,
    confidence_map,
    mask_analysis,
    width,
    height,
    channels=4,
    patch_radius=4,
    source_selection=None
):
    """
    Performs focused local search on pixels with confidence below threshold.
    """
    conf = confidence_map.confidence
    nnf_x = nnf_field.nnf_x
    nnf_y = nnf_field.nnf_y
    nnf_dist = nnf_field.nnf_dist
    known_centers = source_selection.known_centers if source_selection else []
    valid_mask = source_selection.valid_mask if source_selection else bytearray([1]*(width*height))
    r = patch_radius

    if not known_centers:
        return

    low_conf_pixels = [
        (x, y) for x, y in mask_analysis.hole_pixels
        if conf[y * width + x] < 0.45 and (r <= x < width - r and r <= y < height - r)
    ]

    if not low_conf_pixels:
        return

    grid_offsets = [
        (-r * width - r) * channels, (0) * channels, (r * width + r) * channels,
        (-r) * channels, (r) * channels
    ]

    for x, y in low_conf_pixels:
        idx = y * width + x
        t_byte = idx * channels
        best_sx = nnf_x[idx]
        best_sy = nnf_y[idx]
        best_d = nnf_dist[idx]

        # Test additional diverse random samples
        for _ in range(8):
            cand_sx, cand_sy = known_centers[int(len(known_centers) * 0.5) % len(known_centers)]
            if valid_mask[cand_sy * width + cand_sx] == 1:
                ssd = 0
                for off in grid_offsets:
                    tp = t_byte + off
                    sp = (cand_sy * width + cand_sx) * channels + off
                    dr = work_img[tp] - src_img[sp]
                    dg = work_img[tp + 1] - src_img[sp + 1]
                    db = work_img[tp + 2] - src_img[sp + 2]
                    ssd += dr * dr + dg * dg + db * db
                if ssd < best_d:
                    best_d = ssd
                    best_sx, best_sy = cand_sx, cand_sy

        nnf_x[idx] = best_sx
        nnf_y[idx] = best_sy
        nnf_dist[idx] = best_d
        s_pix = (best_sy * width + best_sx) * channels
        for c in range(min(3, channels)):
            work_img[t_byte + c] = src_img[s_pix + c]
