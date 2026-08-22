#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3: Structure Propagation & Priority Front
================================================
Implements Criminisi-style priority ordering with structure tensor enhancement:
- Priority = Confidence(p) x Data(p) x (1.0 + Coherence(p))
- Edge trajectory tracing across the hole to guide structural continuity
- Structure constraint maps for PatchMatch candidate selection
"""

import math
import array
import heapq

class StructureTrajectory:
    def __init__(self, start_pt, end_pt, angle, strength):
        self.start_pt = start_pt  # (x1, y1)
        self.end_pt = end_pt      # (x2, y2)
        self.angle = angle        # Orientation
        self.strength = strength  # Gradient magnitude


def compute_priority_front(mask_analysis, structure_maps, patch_radius=4):
    """
    Computes priority values for all boundary pixels along the fill front.
    """
    r = patch_radius
    patch_area = float((2 * r + 1) * (2 * r + 1))
    width = mask_analysis.width
    height = mask_analysis.height
    total = mask_analysis.total_pixels

    confidence = array.array('f', [1.0] * total)
    for x, y in mask_analysis.hole_pixels:
        confidence[y * width + x] = 0.0

    front_heap = []
    normals = mask_analysis.normals
    grad_mag = structure_maps.grad_mag
    gx = structure_maps.gx
    gy = structure_maps.gy
    coherence = structure_maps.coherence

    for px, py in mask_analysis.boundary_pixels:
        p_idx = py * width + px
        nx, ny = normals.get((px, py), (0.0, 1.0))

        # 1. Confidence Term C(p)
        c_sum = 0.0
        for dy in range(-r, r + 1):
            qy = py + dy
            if 0 <= qy < height:
                row_q = qy * width
                for dx in range(-r, r + 1):
                    qx = px + dx
                    if 0 <= qx < width:
                        c_sum += confidence[row_q + qx]
        cp = c_sum / patch_area

        # 2. Data / Structure Term D(p) = |grad_perp . n_p| / alpha
        # Isophote direction orthogonal to gradient: (-gy, gx)
        val_gx = gx[p_idx]
        val_gy = gy[p_idx]
        isophote_x = -val_gy
        isophote_y = val_gx

        dp = abs(isophote_x * nx + isophote_y * ny) / 255.0
        dp = max(0.01, min(1.0, dp))

        # 3. Structure Tensor Coherence Bonus
        coh = coherence[p_idx]
        priority = cp * dp * (1.0 + 2.0 * coh)

        # Min-heap stores negative priority for max-priority pop
        heapq.heappush(front_heap, (-priority, px, py))

    return front_heap


def trace_structural_trajectories(mask_analysis, structure_maps, min_strength=20.0):
    """
    Traces strong linear structures entering the hole boundary and matches them
    across opposite sides to form structural bridge lines.
    """
    width = mask_analysis.width
    height = mask_analysis.height
    boundary_pixels = mask_analysis.boundary_pixels
    grad_mag = structure_maps.grad_mag
    coherence = structure_maps.coherence
    dominant_angle = structure_maps.dominant_angle

    strong_entries = []
    for x, y in boundary_pixels:
        idx = y * width + x
        mag = grad_mag[idx]
        coh = coherence[idx]
        if mag >= min_strength and coh >= 0.4:
            strong_entries.append((x, y, dominant_angle[idx], mag))

    trajectories = []
    used = set()

    for i in range(len(strong_entries)):
        if i in used:
            continue
        x1, y1, ang1, mag1 = strong_entries[i]

        best_match = None
        min_angle_diff = 0.35  # ~20 degrees
        min_proj_dist = float('inf')

        dx_ray = math.cos(ang1)
        dy_ray = math.sin(ang1)

        for j in range(i + 1, len(strong_entries)):
            if j in used:
                continue
            x2, y2, ang2, mag2 = strong_entries[j]
            dist_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if dist_sq < 16:  # Skip nearby points on same boundary
                continue

            # Angular alignment
            ang_diff = abs(math.atan2(math.sin(ang1 - ang2), math.cos(ang1 - ang2)))
            if ang_diff > math.pi * 0.5:
                ang_diff = abs(math.pi - ang_diff)

            if ang_diff < min_angle_diff:
                # Geometric collinearity
                vec_x = x2 - x1
                vec_y = y2 - y1
                vec_len = math.sqrt(vec_x * vec_x + vec_y * vec_y)
                if vec_len > 0:
                    vec_x /= vec_len
                    vec_y /= vec_len
                    collinearity = abs(vec_x * dx_ray + vec_y * dy_ray)
                    if collinearity > 0.85:
                        trajectories.append(StructureTrajectory((x1, y1), (x2, y2), ang1, (mag1 + mag2) * 0.5))
                        used.add(i)
                        used.add(j)
                        break

    return trajectories
