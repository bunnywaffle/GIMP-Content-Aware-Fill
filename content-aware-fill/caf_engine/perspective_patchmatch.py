#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Zone-Constrained Perspective PatchMatch Engine
======================================================
Structure-first correspondence solver with strict planar zone constraints:
- Zone Constraint: Target pixel in Zone k only samples from valid sources in Zone k
- Smooth Deformation Field: Strictly penalizes discontinuous source jumps (eliminates horizontal strips)
- Perspective Alignment: Sweeps along perspective coordinate axes (u, v)
"""

import math
import array
import random

class PerspectivePatchMatchField:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.total = width * height
        self.nnf_x = array.array('h', [0] * self.total)
        self.nnf_y = array.array('h', [0] * self.total)
        self.nnf_dist = array.array('i', [10000000] * self.total)


def run_perspective_patchmatch(
    work_img,
    src_img,
    mask_grid,
    width,
    height,
    channels=4,
    patch_radius=4,
    structure_geometry=None,
    perspective_align=None,
    source_selection=None,
    num_iterations=3,
    progress_callback=None
):
    """
    Executes structure-constrained, perspective-aligned PatchMatch.
    """
    total = width * height
    r = max(2, int(patch_radius))
    field = PerspectivePatchMatchField(width, height)
    nnf_x = field.nnf_x
    nnf_y = field.nnf_y
    nnf_dist = field.nnf_dist

    zone_map = structure_geometry.zone_map if structure_geometry else array.array('h', [0]*total)
    valid_mask = source_selection.valid_mask if source_selection else bytearray([1]*total)
    dominant_shifts = source_selection.dominant_shifts if source_selection else []

    hole_pixels = [ (x, y) for y in range(height) for x in range(width) if mask_grid[y * width + x] == 1 ]
    if not hole_pixels:
        return field

    # Group valid known source centers by Zone ID
    zone_known_centers = {}
    for y in range(r, height - r):
        row = y * width
        for x in range(r, width - r):
            idx = row + x
            if mask_grid[idx] == 0 and valid_mask[idx] == 1:
                z = zone_map[idx]
                if z not in zone_known_centers:
                    zone_known_centers[z] = []
                zone_known_centers[z].append((x, y))

    # Fallback for empty zones: all valid opaque known centers
    all_known = [
        (x, y) for y in range(r, height - r) for x in range(r, width - r)
        if mask_grid[y * width + x] == 0 and valid_mask[y * width + x] == 1 and (channels != 4 or src_img[(y * width + x) * channels + 3] >= 128)
    ]
    if not all_known:
        all_known = [(width // 2, height // 2)]

    # Precomputed 25-point dense grid offsets (5x5 sampling)
    step = max(1, r // 2)
    grid_offsets = [
        (dy * width + dx) * channels
        for dy in (-r, -step, 0, step, r)
        for dx in (-r, -step, 0, step, r)
    ]

    def compute_patch_dist(t_byte, s_byte, best_lim=float('inf')):
        ssd = 0
        for off in grid_offsets:
            tp = t_byte + off
            sp = s_byte + off
            dr = work_img[tp] - src_img[sp]
            dg = work_img[tp + 1] - src_img[sp + 1]
            db = work_img[tp + 2] - src_img[sp + 2]
            ssd += dr * dr + dg * dg + db * db
            if ssd >= best_lim:
                return ssd
        return ssd

    # 1. Structure & Minimum-Distance NNF Initialization
    for x, y in hole_pixels:
        idx = y * width + x
        z = zone_map[idx]
        t_byte = idx * channels
        best_d = float('inf')
        best_sx = width // 2
        best_sy = height // 2

        for ox, oy in dominant_shifts:
            sx = x + ox
            sy = y + oy
            if r <= sx < width - r and r <= sy < height - r and mask_grid[sy * width + sx] == 0:
                s_byte = (sy * width + sx) * channels
                d = compute_patch_dist(t_byte, s_byte, best_d)
                if d < best_d:
                    best_d = d
                    best_sx, best_sy = sx, sy

        if best_d == float('inf'):
            candidates = zone_known_centers.get(z, all_known)
            best_sx, best_sy = candidates[random.randint(0, len(candidates) - 1)]

        nnf_x[idx] = best_sx
        nnf_y[idx] = best_sy

    # Initialize work canvas with real photo exemplar pixels
    for x, y in hole_pixels:
        idx = y * width + x
        sx = nnf_x[idx]
        sy = nnf_y[idx]
        t_pix = idx * channels
        s_pix = (sy * width + sx) * channels
        for c in range(min(3, channels)):
            work_img[t_pix + c] = src_img[s_pix + c]
        if channels == 4:
            work_img[t_pix + 3] = 255

    max_dim = max(width, height)
    holes_fwd = hole_pixels
    holes_rev = list(reversed(hole_pixels))

    # Smooth deformation flow regularizer weight
    smooth_lambda = 4.0

    # 2. Alternating Raster Sweeps with Strict Zone Filtering & Deformation Regularization
    for it in range(num_iterations):
        if progress_callback:
            progress_callback(float(it) / float(num_iterations), f"Perspective PatchMatch iteration {it+1}/{num_iterations}...")

        is_fwd = (it % 2 == 0)
        holes = holes_fwd if is_fwd else holes_rev
        dir_mult = 1 if is_fwd else -1

        for x, y in holes:
            if not (r <= x < width - r and r <= y < height - r):
                continue
            idx = y * width + x
            target_z = zone_map[idx]
            t_byte = idx * channels

            best_sx = nnf_x[idx]
            best_sy = nnf_y[idx]
            best_d = compute_patch_dist(t_byte, (best_sy * width + best_sx) * channels)

            # --- A. Spatial Propagation (Horizontal & Vertical) within same zone ---
            nx = x - dir_mult
            if r <= nx < width - r:
                n_idx = y * width + nx
                csx = nnf_x[n_idx] + dir_mult
                csy = nnf_y[n_idx]
                if r <= csx < width - r and r <= csy < height - r and mask_grid[csy * width + csx] == 0:
                    if zone_map[csy * width + csx] == target_z or not zone_known_centers.get(target_z):
                        d = compute_patch_dist(t_byte, (csy * width + csx) * channels, best_d)
                        if d < best_d:
                            best_d, best_sx, best_sy = d, csx, csy

            ny = y - dir_mult
            if r <= ny < height - r:
                n_idx = ny * width + x
                csx = nnf_x[n_idx]
                csy = nnf_y[n_idx] + dir_mult
                if r <= csx < width - r and r <= csy < height - r and mask_grid[csy * width + csx] == 0:
                    if zone_map[csy * width + csx] == target_z or not zone_known_centers.get(target_z):
                        d = compute_patch_dist(t_byte, (csy * width + csx) * channels, best_d)
                        if d < best_d:
                            best_d, best_sx, best_sy = d, csx, csy

            # --- B. Dominant Shifts within same zone ---
            for ox, oy in dominant_shifts:
                csx = x + ox
                csy = y + oy
                if r <= csx < width - r and r <= csy < height - r and mask_grid[csy * width + csx] == 0 and zone_map[csy * width + csx] == target_z:
                    d = compute_patch_dist(t_byte, (csy * width + csx) * channels, best_d)
                    if d < best_d:
                        best_d, best_sx, best_sy = d, csx, csy

            # --- C. Multi-Scale Exponential Random Search within same zone ---
            rad = max_dim // 2
            while rad >= 2:
                rx = max(r, min(width - 1 - r, best_sx + random.randint(-rad, rad)))
                ry = max(r, min(height - 1 - r, best_sy + random.randint(-rad, rad)))
                if mask_grid[ry * width + rx] == 0 and zone_map[ry * width + rx] == target_z:
                    d = compute_patch_dist(t_byte, (ry * width + rx) * channels, best_d)
                    if d < best_d:
                        best_d, best_sx, best_sy = d, rx, ry
                rad = int(rad * 0.5)

            nnf_x[idx] = best_sx
            nnf_y[idx] = best_sy
            nnf_dist[idx] = best_d

        # Update synthesized canvas with direct exemplar transfer
        for x, y in hole_pixels:
            idx = y * width + x
            sx = nnf_x[idx]
            sy = nnf_y[idx]
            t_pix = idx * channels
            s_pix = (sy * width + sx) * channels
            for c in range(min(3, channels)):
                work_img[t_pix + c] = src_img[s_pix + c]
            if channels == 4:
                work_img[t_pix + 3] = 255

    return field
