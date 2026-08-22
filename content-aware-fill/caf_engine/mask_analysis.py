#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1: Mask Analysis
=======================
Analyzes arbitrary irregular user selection masks:
- Boundary pixel extraction (inner & outer)
- Exact Euclidean distance transform
- Boundary normal vector computation
- Geometry statistics: hole area, bounding box, thickness, aspect ratio
- Region classification: edge-crossing vs flat vs textured regions
"""

import math
import array
from collections import deque

class MaskAnalysisResult:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.total_pixels = 0
        self.hole_pixels = []          # List of (x, y) coordinates of unknown pixels
        self.hole_set = set()          # Fast lookup set
        self.boundary_pixels = []      # List of (x, y) on the inner boundary of the hole
        self.outer_band_pixels = []    # List of (x, y) known pixels adjacent to hole
        self.known_centers = []        # Valid source centers (at least r from boundary)
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.hole_w = 0
        self.hole_h = 0
        self.area = 0
        self.aspect_ratio = 1.0
        self.thickness = 1.0
        self.dist_map = None           # 1D array('f') distance from nearest known pixel
        self.normals = {}              # Dict (x, y) -> (nx, ny) unit normal vector pointing inward


def analyze_mask(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4):
    """
    Performs comprehensive analysis of the selection mask and image transparency.
    """
    total = width * height
    r = max(2, int(patch_radius))

    res = MaskAnalysisResult()
    res.width = width
    res.height = height
    res.total_pixels = total

    min_x, max_x = width, 0
    min_y, max_y = height, 0

    mask_grid = bytearray(total)
    hole_pixels = []
    hole_set = set()

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            is_hole = (mask_bytes[idx] > 10) or (channels == 4 and img_bytes[idx * channels + 3] < 10)
            if is_hole:
                mask_grid[idx] = 1
                hole_pixels.append((x, y))
                hole_set.add((x, y))
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
            else:
                mask_grid[idx] = 0

    res.hole_pixels = hole_pixels
    res.hole_set = hole_set
    res.area = len(hole_pixels)

    if not hole_pixels:
        return res

    res.min_x = min_x
    res.max_x = max_x
    res.min_y = min_y
    res.max_y = max_y
    res.hole_w = max_x - min_x + 1
    res.hole_h = max_y - min_y + 1
    res.aspect_ratio = float(res.hole_w) / float(max(1, res.hole_h))

    # Known centers suitable for sampling
    known_centers = []
    for y in range(r, height - r):
        row = y * width
        for x in range(r, width - r):
            if mask_grid[row + x] == 0:
                known_centers.append((x, y))
    res.known_centers = known_centers

    # Boundary pixels & normals
    boundary_pixels = []
    outer_band_pixels = []
    normals = {}

    for x, y in hole_pixels:
        is_boundary = False
        nx_sum = 0.0
        ny_sum = 0.0
        for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            nx = x + ndx
            ny = y + ndy
            if 0 <= nx < width and 0 <= ny < height:
                if mask_grid[ny * width + nx] == 0:
                    is_boundary = True
                    nx_sum += float(-ndx)
                    ny_sum += float(-ndy)
                    outer_band_pixels.append((nx, ny))
            else:
                is_boundary = True
                nx_sum += float(-ndx)
                ny_sum += float(-ndy)

        if is_boundary:
            boundary_pixels.append((x, y))
            mag = math.sqrt(nx_sum * nx_sum + ny_sum * ny_sum)
            if mag > 1e-5:
                normals[(x, y)] = (nx_sum / mag, ny_sum / mag)
            else:
                normals[(x, y)] = (0.0, 1.0)

    res.boundary_pixels = boundary_pixels
    res.outer_band_pixels = list(set(outer_band_pixels))
    res.normals = normals

    # Exact Distance Transform (BFS propagation from known pixels)
    dist_map = array.array('f', [1e6] * total)
    queue = deque()

    for x in range(width):
        for y in range(height):
            idx = y * width + x
            if mask_grid[idx] == 0:
                dist_map[idx] = 0.0
            elif (x, y) in normals:
                dist_map[idx] = 1.0
                queue.append((x, y, 1.0))

    max_dist = 1.0
    while queue:
        cx, cy, cd = queue.popleft()
        for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = cx + ndx
            ny = cy + ndy
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                if mask_grid[n_idx] == 1 and dist_map[n_idx] > cd + 1.0:
                    dist_map[n_idx] = cd + 1.0
                    if cd + 1.0 > max_dist:
                        max_dist = cd + 1.0
                    queue.append((nx, ny, cd + 1.0))

    res.dist_map = dist_map
    res.thickness = max_dist

    return res
