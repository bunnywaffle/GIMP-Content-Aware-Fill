#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Explicit Structure Propagation & Planar Zone Partitioning
==================================================================
Propagates 2D geometric lines across the selection hole and generates a Planar Zone Map:
- Clusters minor lines into Major Structural Boundaries (max 3-5 macro-zones)
- Guarantees large, contiguous candidate source pools per zone to prevent striping
- Enforces strict macro-zone constraints
"""

import math
import array

class StructureGeometryResult:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.total = width * height
        self.zone_map = array.array('h', [0] * self.total)
        self.propagated_lines = []
        self.num_zones = 1


def propagate_structure_and_partition_zones(mask_analysis, perspective_model, structure_maps):
    """
    Constructs continuous geometric partition lines across the hole and divides the canvas into macro-zones.
    """
    width = mask_analysis.width
    height = mask_analysis.height
    total = width * height
    res = StructureGeometryResult(width, height)
    zone_map = res.zone_map

    hole_set = mask_analysis.hole_set
    line_segments = perspective_model.line_segments
    dominant_angles = perspective_model.dominant_angles

    min_x = mask_analysis.min_x
    max_x = mask_analysis.max_x
    min_y = mask_analysis.min_y
    max_y = mask_analysis.max_y

    # 1. Identify dominant horizontal lines and cluster them into major boundaries
    h_lines = [l for l in line_segments if abs(l.angle) < 0.35]
    h_lines.sort(key=lambda l: l.y1)

    clustered_h_lines = []
    for line in h_lines:
        if not clustered_h_lines:
            clustered_h_lines.append(line)
        else:
            prev = clustered_h_lines[-1]
            # If line is within 50px of previous, merge / keep stronger one
            if abs(line.y1 - prev.y1) < 50:
                if line.strength > prev.strength:
                    clustered_h_lines[-1] = line
            else:
                clustered_h_lines.append(line)

    # Limit to top 3 major horizontal dividing lines
    major_h_lines = clustered_h_lines[:3]
    res.propagated_lines = major_h_lines

    # 2. Detect vertical frame boundary on left / right
    vert_frame_x = None
    v_lines = [l for l in line_segments if abs(abs(l.angle) - math.pi * 0.5) < 0.2]
    if v_lines:
        v_lines.sort(key=lambda l: l.strength, reverse=True)
        for vl in v_lines:
            if vl.x1 < min_x + 80:
                vert_frame_x = vl.x1
                break

    # 3. Partition canvas into Macro-Zones (Zone 1: Frame, Zone 2..N: Planar Layers)
    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x

            if vert_frame_x is not None and x <= vert_frame_x:
                zone_map[idx] = 1
                continue

            if not major_h_lines:
                # 2 macro-zones split at middle of hole
                zone_map[idx] = 2 if y < (min_y + max_y) // 2 else 3
            else:
                z = 2
                for l in major_h_lines:
                    line_y = l.evaluate_y_at_x(x)
                    if y > line_y:
                        z += 1
                zone_map[idx] = z

    res.num_zones = 1 + (1 if vert_frame_x is not None else 0) + max(1, len(major_h_lines) + 1)
    return res
