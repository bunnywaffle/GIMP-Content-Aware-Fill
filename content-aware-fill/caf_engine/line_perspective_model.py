#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Line & Perspective Model
================================
Detects long structural line segments and vanishing directions:
- Extracts high-gradient edge contours around the selection boundary
- Fits linear geometric segments using RANSAC and angular clustering
- Identifies dominant perspective slopes (e.g. tilted shelf angles, road angles)
- Computes perspective vanishing rays and directional vector fields
"""

import math
import array
import random
from collections import defaultdict

class LineSegment:
    def __init__(self, x1, y1, x2, y2, angle, strength):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.angle = angle          # Angle in radians [-pi/2, pi/2]
        self.strength = strength    # Average gradient magnitude
        self.length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def point_distance(self, px, py):
        """Orthogonal distance from (px, py) to the line."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - self.x1)**2 + (py - self.y1)**2)
        return abs(dy * px - dx * py + self.x2 * self.y1 - self.y2 * self.x1) / math.sqrt(dx*dx + dy*dy)

    def evaluate_y_at_x(self, x):
        """Evaluates y = y1 + m*(x - x1) on the line."""
        dx = self.x2 - self.x1
        if abs(dx) < 1e-4:
            return self.y1
        slope = (self.y2 - self.y1) / dx
        return self.y1 + slope * (x - self.x1)


class PerspectiveModel:
    def __init__(self, dominant_angles, line_segments, vanishing_point=None):
        self.dominant_angles = dominant_angles # List of dominant angles (e.g. [-0.15, 1.52])
        self.line_segments = line_segments     # Detected strong line segments
        self.vanishing_point = vanishing_point # (vx, vy) if lines converge, or None


def detect_line_and_perspective_model(structure_maps, mask_analysis, min_length=20.0):
    """
    Analyzes known regions around the hole to detect line segments and perspective angles.
    """
    width = structure_maps.width
    height = structure_maps.height
    grad_mag = structure_maps.grad_mag
    dominant_angle = structure_maps.dominant_angle
    coherence = structure_maps.coherence
    hole_set = mask_analysis.hole_set

    # 1. Collect candidate edge points with high coherence and gradient magnitude
    edge_points = []
    angle_hist = defaultdict(list)

    for y in range(2, height - 2, 2):
        row = y * width
        for x in range(2, width - 2, 2):
            idx = row + x
            if (x, y) not in hole_set and grad_mag[idx] > 25.0 and coherence[idx] > 0.4:
                ang = dominant_angle[idx]
                # Normalize angle to [-pi/2, pi/2]
                while ang > math.pi * 0.5: ang -= math.pi
                while ang < -math.pi * 0.5: ang += math.pi
                edge_points.append((x, y, ang, grad_mag[idx]))
                # Quantize angle into 5-degree bins (36 bins)
                bin_idx = int(round(ang / (math.pi / 36.0)))
                angle_hist[bin_idx].append((x, y, ang, grad_mag[idx]))

    # 2. Extract Dominant Orientation Clusters
    dominant_angles = []
    sorted_bins = sorted(angle_hist.items(), key=lambda item: len(item[1]), reverse=True)
    for bin_idx, pts in sorted_bins[:4]:
        if len(pts) >= 15:
            avg_ang = sum(p[2] for p in pts) / float(len(pts))
            dominant_angles.append(avg_ang)

    if not dominant_angles:
        dominant_angles = [0.0, math.pi * 0.5]  # Default horizontal and vertical

    # 3. Fit Line Segments using RANSAC along Dominant Angles
    line_segments = []
    for ang in dominant_angles:
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        # Normal vector perpendicular to line: (-sin_a, cos_a)
        nx = -sin_a
        ny = cos_a

        cluster_pts = [p for p in edge_points if abs(math.atan2(math.sin(p[2] - ang), math.cos(p[2] - ang))) < 0.25]
        if len(cluster_pts) < 10:
            continue

        # Project points onto line normal: rho = x*nx + y*ny
        rho_bins = defaultdict(list)
        for px, py, _, mag in cluster_pts:
            rho = px * nx + py * ny
            rho_bin = int(round(rho / 6.0))
            rho_bins[rho_bin].append((px, py, mag))

        for rho_bin, pts in rho_bins.items():
            if len(pts) >= 6:
                # Find extent along the line tangent (cos_a, sin_a)
                projections = [(p[0] * cos_a + p[1] * sin_a, p[0], p[1], p[2]) for p in pts]
                projections.sort(key=lambda item: item[0])
                min_p = projections[0]
                max_p = projections[-1]
                length = max_p[0] - min_p[0]
                if length >= min_length:
                    avg_strength = sum(p[3] for p in projections) / float(len(projections))
                    line_segments.append(LineSegment(min_p[1], min_p[2], max_p[1], max_p[2], ang, avg_strength))

    # 4. Vanishing Point Estimation (Intersection of Non-Parallel Dominant Lines)
    vanishing_point = None
    if len(line_segments) >= 2:
        # Find intersections of lines with similar angle (perspective convergence)
        intersections = []
        for i in range(len(line_segments)):
            l1 = line_segments[i]
            for j in range(i + 1, len(line_segments)):
                l2 = line_segments[j]
                ang_diff = abs(l1.angle - l2.angle)
                if 0.02 < ang_diff < 0.20:  # Slight perspective tilt convergence
                    # Line equations: A1*x + B1*y = C1, A2*x + B2*y = C2
                    a1 = l1.y2 - l1.y1
                    b1 = l1.x1 - l1.x2
                    c1 = a1 * l1.x1 + b1 * l1.y1

                    a2 = l2.y2 - l2.y1
                    b2 = l2.x1 - l2.x2
                    c2 = a2 * l2.x1 + b2 * l2.y1

                    det = a1 * b2 - a2 * b1
                    if abs(det) > 1e-4:
                        ix = (b2 * c1 - b1 * c2) / det
                        iy = (a1 * c2 - a2 * c1) / det
                        if -5000 < ix < 5000 and -5000 < iy < 5000:
                            intersections.append((ix, iy))

        if intersections:
            avg_vx = sum(p[0] for p in intersections) / float(len(intersections))
            avg_vy = sum(p[1] for p in intersections) / float(len(intersections))
            vanishing_point = (avg_vx, avg_vy)

    return PerspectiveModel(dominant_angles, line_segments, vanishing_point)
