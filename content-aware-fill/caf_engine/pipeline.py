#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 14: Content-Aware Fill Pipeline Orchestrator
===================================================
Coordinates the complete Structure-First 2-Layer Inpainting Architecture:
1. Mask Analysis
2. Structure Analysis (Scharr & Structure Tensors)
3. Line Segment Fitting & Vanishing Direction Perspective Model
4. Explicit 2D Geometry Bridging & Planar Zone Partitioning
5. Multi-Scale Perspective Alignment
6. Zone-Constrained Perspective PatchMatch
7. Global MRF Regularization (Smooth Deformation Flow)
8. Minimum-Error Direct Seam Transfer
9. Local Color & Exposure Adaptation
10. Screened Harmonic Poisson Residual Healing
11. Output Generation
"""

import time
import array
import math

from .mask_analysis import analyze_mask
from .structure_detection import compute_structure_maps
from .line_perspective_model import detect_line_and_perspective_model
from .structure_geometry import propagate_structure_and_partition_zones
from .perspective_alignment import build_perspective_alignment
from .source_selection import compute_source_suitability
from .perspective_patchmatch import run_perspective_patchmatch
from .global_consistency import optimize_global_consistency
from .seam_optimization import optimize_patch_seams
from .color_adaptation import adapt_patch_colors
from .poisson_blending import solve_poisson_residual_blending
from .iterative_refine import compute_confidence_map, refine_low_confidence_regions


def execute_content_aware_fill_pipeline(
    img_bytes,
    mask_bytes,
    width,
    height,
    channels=4,
    patch_radius=4,
    sample_source="auto",
    adaptation_mode="none",
    quality_preset="balanced",
    structure_preset="medium",
    manual_source_mask=None,
    progress_callback=None
):
    """
    Executes the full Structure-First Perspective Content-Aware Fill pipeline.
    """
    t_start = time.time()

    def report(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)

    # 1. Mask Analysis
    report(0.05, "Analyzing selection mask & boundary geometry...")
    mask_res = analyze_mask(img_bytes, mask_bytes, width, height, channels, patch_radius)
    if not mask_res.hole_pixels or not mask_res.known_centers:
        return img_bytes

    # 2. Structure & Edge Analysis
    report(0.12, "Detecting Scharr gradients & structure tensors...")
    mask_grid = bytearray(width * height)
    for x, y in mask_res.hole_pixels:
        mask_grid[y * width + x] = 1
    s_maps = compute_structure_maps(img_bytes, width, height, channels, mask_grid)

    # 3. Line & Perspective Modeling
    report(0.18, "Fitting structural line segments & perspective vanishing model...")
    persp_model = detect_line_and_perspective_model(s_maps, mask_res)

    # 4. Explicit Structure Propagation & Planar Zone Partitioning
    report(0.24, "Propagating geometric boundary lines & partitioning planar zones...")
    struct_geom = propagate_structure_and_partition_zones(mask_res, persp_model, s_maps)

    # 5. Perspective Coordinate Alignment
    persp_align = build_perspective_alignment(persp_model)

    # 6. Source Suitability & Dominant Shifts
    report(0.30, "Evaluating multi-zone candidate source pools & trajectory shifts...")
    source_sel = compute_source_suitability(
        img_bytes, mask_bytes, width, height, channels,
        patch_radius, sample_source, manual_source_mask, mask_res, struct_geom.propagated_lines
    )

    # Iteration settings based on quality preset
    if quality_preset == "fast":
        em_iters = 1
    elif quality_preset == "high":
        em_iters = 3
    else:  # balanced
        em_iters = 2

    # 7. Zone-Constrained Perspective PatchMatch Correspondence
    report(0.40, "Running Zone-Constrained Perspective PatchMatch...")
    work_canvas = bytearray(img_bytes)
    field = run_perspective_patchmatch(
        work_canvas,
        img_bytes,
        mask_grid,
        width,
        height,
        channels=channels,
        patch_radius=patch_radius,
        structure_geometry=struct_geom,
        perspective_align=persp_align,
        source_selection=source_sel,
        num_iterations=em_iters,
        progress_callback=lambda f, m: report(0.40 + 0.30 * f, m)
    )

    # 8. Global MRF Consistency Optimization (Smooth Deformation Regularization)
    report(0.75, "Optimizing global spatial deformation consistency...")
    optimize_global_consistency(
        field, work_canvas, img_bytes, mask_grid, width, height,
        channels=channels, patch_radius=patch_radius, source_selection=source_sel, num_passes=1
    )

    # 9. Seam Optimization (Direct Exemplar Transfer with Zero Muddy Blending)
    report(0.82, "Enforcing direct exemplar patch transfers...")
    optimize_patch_seams(
        work_canvas, img_bytes, field, mask_grid, width, height,
        channels=channels, patch_radius=patch_radius
    )

    # 10. Local Color & Exposure Adaptation (if requested)
    if adaptation_mode != "none":
        report(0.88, "Harmonizing local lighting & contrast...")
        adapt_patch_colors(work_canvas, img_bytes, mask_res, width, height, channels, patch_radius)

    # 11. Gradient-Domain Screened Poisson Residual Healing
    report(0.93, "Solving harmonic boundary residual field...")
    solve_poisson_residual_blending(
        work_canvas, img_bytes, mask_res, width, height, channels,
        num_iterations=8, progress_callback=None
    )

    # 12. Confidence Map & Iterative Refinement
    report(0.97, "Evaluating pixel confidence & fine details...")
    cmap = compute_confidence_map(field, mask_res, width, height, channels)
    refine_low_confidence_regions(
        work_canvas, img_bytes, field, cmap, mask_res, width, height,
        channels=channels, patch_radius=patch_radius, source_selection=source_sel
    )

    elapsed = time.time() - t_start
    print(f"[CAF Pipeline] Successfully reconstructed {width}x{height} in {elapsed:.3f}s")

    return work_canvas
