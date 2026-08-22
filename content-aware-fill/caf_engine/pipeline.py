#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 14: Content-Aware Fill Pipeline Orchestrator
===================================================
Coordinates the complete 14-stage non-AI classical inpainting pipeline:
1. Mask Analysis
2. Structure Detection (Scharr & Structure Tensor)
3. Structure Propagation & Trajectory Tracing
4. Multi-Scale Image Pyramid Construction
5. Source Region Selection & Dominant Shifts
6. Multi-Scale PatchMatch NNF Optimization
7. Global MRF Consistency Optimization
8. Minimum-Error Seam Optimization
9. Local Color & Exposure Adaptation
10. Gradient-Domain Poisson Residual Healing
11. Confidence-Driven Iterative Refinement
12. Final Buffer Output
"""

import time
import array
import math

from .mask_analysis import analyze_mask
from .structure_detection import compute_structure_maps
from .structure_propagation import trace_structural_trajectories, compute_priority_front
from .pyramid import build_image_pyramid, project_nnf_to_finer_scale
from .source_selection import compute_source_suitability
from .patch_distance import PatchDistanceEvaluator
from .transformations import get_standard_transformations
from .patchmatch import run_patchmatch_solver
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
    Executes the full modular Content-Aware Fill pipeline.
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

    # 3. Structure Trajectories
    report(0.18, "Tracing structural boundary trajectories...")
    trajectories = trace_structural_trajectories(mask_res, s_maps)

    # 4. Source Suitability & Dominant Shifts
    report(0.24, "Evaluating candidate source regions & dominant shifts...")
    source_sel = compute_source_suitability(
        img_bytes, mask_bytes, width, height, channels,
        patch_radius, sample_source, manual_source_mask, mask_res, trajectories
    )

    # 5. Multi-Scale Image Pyramid
    report(0.30, "Building multi-scale Gaussian pyramid...")
    pyramid_levels = build_image_pyramid(
        img_bytes, mask_bytes, width, height, channels,
        max_hole_dim=max(mask_res.hole_w, mask_res.hole_h)
    )

    # Iteration settings based on quality preset
    if quality_preset == "fast":
        em_iters = 1
    elif quality_preset == "high":
        em_iters = 3
    else:  # balanced
        em_iters = 2

    transforms = get_standard_transformations(adaptation_mode)

    # 6. Multi-Scale PatchMatch Processing
    current_nnf = None
    work_canvas = bytearray(img_bytes)

    for lvl_idx, lvl in enumerate(pyramid_levels):
        is_finest = (lvl_idx == len(pyramid_levels) - 1)
        lvl_frac_base = 0.35 + 0.35 * (lvl_idx / float(len(pyramid_levels)))
        report(lvl_frac_base, f"Synthesizing texture at scale {lvl.width}x{lvl.height}...")

        lvl_mask_res = analyze_mask(lvl.img_bytes, lvl.mask_bytes, lvl.width, lvl.height, channels, patch_radius)
        lvl_mask_grid = bytearray(lvl.width * lvl.height)
        for lx, ly in lvl_mask_res.hole_pixels:
            lvl_mask_grid[ly * lvl.width + lx] = 1

        lvl_source_sel = compute_source_suitability(
            lvl.img_bytes, lvl.mask_bytes, lvl.width, lvl.height, channels,
            patch_radius, sample_source, manual_source_mask, lvl_mask_res
        )

        lvl_work = bytearray(lvl.img_bytes)

        # Upsample NNF from previous coarser level if available
        init_nnf = None
        if current_nnf is not None:
            prev_lvl = pyramid_levels[lvl_idx - 1]
            init_nnf = project_nnf_to_finer_scale(
                current_nnf, prev_lvl.width, prev_lvl.height, lvl.width, lvl.height
            )

        dist_eval = PatchDistanceEvaluator(lvl.width, lvl.height, channels, patch_radius)

        field = run_patchmatch_solver(
            lvl_work,
            lvl.img_bytes,
            lvl_mask_grid,
            lvl.width,
            lvl.height,
            channels=channels,
            patch_radius=patch_radius,
            source_selection=lvl_source_sel,
            num_iterations=em_iters,
            initial_nnf=init_nnf,
            transformations=transforms,
            dist_evaluator=dist_eval,
            progress_callback=None
        )

        current_nnf = (field.nnf_x, field.nnf_y)
        if is_finest:
            work_canvas = lvl_work

    # 7. Global MRF Consistency Optimization
    report(0.75, "Optimizing global spatial patch consistency...")
    optimize_global_consistency(
        field, work_canvas, img_bytes, mask_grid, width, height,
        channels=channels, patch_radius=patch_radius, source_selection=source_sel, num_passes=1
    )

    # 8. Seam Optimization
    report(0.82, "Optimizing minimum-error patch seams...")
    optimize_patch_seams(
        work_canvas, img_bytes, field, mask_grid, width, height,
        channels=channels, patch_radius=patch_radius
    )

    # 9. Local Color & Exposure Adaptation
    report(0.88, "Adapting local lighting & contrast...")
    adapt_patch_colors(work_canvas, img_bytes, mask_res, width, height, channels, patch_radius)

    # 10. Gradient-Domain Poisson Residual Healing
    report(0.93, "Solving harmonic Poisson boundary field...")
    solve_poisson_residual_blending(
        work_canvas, img_bytes, mask_res, width, height, channels,
        num_iterations=10, progress_callback=None
    )

    # 11. Confidence Map & Iterative Refinement
    report(0.97, "Evaluating pixel confidence & fine details...")
    cmap = compute_confidence_map(field, mask_res, width, height, channels)
    refine_low_confidence_regions(
        work_canvas, img_bytes, field, cmap, mask_res, width, height,
        channels=channels, patch_radius=patch_radius, source_selection=source_sel
    )

    # Final Touch-Up Poisson Pass
    solve_poisson_residual_blending(
        work_canvas, img_bytes, mask_res, width, height, channels,
        num_iterations=4, progress_callback=None
    )

    elapsed = time.time() - t_start
    print(f"[CAF Pipeline] Successfully reconstructed {width}x{height} in {elapsed:.3f}s")

    return work_canvas
