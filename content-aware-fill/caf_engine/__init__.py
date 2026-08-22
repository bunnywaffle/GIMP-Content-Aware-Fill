#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAF Engine Package
==================
Modular Non-AI Classical Content-Aware Fill Pipeline for GIMP 3.
"""

from .pipeline import execute_content_aware_fill_pipeline
from .mask_analysis import analyze_mask
from .structure_detection import compute_structure_maps
from .structure_propagation import trace_structural_trajectories, compute_priority_front
from .pyramid import build_image_pyramid, project_nnf_to_finer_scale
from .patch_distance import PatchDistanceEvaluator
from .transformations import get_standard_transformations
from .source_selection import compute_source_suitability
from .patchmatch import run_patchmatch_solver
from .global_consistency import optimize_global_consistency
from .seam_optimization import optimize_patch_seams
from .color_adaptation import adapt_patch_colors
from .poisson_blending import solve_poisson_residual_blending
from .iterative_refine import compute_confidence_map, refine_low_confidence_regions

__all__ = [
    "execute_content_aware_fill_pipeline",
    "analyze_mask",
    "compute_structure_maps",
    "trace_structural_trajectories",
    "compute_priority_front",
    "build_image_pyramid",
    "project_nnf_to_finer_scale",
    "PatchDistanceEvaluator",
    "get_standard_transformations",
    "compute_source_suitability",
    "run_patchmatch_solver",
    "optimize_global_consistency",
    "optimize_patch_seams",
    "adapt_patch_colors",
    "solve_poisson_residual_blending",
    "compute_confidence_map",
    "refine_low_confidence_regions",
]
