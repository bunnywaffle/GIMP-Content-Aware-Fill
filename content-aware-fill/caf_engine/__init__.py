#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAF Engine Package
==================
Structure-First Modular Non-AI Classical Content-Aware Fill Pipeline for GIMP 3.
"""

from .pipeline import execute_content_aware_fill_pipeline
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

__all__ = [
    "execute_content_aware_fill_pipeline",
    "analyze_mask",
    "compute_structure_maps",
    "detect_line_and_perspective_model",
    "propagate_structure_and_partition_zones",
    "build_perspective_alignment",
    "compute_source_suitability",
    "run_perspective_patchmatch",
    "optimize_global_consistency",
    "optimize_patch_seams",
    "adapt_patch_colors",
    "solve_poisson_residual_blending",
    "compute_confidence_map",
    "refine_low_confidence_regions",
]
