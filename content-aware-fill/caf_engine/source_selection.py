#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 7: Source Region Selection & Suitability
===============================================
Constructs source suitability maps and discovers multi-zone dominant shift candidates:
- Evaluates candidate shifts on localized boundary segments (multi-zone clustering)
- Supports user directional presets (Auto, Right, Left, Above, Below, All Around)
- Multi-threaded candidate shift evaluation (He & Sun 2012)
"""

import math
import array
import os
import concurrent.futures

class SourceSelectionMap:
    def __init__(self, width, height, valid_mask, known_centers, dominant_shifts):
        self.width = width
        self.height = height
        self.valid_mask = valid_mask          # bytearray(total) where 1 = valid source
        self.known_centers = known_centers    # list of (sx, sy)
        self.dominant_shifts = dominant_shifts # list of (dx, dy)


def compute_source_suitability(
    img_bytes,
    mask_bytes,
    width,
    height,
    channels=4,
    patch_radius=4,
    sample_source="auto",
    manual_source_mask=None,
    mask_analysis=None
):
    """
    Computes valid source locations and evaluates dominant spatial shift vectors across multiple image zones.
    """
    total = width * height
    r = max(2, int(patch_radius))
    cpu_cores = os.cpu_count() or 4

    valid_mask = bytearray(total)
    known_centers = []

    # 1. Base validity: outside hole + safety margin
    for y in range(r, height - r):
        row = y * width
        for x in range(r, width - r):
            idx = row + x
            is_hole = (mask_bytes[idx] > 10) or (channels == 4 and img_bytes[idx * channels + 3] < 10)
            if not is_hole:
                if manual_source_mask is not None:
                    if manual_source_mask[idx] > 10:
                        valid_mask[idx] = 1
                        known_centers.append((x, y))
                else:
                    valid_mask[idx] = 1
                    known_centers.append((x, y))

    if not known_centers:
        # Fallback: all non-hole pixels with clamped patch centers
        for y in range(height):
            row = y * width
            for x in range(width):
                idx = row + x
                is_hole = (mask_bytes[idx] > 10) or (channels == 4 and img_bytes[idx * channels + 3] < 10)
                if not is_hole:
                    valid_mask[idx] = 1
                    cx = max(r, min(width - 1 - r, x))
                    cy = max(r, min(height - 1 - r, y))
                    known_centers.append((cx, cy))

    # 2. Dominant Shift Vector Discovery
    sel_w = mask_analysis.hole_w if mask_analysis else width // 4
    sel_h = mask_analysis.hole_h if mask_analysis else height // 4
    min_x = mask_analysis.min_x if mask_analysis else 0
    max_x = mask_analysis.max_x if mask_analysis else width - 1
    min_y = mask_analysis.min_y if mask_analysis else 0
    max_y = mask_analysis.max_y if mask_analysis else height - 1

    eval_band = mask_analysis.boundary_pixels[::max(1, len(mask_analysis.boundary_pixels) // 50)] if mask_analysis else []
    num_eval = len(eval_band)

    if sample_source == "right":
        dx_cands = list(range(max(4, sel_w // 4), min(width - min_x - 1, sel_w * 2 + 80), 8))
        dy_cands = list(range(-32, 33, 8))
    elif sample_source == "left":
        dx_cands = list(range(-min(max_x - 1, sel_w * 2 + 80), -max(4, sel_w // 4), 8))
        dy_cands = list(range(-32, 33, 8))
    elif sample_source == "above":
        dx_cands = list(range(-32, 33, 8))
        dy_cands = list(range(-min(max_y - 1, sel_h * 2 + 80), -max(4, sel_h // 4), 8))
    elif sample_source == "below":
        dx_cands = list(range(-32, 33, 8))
        dy_cands = list(range(max(4, sel_h // 4), min(height - min_y - 1, sel_h * 2 + 80), 8))
    else:  # Auto
        dx_cands = list(range(-min(max_x - 1, sel_w + 80), -max(4, sel_w // 4), 8)) + \
                    list(range(max(4, sel_w // 4), min(width - min_x - 1, sel_w + 80), 8)) + [0]
        dy_cands = list(range(-min(max_y - 1, sel_h + 80), -max(4, sel_h // 4), 8)) + \
                    list(range(max(4, sel_h // 4), min(height - min_y - 1, sel_h + 80), 8)) + [0]

    all_candidate_shifts = [(dx, dy) for dy in dy_cands for dx in dx_cands if (dx != 0 or dy != 0)]

    def evaluate_shift_chunk(shifts_chunk):
        results = []
        for dx, dy in shifts_chunk:
            tested = 0
            ssd = 0
            best_local_ssd = float('inf')
            for bx, by in eval_band:
                sx = bx + dx
                sy = by + dy
                if 0 <= sx < width and 0 <= sy < height:
                    s_idx = sy * width + sx
                    if valid_mask[s_idx] == 1:
                        tested += 1
                        b_pix = (by * width + bx) * channels
                        s_pix = s_idx * channels
                        dr = img_bytes[b_pix] - img_bytes[s_pix]
                        dg = img_bytes[b_pix + 1] - img_bytes[s_pix + 1]
                        db = img_bytes[b_pix + 2] - img_bytes[s_pix + 2]
                        pt_ssd = dr * dr + dg * dg + db * db
                        ssd += pt_ssd
                        if pt_ssd < best_local_ssd:
                            best_local_ssd = pt_ssd
            if tested >= max(3, num_eval // 6):
                avg_err = ssd / float(tested)
                # Combine global average with local best segment fit
                combined_score = 0.5 * avg_err + 0.5 * best_local_ssd
                results.append((combined_score, (dx, dy)))
        return results

    ranked_shifts = []
    if all_candidate_shifts and eval_band:
        chunk_size = max(1, len(all_candidate_shifts) // cpu_cores)
        chunks = [all_candidate_shifts[i:i + chunk_size] for i in range(0, len(all_candidate_shifts), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cores) as executor:
            futures = [executor.submit(evaluate_shift_chunk, c) for c in chunks]
            for f in concurrent.futures.as_completed(futures):
                ranked_shifts.extend(f.result())

    ranked_shifts.sort(key=lambda item: item[0])
    dominant_shifts = [shift for _, shift in ranked_shifts[:10]]
    
    # Always include cardinal directional shifts
    for card_shift in [(sel_w, 0), (-sel_w, 0), (0, sel_h), (0, -sel_h)]:
        if card_shift not in dominant_shifts:
            dominant_shifts.append(card_shift)

    return SourceSelectionMap(width, height, valid_mask, known_centers, dominant_shifts)
