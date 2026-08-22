#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Test Suite for the Modular Content-Aware Fill Pipeline.
"""

import sys
import os
import time

# Add content-aware-fill directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "content-aware-fill")))

from caf_engine import execute_content_aware_fill_pipeline

def create_synthetic_bookshelf_test():
    width = 400
    height = 400
    channels = 4
    img = bytearray(width * height * channels)
    mask = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            pix = idx * channels
            # Shelf bar at y = 200..220
            if 200 <= y <= 220:
                img[pix] = 120
                img[pix + 1] = 80
                img[pix + 2] = 50
                img[pix + 3] = 255
            # Books in bottom shelf y = 221..380
            elif 221 < y < 380:
                book_idx = (x // 25) % 5
                colors = [
                    (180, 40, 40),
                    (40, 140, 180),
                    (40, 180, 60),
                    (200, 180, 40),
                    (140, 60, 180)
                ]
                c = colors[book_idx]
                img[pix] = c[0]
                img[pix + 1] = c[1]
                img[pix + 2] = c[2]
                img[pix + 3] = 255
            else:
                # Wall
                img[pix] = 210
                img[pix + 1] = 200
                img[pix + 2] = 180
                img[pix + 3] = 255

    # Hole covering the shelf bar and books in bottom-left x = 100..260, y = 180..340
    for y in range(180, 340):
        for x in range(100, 260):
            mask[y * width + x] = 255

    return img, mask, width, height, channels


def test_bookshelf_pipeline():
    print("Running Test 1: Modular Pipeline on Bookshelf & Shelf Continuation...")
    img, mask, w, h, ch = create_synthetic_bookshelf_test()

    t0 = time.time()
    res = execute_content_aware_fill_pipeline(
        img, mask, w, h, channels=ch, patch_radius=4,
        sample_source="auto", quality_preset="balanced"
    )
    t1 = time.time()
    print(f"Test 1 completed in {t1 - t0:.3f}s")

    # Check that the shelf bar at y=210 across x=100..260 was reconstructed
    shelf_matches = sum(1 for x in range(100, 260) if abs(res[(210 * w + x) * ch] - 120) < 35)
    print(f"Shelf bar continuity score: {shelf_matches}/160")
    assert shelf_matches >= 140, f"Shelf continuity too low: {shelf_matches}"
    print("Test 1 PASSED: Shelf bar successfully continued across hole!")


def test_transparent_hole_pipeline():
    print("\nRunning Test 2: Modular Pipeline on Transparent Hole (Alpha = 0)...")
    img, mask, w, h, ch = create_synthetic_bookshelf_test()

    # Make hole transparent
    for y in range(180, 340):
        for x in range(100, 260):
            pix = (y * w + x) * ch
            img[pix + 3] = 0  # Alpha = 0

    t0 = time.time()
    res = execute_content_aware_fill_pipeline(
        img, mask, w, h, channels=ch, patch_radius=4,
        sample_source="auto", quality_preset="fast"
    )
    t1 = time.time()
    print(f"Test 2 completed in {t1 - t0:.3f}s")

    # Verify all synthesized pixels have alpha = 255
    transparent_count = sum(1 for y in range(180, 340) for x in range(100, 260) if res[(y * w + x) * ch + 3] < 250)
    print(f"Unfilled transparent pixels: {transparent_count}")
    assert transparent_count == 0, f"Found {transparent_count} unfilled transparent pixels"
    print("Test 2 PASSED: Transparent selection filled with 100% opaque texture!")


if __name__ == "__main__":
    test_bookshelf_pipeline()
    test_transparent_hole_pipeline()
    print("\nALL PIPELINE TESTS PASSED!")
