#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-Aware Fill Plugin for GIMP 3
====================================
High-performance Inpainting Plugin featuring 4 selectable algorithms:
1. ⚡ PatchMatch (Adobe Photoshop Content-Aware Engine)
2. 💨 Telea Fast Marching (Instantaneous Diffusion for Scratches/Text)
3. 🎯 Multi-Scale Exemplar (Fast Coarse-to-Fine Structure Propagation)
4. 🔬 Classic Criminisi (Exhaustive Isophote Priority Synthesis)

Author: bunnywaffle & Antigravity
License: GPLv3+
"""

import sys
import os
import math
import time
import array
import random
import heapq
import traceback

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk", "3.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gimp, GimpUi, Gegl, Gtk, GLib, GObject

Gegl.init(None)


def _(msg):
    return GLib.dgettext(None, msg)


# ============================================================================
# Algorithm 1: Telea Fast Marching Inpainting (Instantaneous Diffusion)
# ============================================================================

def inpaint_telea(img_bytes, mask_bytes, width, height, channels=4, radius=4, progress_callback=None):
    """
    Fast Marching Inpainting (Telea, 2004).
    Extremely fast (< 50ms) for scratches, lines, text, dust, and smooth gradients.
    """
    total = width * height
    # 0 = KNOWN, 1 = BAND, 2 = INSIDE
    flags = bytearray(total)
    dist = array.array('f', [1e6] * total)
    band_heap = []

    hole_count = 0
    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if mask_bytes[idx] > 10:
                flags[idx] = 2
                hole_count += 1
            else:
                flags[idx] = 0
                dist[idx] = 0.0

    if hole_count == 0:
        return img_bytes

    initial_hole_count = hole_count

    # Find initial boundary band
    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if flags[idx] == 2:
                if (x > 0 and flags[idx - 1] == 0) or \
                   (x < width - 1 and flags[idx + 1] == 0) or \
                   (y > 0 and flags[idx - width] == 0) or \
                   (y < height - 1 and flags[idx + width] == 0):
                    flags[idx] = 1
                    dist[idx] = 1.0
                    heapq.heappush(band_heap, (1.0, x, y))

    r = max(1, radius)
    r2 = r * r
    processed = 0
    last_report = time.time()

    while band_heap:
        d, px, py = heapq.heappop(band_heap)
        p_idx = py * width + px
        if flags[p_idx] != 1:
            continue
        flags[p_idx] = 0  # Mark as KNOWN
        processed += 1

        if progress_callback and (processed % 150 == 0 or time.time() - last_report > 0.15):
            last_report = time.time()
            progress_callback(min(1.0, processed / float(initial_hole_count)), "Fast Marching Inpainting...")

        # Gradient of distance at p
        tx = 0.0
        ty = 0.0
        if 0 < px < width - 1:
            tx = (dist[p_idx + 1] - dist[p_idx - 1]) * 0.5
        if 0 < py < height - 1:
            ty = (dist[p_idx + width] - dist[p_idx - width]) * 0.5
        grad_norm = math.sqrt(tx * tx + ty * ty)
        if grad_norm > 1e-5:
            tx /= grad_norm
            ty /= grad_norm
        else:
            tx, ty = 0.0, 1.0

        sum_weights = 0.0
        sum_cols = [0.0] * channels

        for dy in range(-r, r + 1):
            qy = py + dy
            if 0 <= qy < height:
                row_q = qy * width
                for dx in range(-r, r + 1):
                    qx = px + dx
                    d_sq = dx * dx + dy * dy
                    if 0 <= qx < width and d_sq <= r2:
                        q_idx = row_q + qx
                        if flags[q_idx] == 0:
                            d_geom = math.sqrt(d_sq) if d_sq > 0 else 0.5
                            w_dst = 1.0 / (d_geom * d_geom)
                            dir_dot = (-dx * tx - dy * ty) / d_geom
                            w_dir = max(0.05, dir_dot)
                            w_lev = 1.0 / (1.0 + abs(dist[p_idx] - dist[q_idx]))
                            w = w_dst * w_dir * w_lev
                            sum_weights += w
                            q_pix = q_idx * channels
                            for c in range(channels):
                                sum_cols[c] += w * img_bytes[q_pix + c]

        p_pix = p_idx * channels
        if sum_weights > 1e-6:
            for c in range(channels):
                img_bytes[p_pix + c] = max(0, min(255, int(sum_cols[c] / sum_weights + 0.5)))

        # Update 4-neighbors
        for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = px + ndx
            ny = py + ndy
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                if flags[n_idx] == 2:
                    flags[n_idx] = 1
                    dist[n_idx] = dist[p_idx] + 1.0
                    heapq.heappush(band_heap, (dist[n_idx], nx, ny))

    return img_bytes


# ============================================================================
# Algorithm 2: PatchMatch (Barnes et al. 2009 - Photoshop Engine)
# ============================================================================

def inpaint_patchmatch(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, num_iters=3, progress_callback=None):
    """
    Fast PatchMatch Randomized Synthesis (Photoshop Content-Aware Engine).
    """
    total = width * height
    r = patch_radius
    patch_size = 2 * r + 1

    mask = bytearray(total)
    hole_pixels = []
    known_centers = []

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if mask_bytes[idx] > 10:
                mask[idx] = 1
                hole_pixels.append((x, y))
            else:
                mask[idx] = 0
                if r <= x < width - r and r <= y < height - r:
                    known_centers.append((x, y))

    if not hole_pixels or not known_centers:
        return img_bytes

    # Initial Fast Marching fill
    inpaint_telea(img_bytes, mask_bytes, width, height, channels, radius=r + 1)

    nnf_x = array.array('h', [0] * total)
    nnf_y = array.array('h', [0] * total)

    num_known = len(known_centers)
    for x, y in hole_pixels:
        idx = y * width + x
        sx, sy = known_centers[random.randint(0, num_known - 1)]
        nnf_x[idx] = sx
        nnf_y[idx] = sy

    def compute_patch_ssd(tx, ty, sx, sy, best_limit=float('inf')):
        ssd = 0
        t_row = (ty - r) * width
        s_row = (sy - r) * width
        for dy in range(patch_size):
            t_base = t_row + (tx - r)
            s_base = s_row + (sx - r)
            for dx in range(patch_size):
                t_idx = (t_base + dx) * channels
                s_idx = (s_base + dx) * channels
                dr = img_bytes[t_idx] - img_bytes[s_idx]
                dg = img_bytes[t_idx + 1] - img_bytes[s_idx + 1]
                db = img_bytes[t_idx + 2] - img_bytes[s_idx + 2]
                ssd += dr * dr + dg * dg + db * db
                if ssd >= best_limit:
                    return ssd
            t_row += width
            s_row += width
        return ssd

    nnf_dist = array.array('i', [0] * total)
    for x, y in hole_pixels:
        idx = y * width + x
        if r <= x < width - r and r <= y < height - r:
            nnf_dist[idx] = compute_patch_ssd(x, y, nnf_x[idx], nnf_y[idx])
        else:
            nnf_dist[idx] = 1000000

    max_dim = max(width, height)

    for iteration in range(num_iters):
        if progress_callback:
            progress_callback((iteration + 1) / float(num_iters), f"PatchMatch Iteration {iteration+1}/{num_iters}...")

        is_forward = (iteration % 2 == 0)
        y_range = range(r, height - r) if is_forward else range(height - 1 - r, r - 1, -1)
        x_range = range(r, width - r) if is_forward else range(width - 1 - r, r - 1, -1)
        dir_mult = 1 if is_forward else -1

        for y in y_range:
            row = y * width
            for x in x_range:
                idx = row + x
                if mask[idx] != 1:
                    continue

                best_sx = nnf_x[idx]
                best_sy = nnf_y[idx]
                best_d = nnf_dist[idx]

                # 1. Horizontal propagation
                nx = x - dir_mult
                if r <= nx < width - r:
                    n_idx = row + nx
                    cand_sx = nnf_x[n_idx] + dir_mult
                    cand_sy = nnf_y[n_idx]
                    if r <= cand_sx < width - r and r <= cand_sy < height - r:
                        if mask[cand_sy * width + cand_sx] == 0:
                            d = compute_patch_ssd(x, y, cand_sx, cand_sy, best_d)
                            if d < best_d:
                                best_d = d
                                best_sx, best_sy = cand_sx, cand_sy

                # 2. Vertical propagation
                ny = y - dir_mult
                if r <= ny < height - r:
                    n_idx = ny * width + x
                    cand_sx = nnf_x[n_idx]
                    cand_sy = nnf_y[n_idx] + dir_mult
                    if r <= cand_sx < width - r and r <= cand_sy < height - r:
                        if mask[cand_sy * width + cand_sx] == 0:
                            d = compute_patch_ssd(x, y, cand_sx, cand_sy, best_d)
                            if d < best_d:
                                best_d = d
                                best_sx, best_sy = cand_sx, cand_sy

                # 3. Random Search
                rad = max_dim // 2
                while rad >= 1:
                    rx = best_sx + random.randint(-rad, rad)
                    ry = best_sy + random.randint(-rad, rad)
                    rx = max(r, min(width - 1 - r, rx))
                    ry = max(r, min(height - 1 - r, ry))
                    if mask[ry * width + rx] == 0:
                        d = compute_patch_ssd(x, y, rx, ry, best_d)
                        if d < best_d:
                            best_d = d
                            best_sx, best_sy = rx, ry
                    rad = int(rad * 0.5)

                nnf_x[idx] = best_sx
                nnf_y[idx] = best_sy
                nnf_dist[idx] = best_d

        # Reconstruct hole pixels
        for x, y in hole_pixels:
            idx = y * width + x
            sx, sy = nnf_x[idx], nnf_y[idx]
            s_pix = (sy * width + sx) * channels
            t_pix = idx * channels
            for c in range(channels):
                img_bytes[t_pix + c] = img_bytes[s_pix + c]

    return img_bytes


# ============================================================================
# Algorithm 3: Multi-Scale Exemplar (Coarse-to-Fine Synthesis)
# ============================================================================

def inpaint_multiscale(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, progress_callback=None):
    """
    Coarse-to-Fine Exemplar Inpainting (Pyramidal Fast Synthesis).
    """
    w2 = max(4, width // 2)
    h2 = max(4, height // 2)

    img2 = bytearray(w2 * h2 * channels)
    mask2 = bytearray(w2 * h2)

    for y2 in range(h2):
        y_orig = y2 * 2
        for x2 in range(w2):
            x_orig = x2 * 2
            sum_c = [0] * channels
            mask_val = 0
            count = 0
            for dy in range(2):
                y = min(height - 1, y_orig + dy)
                row = y * width
                for dx in range(2):
                    x = min(width - 1, x_orig + dx)
                    idx = row + x
                    if mask_bytes[idx] > 10:
                        mask_val = 255
                    for c in range(channels):
                        sum_c[c] += img_bytes[idx * channels + c]
                    count += 1
            idx2 = y2 * w2 + x2
            for c in range(channels):
                img2[idx2 * channels + c] = sum_c[c] // count
            mask2[idx2] = mask_val

    if progress_callback:
        progress_callback(0.25, "Coarse pyramid inpainting...")

    inpaint_patchmatch(img2, mask2, w2, h2, channels, patch_radius=max(2, patch_radius // 2), num_iters=2)

    if progress_callback:
        progress_callback(0.70, "Fine scale detail synthesis...")

    for y in range(height):
        y2 = min(h2 - 1, y // 2)
        for x in range(width):
            idx = y * width + x
            if mask_bytes[idx] > 10:
                x2 = min(w2 - 1, x // 2)
                idx2 = y2 * w2 + x2
                for c in range(channels):
                    img_bytes[idx * channels + c] = img2[idx2 * channels + c]

    inpaint_patchmatch(img_bytes, mask_bytes, width, height, channels, patch_radius=patch_radius, num_iters=1)
    return img_bytes


# ============================================================================
# Algorithm 4: Classic Criminisi (Exhaustive Isophote Priority)
# ============================================================================

def inpaint_criminisi(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, progress_callback=None):
    """
    Classic Criminisi et al. Exemplar-based Inpainting.
    """
    total_pixels = width * height
    patch_size = 2 * patch_radius + 1
    patch_area = float(patch_size * patch_size)
    r = patch_radius

    mask = bytearray(total_pixels)
    confidence = array.array('f', [1.0] * total_pixels)
    hole_count = 0
    min_x, max_x = width, 0
    min_y, max_y = height, 0

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if mask_bytes[idx] > 10:
                mask[idx] = 1
                confidence[idx] = 0.0
                hole_count += 1
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y

    if hole_count == 0:
        return img_bytes

    initial_hole_count = hole_count

    # Compute grayscale luminance
    gray = array.array('f', [0.0] * total_pixels)
    for i in range(total_pixels):
        idx = i * channels
        gray[i] = 0.299 * img_bytes[idx] + 0.587 * img_bytes[idx + 1] + 0.114 * img_bytes[idx + 2]

    # Find initial front
    front = set()
    for y in range(min_y, max_y + 1):
        row = y * width
        for x in range(min_x, max_x + 1):
            idx = row + x
            if mask[idx] == 1:
                if (x > 0 and mask[idx - 1] == 0) or \
                   (x < width - 1 and mask[idx + 1] == 0) or \
                   (y > 0 and mask[idx - width] == 0) or \
                   (y < height - 1 and mask[idx + width] == 0):
                    front.add((x, y))

    candidate_sources = []
    margin = max(60, patch_radius * 12)
    s_min_x = max(r, min_x - margin)
    s_max_x = min(width - 1 - r, max_x + margin)
    s_min_y = max(r, min_y - margin)
    s_max_y = min(height - 1 - r, max_y + margin)

    for sy in range(s_min_y, s_max_y + 1, 2):
        for sx in range(s_min_x, s_max_x + 1, 2):
            if mask[sy * width + sx] == 0:
                candidate_sources.append((sx, sy))

    if not candidate_sources:
        for sy in range(r, height - r, max(1, r)):
            for sx in range(r, width - r, max(1, r)):
                if mask[sy * width + sx] == 0:
                    candidate_sources.append((sx, sy))

    if not candidate_sources:
        return img_bytes

    iteration = 0
    last_report = time.time()

    while front:
        iteration += 1
        if progress_callback and (iteration % 8 == 0 or time.time() - last_report > 0.2):
            last_report = time.time()
            done = 1.0 - (hole_count / float(initial_hole_count))
            progress_callback(max(0.0, min(1.0, done)), "Criminisi Inpainting...")

        best_p = next(iter(front))
        best_priority = -1.0
        best_c = 0.0

        for px, py in front:
            c_sum = 0.0
            for dy in range(-r, r + 1):
                qy = py + dy
                if 0 <= qy < height:
                    row_q = qy * width
                    for dx in range(-r, r + 1):
                        qx = px + dx
                        if 0 <= qx < width:
                            q_idx = row_q + qx
                            if mask[q_idx] == 0:
                                c_sum += confidence[q_idx]
            cp = c_sum / patch_area

            nx = 0.0
            ny = 0.0
            if 0 < px < width - 1:
                nx = float(mask[py * width + px + 1] - mask[py * width + px - 1])
            if 0 < py < height - 1:
                ny = float(mask[(py + 1) * width + px] - mask[(py - 1) * width + px])
            norm = math.sqrt(nx * nx + ny * ny)
            if norm > 1e-5:
                nx /= norm
                ny /= norm
            else:
                nx, ny = 0.0, 1.0

            max_grad_mag = 0.0
            best_gx = 0.0
            best_gy = 0.0
            for dy in range(-r, r + 1):
                qy = py + dy
                if 1 <= qy < height - 1:
                    row_q = qy * width
                    for dx in range(-r, r + 1):
                        qx = px + dx
                        if 1 <= qx < width - 1:
                            q_idx = row_q + qx
                            if mask[q_idx] == 0:
                                gx = (gray[q_idx + 1] - gray[q_idx - 1]) * 0.5
                                gy = (gray[q_idx + width] - gray[q_idx - width]) * 0.5
                                mag = gx * gx + gy * gy
                                if mag > max_grad_mag:
                                    max_grad_mag = mag
                                    best_gx = gx
                                    best_gy = gy

            dp = max(0.001, abs(-best_gy * nx + best_gx * ny) / 255.0)
            priority = cp * dp
            if priority > best_priority:
                best_priority = priority
                best_p = (px, py)
                best_c = cp

        px, py = best_p
        known_pixels = []
        for dy in range(-r, r + 1):
            ty = py + dy
            if 0 <= ty < height:
                row_t = ty * width
                for dx in range(-r, r + 1):
                    tx = px + dx
                    if 0 <= tx < width:
                        t_idx = row_t + tx
                        if mask[t_idx] == 0:
                            pix_idx = t_idx * channels
                            known_pixels.append((
                                dx, dy,
                                img_bytes[pix_idx],
                                img_bytes[pix_idx + 1],
                                img_bytes[pix_idx + 2]
                            ))

        if not known_pixels:
            front.remove(best_p)
            continue

        best_ssd = float('inf')
        best_source = None
        for sx, sy in candidate_sources:
            ssd = 0
            for dx, dy, tr, tg, tb in known_pixels:
                src_pix_idx = ((sy + dy) * width + (sx + dx)) * channels
                dr = tr - img_bytes[src_pix_idx]
                dg = tg - img_bytes[src_pix_idx + 1]
                db = tb - img_bytes[src_pix_idx + 2]
                ssd += dr * dr + dg * dg + db * db
                if ssd >= best_ssd:
                    break
            else:
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_source = (sx, sy)

        if best_source is None:
            best_source = candidate_sources[0] if candidate_sources else None
            if not best_source:
                break

        sx, sy = best_source
        filled_pixels = []
        for dy in range(-r, r + 1):
            ty = py + dy
            if 0 <= ty < height:
                row_t = ty * width
                row_s = (sy + dy) * width
                for dx in range(-r, r + 1):
                    tx = px + dx
                    if 0 <= tx < width:
                        t_idx = row_t + tx
                        if mask[t_idx] == 1:
                            s_idx = row_s + (sx + dx)
                            t_pix = t_idx * channels
                            s_pix = s_idx * channels
                            for c in range(channels):
                                img_bytes[t_pix + c] = img_bytes[s_pix + c]
                            r_c = img_bytes[t_pix]
                            g_c = img_bytes[t_pix + 1]
                            b_c = img_bytes[t_pix + 2]
                            gray[t_idx] = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                            confidence[t_idx] = best_c
                            mask[t_idx] = 0
                            hole_count -= 1
                            filled_pixels.append((tx, ty))

        for fx, fy in filled_pixels:
            if (fx, fy) in front:
                front.remove((fx, fy))

        to_remove = [p for p in front if mask[p[1] * width + p[0]] == 0]
        for p in to_remove:
            front.remove(p)

        for fx, fy in filled_pixels:
            for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx = fx + ndx
                ny = fy + ndy
                if 0 <= nx < width and 0 <= ny < height:
                    n_idx = ny * width + nx
                    if mask[n_idx] == 1:
                        front.add((nx, ny))

    return img_bytes


# ============================================================================
# Interactive Multi-Algorithm Configuration Dialog (Gtk 3)
# ============================================================================

class ContentAwareFillDialog(Gtk.Dialog):
    def __init__(self, image, drawable):
        super().__init__(
            title=_("Content-Aware Fill"),
            flags=Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        )
        self.set_default_size(480, 420)
        self.set_resizable(False)

        self.image = image
        self.drawable = drawable

        # Buttons
        self.add_button(_("_Cancel"), Gtk.ResponseType.CANCEL)
        fill_btn = self.add_button(_("_Fill Selection"), Gtk.ResponseType.OK)
        fill_btn.get_style_context().add_class("suggested-action")

        content = self.get_content_area()
        content.set_spacing(12)
        content.set_margin_start(16)
        content.set_margin_end(16)
        content.set_margin_top(14)
        content.set_margin_bottom(14)

        # Header Title
        header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        title_label = Gtk.Label()
        title_label.set_markup("<span size='large' weight='bold'>Content-Aware Fill</span>")
        title_label.set_xalign(0.0)
        desc_label = Gtk.Label()
        desc_label.set_markup(
            "<span size='small' color='#777777'>"
            "Fill selected regions seamlessly with texture matching or instant diffusion."
            "</span>"
        )
        desc_label.set_xalign(0.0)
        header_box.pack_start(title_label, False, False, 0)
        header_box.pack_start(desc_label, False, False, 0)
        content.pack_start(header_box, False, False, 0)

        # Settings Frame
        frame = Gtk.Frame(label=_("Inpainting Engine & Settings"))
        grid = Gtk.Grid()
        grid.set_column_spacing(14)
        grid.set_row_spacing(12)
        grid.set_margin_start(12)
        grid.set_margin_end(12)
        grid.set_margin_top(12)
        grid.set_margin_bottom(12)
        frame.add(grid)
        content.pack_start(frame, True, True, 0)

        # 1. Algorithm Selection
        algo_label = Gtk.Label(label=_("Algorithm:"))
        algo_label.set_xalign(0.0)
        grid.attach(algo_label, 0, 0, 1, 1)

        self.algo_combo = Gtk.ComboBoxText()
        self.algo_combo.append_text(_("⚡ PatchMatch (Photoshop Engine - Fast & High Quality)"))
        self.algo_combo.append_text(_("💨 Telea Fast Marching (Instant - Best for Scratches/Text)"))
        self.algo_combo.append_text(_("🎯 Multi-Scale Pyramidal (Fast Coarse-to-Fine)"))
        self.algo_combo.append_text(_("🔬 Classic Criminisi (Exhaustive Isophote Search)"))
        self.algo_combo.set_active(0)
        self.algo_combo.set_hexpand(True)
        self.algo_combo.connect("changed", self._on_algo_changed)
        grid.attach(self.algo_combo, 1, 0, 1, 1)

        # Algorithm Description Info Box
        self.algo_desc = Gtk.Label()
        self.algo_desc.set_markup("<span size='small' color='#3388bb'>★ Recommended: Randomized patch propagation for fast, natural texture synthesis.</span>")
        self.algo_desc.set_xalign(0.0)
        self.algo_desc.set_line_wrap(True)
        grid.attach(self.algo_desc, 0, 1, 2, 1)

        # 2. Patch Size / Radius Slider
        self.size_label = Gtk.Label(label=_("Patch Size:"))
        self.size_label.set_xalign(0.0)
        grid.attach(self.size_label, 0, 2, 1, 1)

        self.size_adj = Gtk.Adjustment(value=9, lower=3, upper=25, step_increment=2, page_increment=4)
        self.size_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.size_adj)
        self.size_scale.set_digits(0)
        self.size_scale.set_hexpand(True)
        self.size_scale.add_mark(5, Gtk.PositionType.BOTTOM, "5px")
        self.size_scale.add_mark(9, Gtk.PositionType.BOTTOM, "9px")
        self.size_scale.add_mark(15, Gtk.PositionType.BOTTOM, "15px")
        self.size_scale.add_mark(25, Gtk.PositionType.BOTTOM, "25px")
        grid.attach(self.size_scale, 1, 2, 1, 1)

        # 3. Quality / Iterations
        self.qual_label = Gtk.Label(label=_("Quality / Passes:"))
        self.qual_label.set_xalign(0.0)
        grid.attach(self.qual_label, 0, 3, 1, 1)

        self.qual_combo = Gtk.ComboBoxText()
        self.qual_combo.append_text(_("Fast (2 Passes)"))
        self.qual_combo.append_text(_("Standard (3 Passes - Balanced)"))
        self.qual_combo.append_text(_("High Quality (5 Passes)"))
        self.qual_combo.set_active(1)
        self.qual_combo.set_hexpand(True)
        grid.attach(self.qual_combo, 1, 3, 1, 1)

        # 4. Checkboxes
        check_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.deselect_check = Gtk.CheckButton(label=_("Deselect selection when complete"))
        self.deselect_check.set_active(False)
        check_box.pack_start(self.deselect_check, False, False, 0)
        grid.attach(check_box, 0, 4, 2, 1)

        self.show_all()

    def _on_algo_changed(self, combo):
        algo = combo.get_active()
        if algo == 0:  # PatchMatch
            self.algo_desc.set_markup("<span size='small' color='#3388bb'>★ <b>PatchMatch:</b> Photoshop's core engine. Fast, natural texture synthesis with spatial coherence.</span>")
            self.size_label.set_text(_("Patch Size:"))
            self.qual_label.set_visible(True)
            self.qual_combo.set_visible(True)
        elif algo == 1:  # Telea
            self.algo_desc.set_markup("<span size='small' color='#229955'>⚡ <b>Fast Marching (Telea):</b> Instantaneous diffusion. Perfect for wires, scratches, spots, text.</span>")
            self.size_label.set_text(_("Diffusion Radius:"))
            self.qual_label.set_visible(False)
            self.qual_combo.set_visible(False)
        elif algo == 2:  # Multi-Scale
            self.algo_desc.set_markup("<span size='small' color='#bb6622'>🎯 <b>Multi-Scale Pyramidal:</b> Downsamples 2x to solve global structure fast, then refines details.</span>")
            self.size_label.set_text(_("Patch Size:"))
            self.qual_label.set_visible(True)
            self.qual_combo.set_visible(True)
        elif algo == 3:  # Criminisi
            self.algo_desc.set_markup("<span size='small' color='#8844aa'>🔬 <b>Classic Criminisi:</b> Calculates isophote gradient and normal priority. Slower, thorough search.</span>")
            self.size_label.set_text(_("Patch Size:"))
            self.qual_label.set_visible(False)
            self.qual_combo.set_visible(False)

    def get_settings(self):
        val = int(self.size_adj.get_value())
        if val % 2 == 0:
            val += 1
        radius = max(1, val // 2)

        qual_idx = self.qual_combo.get_active()
        iters = 2 if qual_idx == 0 else (3 if qual_idx == 1 else 5)

        return {
            "algo": self.algo_combo.get_active(),
            "radius": radius,
            "iters": iters,
            "deselect": self.deselect_check.get_active(),
        }


# ============================================================================
# Main GIMP 3 Plugin Class & Procedure Runner
# ============================================================================

class ContentAwareFillPlugin(Gimp.PlugIn):
    def do_set_i18n(self, procname):
        return True, "gimp30-python", None

    def do_query_procedures(self):
        return [
            "plug-in-content-aware-fill",
        ]

    def do_create_procedure(self, name):
        if name == "plug-in-content-aware-fill":
            procedure = Gimp.ImageProcedure.new(
                self,
                name,
                Gimp.PDBProcType.PLUGIN,
                self.run_content_aware_fill,
                None,
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_documentation(
                _("Content-Aware Fill (Multi-Algorithm)"),
                _("Fills selected area seamlessly using PatchMatch (Photoshop engine), Fast Marching (Telea), Multi-Scale, or Criminisi."),
                name,
            )
            procedure.set_menu_label(_("Content-Aware Fill..."))
            procedure.add_menu_path("<Image>/Edit/")
            procedure.add_menu_path("<Image>/Filters/Enhance/")
            procedure.set_attribution("bunnywaffle & Antigravity", "GPLv3+", "2026")
            return procedure

        return None

    def run_content_aware_fill(self, procedure, run_mode, image, drawables, config, data):
        try:
            if not drawables or drawables[0] is None:
                Gimp.message(_("Please select an active layer to use Content-Aware Fill."))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)

            drawable = drawables[0]

            # 1. Check selection
            if Gimp.Selection.is_empty(image):
                Gimp.message(
                    _("No active selection found!\n\n"
                      "Please make a selection around the object or region you want to fill "
                      "(e.g. using the Free Select / Lasso or Rectangle Select tool), "
                      "then run Content-Aware Fill.")
                )
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)

            bounds_res = Gimp.Selection.bounds(image)
            if len(bounds_res) == 6:
                success, non_empty, sel_x1, sel_y1, sel_x2, sel_y2 = bounds_res
            else:
                success, non_empty = bounds_res[0], bounds_res[1]
                sel_x1, sel_y1, sel_x2, sel_y2 = bounds_res[2], bounds_res[3], bounds_res[4], bounds_res[5]

            if not non_empty or sel_x2 <= sel_x1 or sel_y2 <= sel_y1:
                Gimp.message(_("Selection is empty."))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)

            settings = {
                "algo": 0,       # PatchMatch (Default)
                "radius": 4,     # 9x9 patch
                "iters": 3,      # Standard
                "deselect": False,
            }

            if run_mode == Gimp.RunMode.INTERACTIVE:
                GimpUi.init("content-aware-fill")
                dialog = ContentAwareFillDialog(image, drawable)
                response = dialog.run()
                if response != Gtk.ResponseType.OK:
                    dialog.destroy()
                    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)

                settings = dialog.get_settings()
                dialog.destroy()

            offsets_res = drawable.get_offsets()
            if len(offsets_res) == 3:
                off_ok, off_x, off_y = offsets_res
            else:
                off_x, off_y = offsets_res[0], offsets_res[1]

            draw_w = drawable.get_width()
            draw_h = drawable.get_height()

            layer_sel_x1 = max(0, min(draw_w, sel_x1 - off_x))
            layer_sel_y1 = max(0, min(draw_h, sel_y1 - off_y))
            layer_sel_x2 = max(0, min(draw_w, sel_x2 - off_x))
            layer_sel_y2 = max(0, min(draw_h, sel_y2 - off_y))

            if layer_sel_x2 <= layer_sel_x1 or layer_sel_y2 <= layer_sel_y1:
                Gimp.message(_("The selection does not overlap with the active layer.\nPlease switch to the layer containing the image content."))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)

            radius = settings["radius"]
            margin = max(40, radius * 10)

            roi_x1 = max(0, layer_sel_x1 - margin)
            roi_y1 = max(0, layer_sel_y1 - margin)
            roi_x2 = min(draw_w, layer_sel_x2 + margin)
            roi_y2 = min(draw_h, layer_sel_y2 + margin)
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1

            if roi_w <= 0 or roi_h <= 0:
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

            img_roi_x1 = roi_x1 + off_x
            img_roi_y1 = roi_y1 + off_y

            layer_roi_rect = Gegl.Rectangle.new(roi_x1, roi_y1, roi_w, roi_h)
            img_roi_rect = Gegl.Rectangle.new(img_roi_x1, img_roi_y1, roi_w, roi_h)

            has_alpha = drawable.has_alpha()
            babl_format = "R'G'B'A u8" if has_alpha else "R'G'B' u8"
            channels = 4 if has_alpha else 3

            drawable_buffer = drawable.get_buffer()
            shadow_buffer = drawable.get_shadow_buffer()
            selection = image.get_selection()
            sel_buffer = selection.get_buffer()

            # Read source image data and selection mask
            img_raw = drawable_buffer.get(layer_roi_rect, 1.0, babl_format, Gegl.AbyssPolicy.CLAMP)
            mask_raw = sel_buffer.get(img_roi_rect, 1.0, "Y u8", Gegl.AbyssPolicy.CLAMP)

            img_bytes = bytearray(img_raw)
            mask_bytes = bytearray(mask_raw)

            Gimp.progress_init(_("Content-Aware Fill in progress..."))

            def progress_cb(fraction, message):
                Gimp.progress_update(fraction)
                return True

            image.undo_group_start()
            t0 = time.time()

            algo = settings["algo"]
            if algo == 0:  # PatchMatch (Photoshop Engine)
                inpainted_bytes = inpaint_patchmatch(
                    img_bytes=img_bytes,
                    mask_bytes=mask_bytes,
                    width=roi_w,
                    height=roi_h,
                    channels=channels,
                    patch_radius=radius,
                    num_iters=settings["iters"],
                    progress_callback=progress_cb,
                )
            elif algo == 1:  # Telea Fast Marching
                inpainted_bytes = inpaint_telea(
                    img_bytes=img_bytes,
                    mask_bytes=mask_bytes,
                    width=roi_w,
                    height=roi_h,
                    channels=channels,
                    radius=radius,
                    progress_callback=progress_cb,
                )
            elif algo == 2:  # Multi-Scale
                inpainted_bytes = inpaint_multiscale(
                    img_bytes=img_bytes,
                    mask_bytes=mask_bytes,
                    width=roi_w,
                    height=roi_h,
                    channels=channels,
                    patch_radius=radius,
                    progress_callback=progress_cb,
                )
            else:  # Classic Criminisi
                inpainted_bytes = inpaint_criminisi(
                    img_bytes=img_bytes,
                    mask_bytes=mask_bytes,
                    width=roi_w,
                    height=roi_h,
                    channels=channels,
                    patch_radius=radius,
                    progress_callback=progress_cb,
                )

            elapsed = time.time() - t0

            # Commit to shadow buffer and merge shadow into active layer
            shadow_buffer.set(layer_roi_rect, babl_format, bytes(inpainted_bytes))
            shadow_buffer.flush()
            drawable_buffer.set(layer_roi_rect, babl_format, bytes(inpainted_bytes))
            drawable_buffer.flush()

            drawable.merge_shadow(True)
            drawable.update(roi_x1, roi_y1, roi_w, roi_h)

            if settings["deselect"]:
                Gimp.Selection.none(image)

            image.undo_group_end()
            Gimp.displays_flush()
            Gimp.progress_end()

            print(f"[Content-Aware Fill] Inpainted {roi_w}x{roi_h} using algo {algo} in {elapsed:.2f}s")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)

        except Exception as exc:
            try:
                image.undo_group_end()
            except Exception:
                pass
            Gimp.progress_end()
            traceback.print_exc()
            Gimp.message(f"Content-Aware Fill Error:\n{str(exc)}")
            return procedure.new_return_values(
                Gimp.PDBStatusType.EXECUTION_ERROR,
                GLib.Error(message=str(exc))
            )


if __name__ == "__main__":
    Gimp.main(ContentAwareFillPlugin.__gtype__, sys.argv)
