#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Benchmark Suite for GIMP 3 Content-Aware Fill Engines
"""

import math
import time
import array
import random
import heapq


def compute_gradients(img_bytes, width, height, channels=4):
    total = width * height
    gray = array.array('f', [0.0] * total)
    for i in range(total):
        idx = i * channels
        gray[i] = 0.299 * img_bytes[idx] + 0.587 * img_bytes[idx + 1] + 0.114 * img_bytes[idx + 2]

    gx = array.array('f', [0.0] * total)
    gy = array.array('f', [0.0] * total)

    for y in range(height):
        row = y * width
        row_prev = max(0, y - 1) * width
        row_next = min(height - 1, y + 1) * width
        for x in range(width):
            idx = row + x
            x_prev = max(0, x - 1)
            x_next = min(width - 1, x + 1)
            gx[idx] = (gray[row + x_next] - gray[row + x_prev]) * 0.5
            gy[idx] = (gray[row_next + x] - gray[row_prev + x]) * 0.5

    return gray, gx, gy


def inpaint_telea(img_bytes, mask_bytes, width, height, channels=4, radius=4):
    total = width * height
    flags = bytearray(total)
    dist = array.array('f', [1e6] * total)
    band_heap = []

    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if mask_bytes[idx] > 10:
                flags[idx] = 2
            else:
                flags[idx] = 0
                dist[idx] = 0.0

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

    while band_heap:
        d, px, py = heapq.heappop(band_heap)
        p_idx = py * width + px
        if flags[p_idx] != 1:
            continue
        flags[p_idx] = 0

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
                            w = 1.0 / (1.0 + math.sqrt(d_sq))
                            sum_weights += w
                            q_pix = q_idx * channels
                            for c in range(channels):
                                sum_cols[c] += w * img_bytes[q_pix + c]

        p_pix = p_idx * channels
        if sum_weights > 1e-6:
            for c in range(channels):
                img_bytes[p_pix + c] = max(0, min(255, int(sum_cols[c] / sum_weights + 0.5)))

        for ndx, ndy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = px + ndx
            ny = py + ndy
            if 0 <= nx < width and 0 <= ny < height:
                n_idx = ny * width + nx
                if flags[n_idx] == 2:
                    flags[n_idx] = 1
                    dist[n_idx] = dist[p_idx] + 1.0
                    heapq.heappush(band_heap, (dist[n_idx], nx, ny))


def inpaint_photoshop_em(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, em_passes=3):
    total = width * height
    r = max(2, patch_radius)
    min_dim = min(width, height)
    num_levels = 3 if min_dim >= 180 else (2 if min_dim >= 80 else 1)

    pyramid_images = [None] * num_levels
    pyramid_masks = [None] * num_levels
    pyramid_dims = [None] * num_levels

    pyramid_images[0] = bytearray(img_bytes)
    pyramid_masks[0] = bytearray(mask_bytes)
    pyramid_dims[0] = (width, height)

    for lvl in range(1, num_levels):
        prev_w, prev_h = pyramid_dims[lvl - 1]
        cur_w = max(4, prev_w // 2)
        cur_h = max(4, prev_h // 2)
        pyramid_dims[lvl] = (cur_w, cur_h)

        prev_img = pyramid_images[lvl - 1]
        prev_mask = pyramid_masks[lvl - 1]

        cur_img = bytearray(cur_w * cur_h * channels)
        cur_mask = bytearray(cur_w * cur_h)

        for y in range(cur_h):
            py0 = y * 2
            for x in range(cur_w):
                px0 = x * 2
                sum_c = [0] * channels
                mask_val = 0
                count = 0
                for dy in range(2):
                    sy = min(prev_h - 1, py0 + dy)
                    row_p = sy * prev_w
                    for dx in range(2):
                        sx = min(prev_w - 1, px0 + dx)
                        p_idx = row_p + sx
                        if prev_mask[p_idx] > 10:
                            mask_val = 255
                        p_pix = p_idx * channels
                        for c in range(channels):
                            sum_c[c] += prev_img[p_pix + c]
                        count += 1
                cur_idx = y * cur_w + x
                cur_pix = cur_idx * channels
                for c in range(channels):
                    cur_img[cur_pix + c] = sum_c[c] // count
                cur_mask[cur_idx] = mask_val

        pyramid_images[lvl] = cur_img
        pyramid_masks[lvl] = cur_mask

    nnf_x = None
    nnf_y = None

    for lvl in range(num_levels - 1, -1, -1):
        lw, lh = pyramid_dims[lvl]
        l_img = pyramid_images[lvl]
        l_mask = pyramid_masks[lvl]
        l_total = lw * lh
        l_r = max(2, int(r * (lw / float(width))))
        patch_size = 2 * l_r + 1

        hole_pixels = []
        known_centers = []

        for y in range(lh):
            row = y * lw
            for x in range(lw):
                idx = row + x
                if l_mask[idx] > 10:
                    hole_pixels.append((x, y))
                else:
                    if l_r <= x < lw - l_r and l_r <= y < lh - l_r:
                        known_centers.append((x, y))

        if not hole_pixels or not known_centers:
            continue

        num_known = len(known_centers)

        if lvl == num_levels - 1:
            inpaint_telea(l_img, l_mask, lw, lh, channels, radius=l_r + 1)
            nnf_x = array.array('h', [0] * l_total)
            nnf_y = array.array('h', [0] * l_total)
            for x, y in hole_pixels:
                idx = y * lw + x
                sx, sy = known_centers[random.randint(0, num_known - 1)]
                nnf_x[idx] = sx
                nnf_y[idx] = sy
        else:
            prev_w, prev_h = pyramid_dims[lvl + 1]
            new_nnf_x = array.array('h', [0] * l_total)
            new_nnf_y = array.array('h', [0] * l_total)
            scale_x = lw / float(prev_w)
            scale_y = lh / float(prev_h)

            for x, y in hole_pixels:
                idx = y * lw + x
                cx = min(prev_w - 1, int(x / scale_x))
                cy = min(prev_h - 1, int(y / scale_y))
                c_idx = cy * prev_w + cx
                sx = int(nnf_x[c_idx] * scale_x)
                sy = int(nnf_y[c_idx] * scale_y)
                sx = max(l_r, min(lw - 1 - l_r, sx))
                sy = max(l_r, min(lh - 1 - l_r, sy))
                if l_mask[sy * lw + sx] > 10:
                    sx, sy = known_centers[random.randint(0, num_known - 1)]
                new_nnf_x[idx] = sx
                new_nnf_y[idx] = sy

            nnf_x = new_nnf_x
            nnf_y = new_nnf_y

        nnf_dist = array.array('i', [0] * l_total)
        nnf_mode = array.array('b', [0] * l_total)

        passes_for_lvl = em_passes if lvl == 0 else max(2, em_passes - 1)

        for em_iter in range(passes_for_lvl):
            gray, gx, gy = compute_gradients(l_img, lw, lh, channels)

            offset_counts = {}
            for x, y in hole_pixels:
                idx = y * lw + x
                ox = nnf_x[idx] - x
                oy = nnf_y[idx] - y
                key = (ox // 3, oy // 3)
                offset_counts[key] = offset_counts.get(key, 0) + 1

            dominant_offsets = sorted(offset_counts.items(), key=lambda item: item[1], reverse=True)[:6]
            dominant_vecs = [(k[0] * 3, k[1] * 3) for k, _ in dominant_offsets]

            def compute_composite_ssd(tx, ty, sx, sy, mode=0, best_limit=float('inf')):
                ssd = 0
                t_row = (ty - l_r) * lw
                gw = 0.35

                for dy in range(patch_size):
                    s_dy = (patch_size - 1 - dy) if mode == 2 else dy
                    t_base = t_row + (tx - l_r)
                    s_base = (sy - l_r + s_dy) * lw + (sx - l_r)

                    for dx in range(patch_size):
                        s_dx = (patch_size - 1 - dx) if mode == 1 else dx
                        t_idx = t_base + dx
                        s_idx = s_base + s_dx

                        t_pix = t_idx * channels
                        s_pix = s_idx * channels

                        dr = l_img[t_pix] - l_img[s_pix]
                        dg = l_img[t_pix + 1] - l_img[s_pix + 1]
                        db = l_img[t_pix + 2] - l_img[s_pix + 2]
                        color_d = dr * dr + dg * dg + db * db

                        dgx = gx[t_idx] - gx[s_idx]
                        dgy = gy[t_idx] - gy[s_idx]
                        grad_d = dgx * dgx + dgy * dgy

                        ssd += int(color_d + gw * grad_d)
                        if ssd >= best_limit:
                            return ssd

                    t_row += lw

                return ssd

            for x, y in hole_pixels:
                idx = y * lw + x
                if l_r <= x < lw - l_r and l_r <= y < lh - l_r:
                    nnf_dist[idx] = compute_composite_ssd(x, y, nnf_x[idx], nnf_y[idx], nnf_mode[idx])
                else:
                    nnf_dist[idx] = 10000000

            max_dim = max(lw, lh)
            is_forward = (em_iter % 2 == 0)
            y_range = range(l_r, lh - l_r) if is_forward else range(lh - 1 - l_r, l_r - 1, -1)
            x_range = range(l_r, lw - l_r) if is_forward else range(lw - 1 - l_r, l_r - 1, -1)
            dir_mult = 1 if is_forward else -1

            for y in y_range:
                row = y * lw
                for x in x_range:
                    idx = row + x
                    if l_mask[idx] <= 10:
                        continue

                    best_sx = nnf_x[idx]
                    best_sy = nnf_y[idx]
                    best_m = nnf_mode[idx]
                    best_d = nnf_dist[idx]

                    nx = x - dir_mult
                    if l_r <= nx < lw - l_r:
                        n_idx = row + nx
                        cand_sx = nnf_x[n_idx] + dir_mult
                        cand_sy = nnf_y[n_idx]
                        if l_r <= cand_sx < lw - l_r and l_r <= cand_sy < lh - l_r:
                            if l_mask[cand_sy * lw + cand_sx] <= 10:
                                d = compute_composite_ssd(x, y, cand_sx, cand_sy, nnf_mode[n_idx], best_d)
                                if d < best_d:
                                    best_d = d
                                    best_sx, best_sy, best_m = cand_sx, cand_sy, nnf_mode[n_idx]

                    ny = y - dir_mult
                    if l_r <= ny < lh - l_r:
                        n_idx = ny * lw + x
                        cand_sx = nnf_x[n_idx]
                        cand_sy = nnf_y[n_idx] + dir_mult
                        if l_r <= cand_sx < lw - l_r and l_r <= cand_sy < lh - l_r:
                            if l_mask[cand_sy * lw + cand_sx] <= 10:
                                d = compute_composite_ssd(x, y, cand_sx, cand_sy, nnf_mode[n_idx], best_d)
                                if d < best_d:
                                    best_d = d
                                    best_sx, best_sy, best_m = cand_sx, cand_sy, nnf_mode[n_idx]

                    for dox, doy in dominant_vecs:
                        dsx = x + dox
                        dsy = y + doy
                        if l_r <= dsx < lw - l_r and l_r <= dsy < lh - l_r:
                            if l_mask[dsy * lw + dsx] <= 10:
                                d = compute_composite_ssd(x, y, dsx, dsy, 0, best_d)
                                if d < best_d:
                                    best_d = d
                                    best_sx, best_sy, best_m = dsx, dsy, 0

                    rad = max_dim // 2
                    while rad >= 1:
                        rx = best_sx + random.randint(-rad, rad)
                        ry = best_sy + random.randint(-rad, rad)
                        rx = max(l_r, min(lw - 1 - l_r, rx))
                        ry = max(l_r, min(lh - 1 - l_r, ry))
                        if l_mask[ry * lw + rx] <= 10:
                            d = compute_composite_ssd(x, y, rx, ry, best_m, best_d)
                            if d < best_d:
                                best_d = d
                                best_sx, best_sy = rx, ry
                        rad = int(rad * 0.5)

                    nnf_x[idx] = best_sx
                    nnf_y[idx] = best_sy
                    nnf_mode[idx] = best_m
                    nnf_dist[idx] = best_d

            vote_r = array.array('f', [0.0] * l_total)
            vote_g = array.array('f', [0.0] * l_total)
            vote_b = array.array('f', [0.0] * l_total)
            vote_w = array.array('f', [0.0] * l_total)

            for qx, qy in hole_pixels:
                q_idx = qy * lw + qx
                sx = nnf_x[q_idx]
                sy = nnf_y[q_idx]
                mode = nnf_mode[q_idx]
                d = nnf_dist[q_idx]
                w = 1.0 / (1.0 + 0.0005 * d)

                for dy in range(-l_r, l_r + 1):
                    py = qy + dy
                    if 0 <= py < lh:
                        s_dy = -dy if mode == 2 else dy
                        src_y = sy + s_dy
                        if 0 <= src_y < lh:
                            row_p = py * lw
                            row_s = src_y * lw
                            for dx in range(-l_r, l_r + 1):
                                px = qx + dx
                                if 0 <= px < lw:
                                    p_idx = row_p + px
                                    if l_mask[p_idx] > 10:
                                        s_dx = -dx if mode == 1 else dx
                                        src_x = sx + s_dx
                                        if 0 <= src_x < lw:
                                            s_pix = (row_s + src_x) * channels
                                            vote_r[p_idx] += w * l_img[s_pix]
                                            vote_g[p_idx] += w * l_img[s_pix + 1]
                                            vote_b[p_idx] += w * l_img[s_pix + 2]
                                            vote_w[p_idx] += w

            for px, py in hole_pixels:
                p_idx = py * lw + px
                if vote_w[p_idx] > 1e-6:
                    inv_w = 1.0 / vote_w[p_idx]
                    p_pix = p_idx * channels
                    l_img[p_pix] = max(0, min(255, int(vote_r[p_idx] * inv_w + 0.5)))
                    l_img[p_pix + 1] = max(0, min(255, int(vote_g[p_idx] * inv_w + 0.5)))
                    l_img[p_pix + 2] = max(0, min(255, int(vote_b[p_idx] * inv_w + 0.5)))

        if lvl > 0:
            next_w, next_h = pyramid_dims[lvl - 1]
            next_img = pyramid_images[lvl - 1]
            next_mask = pyramid_masks[lvl - 1]
            s_x = next_w / float(lw)
            s_y = next_h / float(lh)

            for y in range(next_h):
                src_y = min(lh - 1, int(y / s_y))
                row_n = y * next_w
                row_l = src_y * lw
                for x in range(next_w):
                    n_idx = row_n + x
                    if next_mask[n_idx] > 10:
                        src_x = min(lw - 1, int(x / s_x))
                        src_pix = (row_l + src_x) * channels
                        next_pix = n_idx * channels
                        for c in range(channels):
                            next_img[next_pix + c] = l_img[src_pix + c]

    final_img = pyramid_images[0]
    for i in range(total * channels):
        img_bytes[i] = final_img[i]


if __name__ == "__main__":
    w, h = 250, 250
    img = bytearray([128] * (w * h * 4))
    mask = bytearray(w * h)
    for y in range(100, 150):
        for x in range(100, 150):
            mask[y * w + x] = 255

    t0 = time.time()
    inpaint_photoshop_em(img, mask, w, h, 4, patch_radius=4, em_passes=3)
    print(f"Photoshop Multi-Scale EM test completed in {time.time() - t0:.3f}s!")
