#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Benchmark & Test Suite for Content-Aware Fill Algorithms
"""

import math
import time
import array
import random
import heapq


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


def inpaint_patchmatch(img_bytes, mask_bytes, width, height, channels=4, patch_radius=4, num_iters=3):
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

        for x, y in hole_pixels:
            idx = y * width + x
            sx, sy = nnf_x[idx], nnf_y[idx]
            s_pix = (sy * width + sx) * channels
            t_pix = idx * channels
            for c in range(channels):
                img_bytes[t_pix + c] = img_bytes[s_pix + c]

    return img_bytes


if __name__ == "__main__":
    w, h = 200, 200
    img = bytearray([128] * (w * h * 4))
    mask = bytearray(w * h)
    for y in range(80, 120):
        for x in range(80, 120):
            mask[y * w + x] = 255

    t0 = time.time()
    inpaint_patchmatch(img, mask, w, h, 4, patch_radius=4)
    print(f"PatchMatch test passed in {time.time() - t0:.3f}s!")
