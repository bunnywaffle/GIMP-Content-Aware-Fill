# GIMP Content-Aware Fill

[![GIMP 3](https://img.shields.io/badge/GIMP-3.0%20%7C%203.2-orange.svg)](https://www.gimp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

A state-of-the-art **Photoshop-grade Content-Aware Fill** plugin for **GIMP 3** implementing the complete modern non-AI inpainting pipeline: **Multi-Scale Gaussian Pyramid + Generalized PatchMatch + Wexler EM Global Coherence Optimization + He & Sun Dominant Offset Prior + Gradient/Edge Cost + Poisson Seam Healing**.

---

## 🔬 The Engine Architecture

```
                    ┌─────────────────────────┐
                    │       Input Image       │
                    │      + User Mask Ω      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Multi-Channel Features  │
                    │  - Grayscale & Color    │
                    │  - Gradients (Gx, Gy)   │
                    │  - Structural Edges     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Multi-Scale Pyramid   │
                    │   Coarse (1/4) Level    │
                    │         │               │
                    │   Medium (1/2) Level    │
                    │         │               │
                    │    Fine (1/1) Level     │
                    └────────────┬────────────┘
                                 │
    ┌────────────────────────────┴────────────────────────────┐
    │                                                         │
    ▼ (Per Pyramid Level)                                     ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│       E-Step: PatchMatch NNF         │     │         M-Step: Wexler Voting        │
│ - Spatial Propagation (L/R/U/D)      │────►│ - Weighted average of all overlapping│
│ - Dominant Offset Prior (He & Sun)   │     │   patches: I(p) = Σ w_i S_i / Σ w_i  │
│ - Generalized Rotation & Mirroring   │     │ - Update synthesized canvas          │
│ - Gradient & Edge-Weighted Distance  │◄────│ - Repeat EM loop (2-4 passes)        │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Poisson / Seam Healing  │
                    │ Gradient-domain boundary│
                    │ illumination matching   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Committed Canvas Layer  │
                    └─────────────────────────┘
```

### 1. Multi-Scale Coarse-to-Fine Pyramid
Processes the hole at 1/4 and 1/2 scales before full resolution. Large global structures and horizons are established at coarse scales, then upsampled to guide high-frequency fine texture synthesis without tile repetition.

### 2. Wexler EM Global Coherence & Multi-Patch Voting
Instead of simply copying raw patches, every synthesized pixel receives weighted votes from all $(2r+1)^2$ overlapping patches containing it:
$$I(p) = \frac{\sum_{q, p \in \Psi_q} w(q) \cdot I\big(\text{source}(q) + (p - q)\big)}{\sum_{q, p \in \Psi_q} w(q)}$$
This eliminates blockiness, seams, and patch boundary artifacts.

### 3. He & Sun Dominant Spatial Offset Prior
Extracts dominant offset vectors $(\Delta x, \Delta y)$ from the nearest-neighbor field histogram and uses them as candidates during PatchMatch propagation, drastically speeding up convergence on natural textures.

### 4. Gradient & Edge-Aware Composite Distance Metric
$$D(T, S) = \sum \Big( D_{\text{color}}(T, S) + \beta \cdot D_{\text{gradient}}(T, S) \Big)$$
Preserves sharp architectural edges, horizon lines, and geometric contours across the inpainting boundary.

### 5. Generalized Transformations
Supports horizontal and vertical mirroring as well as rotation adaptation for symmetric textures and patterns.

### 6. Poisson / Gradient-Domain Seam Healing
Boundary relaxation smoothes out lighting, exposure, and color temperature discrepancies between source and target areas.

---

## ⚡ Available Inpainting Engines

| Engine | Method | Speed | Best For |
| :--- | :--- | :--- | :--- |
| **⚡ Photoshop-Grade Multi-Scale EM** *(Default)* | Multi-Scale Pyramid + Wexler Voting + PatchMatch | **~0.8s – 2.5s** | Complex photos, people, objects, architecture, backgrounds |
| **💨 Telea Fast Marching** | Fast Marching Distance Diffusion | **< 0.05s** *(Instant)* | Wires, scratches, dust, text, watermarks, skin blemishes |
| **🔬 Classic Criminisi** | Exhaustive Isophote Priority | **~2.0s – 5.0s** | Traditional geometric line propagation |

---

## 📦 Installation

### Windows
1. Download or clone this repository:
   ```bash
   git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git
   ```
2. Double-click **`install.bat`** (or copy `content-aware-fill` folder into `%APPDATA%\GIMP\3.2\plug-ins\` or `%APPDATA%\GIMP\3.0\plug-ins\`).
3. Restart **GIMP 3**.

### Linux / macOS
1. Clone the repository:
   ```bash
   git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git
   cd GIMP-Content-Aware-Fill
   ```
2. Run the installer script:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
3. Restart **GIMP 3**.

---

## 🚀 How to Use

1. Open your image in **GIMP 3**.
2. Select the object, person, watermark, or defect to remove using the **Free Select / Lasso (`F`)** or **Rectangle Select (`R`)** tool.
3. Open **`Edit` → `Content-Aware Fill...`** (or **`Filters` → `Enhance` → `Content-Aware Fill...`**).
4. Select your preferred engine, patch size, and rotation options, then click **Fill Selection**.

---

## 📄 License

GPLv3. See [LICENSE](LICENSE) for details.
