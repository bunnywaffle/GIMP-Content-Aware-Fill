# GIMP Content-Aware Fill

[![GIMP 3](https://img.shields.io/badge/GIMP-3.0%20%7C%203.2-orange.svg)](https://www.gimp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

A state-of-the-art **Photoshop-grade Content-Aware Fill** plugin for **GIMP 3** implementing the complete modern non-AI inpainting pipeline: **He & Sun (2012) Dominant Offset Statistics + Barnes et al. (2009) Multi-Scale PatchMatch + Direct Exemplar Transfer + User Sampling Area Controls + Seamless Boundary Seam Healing**.

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
                    │  He & Sun (2012) Offset │
                    │   Histogram Statistics  │
                    │  Top K Dominant Shifts  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Multi-Scale Pyramid   │
                    │   Coarse (1/2) Level    │
                    │         │               │
                    │    Fine (1/1) Level     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    PatchMatch 2D NNF    │
                    │ - Spatial Propagation   │
                    │ - Dominant Offset Prior │
                    │ - Multi-Scale Search    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Direct Exemplar Transfer│
                    │ Zero-Blur Sharp Textures│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Boundary Seam Healing   │
                    │ Smooth Illumination &   │
                    │ Color Temperature Match │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Committed Canvas Layer  │
                    └─────────────────────────┘
```

### 1. Dominant Spatial Offset Statistics (He & Sun, ECCV 2012)
Extracts dominant offset vectors $(\Delta x, \Delta y)$ from the boundary context histogram and injects them as high-priority candidates during PatchMatch propagation. This eliminates local minima traps and guarantees repeating rows, columns, textures, and shelf lines align with geometric precision.

### 2. Multi-Scale PatchMatch Propagation (Barnes et al., SIGGRAPH 2009)
Sweeps in forward and reverse alternating raster passes across the selection hole:
- **Spatial Neighbor Propagation**: Transmits coherent displacement vectors from adjacent pixels.
- **Dominant Offset Testing**: Tests image-wide structural shift vectors.
- **Multi-Scale Random Refinement**: Explores local offsets with exponentially decaying radii.

### 3. Direct Exemplar Transfer (Zero-Blur Guarantee)
Unlike naive multi-patch averaging filters that turn high frequencies into blur smudges, the engine transfers intact, razor-sharp exemplar textures directly from the source image.

### 4. Seamless Boundary Seam Healing
Applies boundary relaxation strictly to the 1–2px seam bordering known pixels to eliminate illumination and exposure discrepancies while preserving 100% sharp texture in the interior.

---

## ⚡ Available Inpainting Engines

| Engine | Method | Speed | Best For |
| :--- | :--- | :--- | :--- |
| **⚡ Structural PatchMatch** *(Default)* | He & Sun 2012 + Barnes 2009 + Direct Exemplar | **~0.2s – 0.8s** | Complex photos, patterns, architectural lines, shelves, textured objects |
| **🎯 Structural Shift-Map** | Direct Single-Vector Optimal Offset Alignment | **< 0.05s** *(Instant)* | Uniform repeating textures, rows, and directional extensions |
| **💨 Telea Fast Marching** | Fast Marching Distance PDE Diffusion | **< 0.02s** *(Instant)* | Wires, scratches, dust, text, watermarks, skin blemishes |
| **🔬 Classic Criminisi** | Exhaustive Isophote Priority Synthesis | **~1.5s – 3.5s** | Traditional geometric line and curve completion |

---

## 🎛️ User Sampling Area Controls

Like Photoshop's Content-Aware Fill workspace, you can define exactly where the algorithm draws source pixels:
- **`Auto (Smart Context Continuation)`** *(Default)*: Automatically searches all surrounding regions for the best matching textures.
- **`Sample from Right →`**: Restricts / prioritizes source sampling to the right (e.g. extending horizontal rows of books or patterns leftward).
- **`Sample from Left ←`**: Restricts / prioritizes source sampling to the left (e.g. extending horizontal patterns rightward).
- **`Sample from Above ↓`**: Restricts / prioritizes source sampling to above (e.g. extending vertical columns, pillars, trees downward).
- **`Sample from Below ↑`**: Restricts / prioritizes source sampling to below (e.g. extending vertical textures upward).
- **`All Around`**: Full surrounding canvas margin search.

---

## 📦 Installation

### Windows
1. Clone or download this repository:
   ```powershell
   git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git
   ```
2. Run **`install.bat`** (or copy the `content-aware-fill` folder into `%APPDATA%\GIMP\3.2\plug-ins\` or `%APPDATA%\GIMP\3.0\plug-ins\`).
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
4. Select your preferred engine, sampling area, and patch size, then click **Fill Selection**.

---

## 📄 References & Papers

1. K. He, J. Sun, *"Statistics of Patch Offsets for Image Completion"*, ECCV 2012 / Microsoft Research. [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/05/stat_completion.pdf)
2. C. Barnes, E. Shechtman, A. Finkelstein, D. B. Goldman, *"PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing"*, ACM SIGGRAPH 2009. [PDF](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)
3. A. Telea, *"An Image Inpainting Technique Based on the Fast Marching Method"*, Journal of Graphics Tools, 2004.
4. A. Criminisi, P. Pérez, K. Toyama, *"Region Filling and Object Removal by Exemplar-Based Image Inpainting"*, IEEE Transactions on Image Processing, 2004.

---

## 📄 License

GPLv3+. See [LICENSE](LICENSE) for details.
