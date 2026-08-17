# GIMP Content-Aware Fill

[![GIMP 3](https://img.shields.io/badge/GIMP-3.0%20%7C%203.2-orange.svg)](https://www.gimp.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)

A high-performance **Photoshop-style Content-Aware Fill** plugin for **GIMP 3** featuring 4 selectable inpainting algorithms ranging from **instantaneous diffusion (<50ms)** to **PatchMatch** and **exemplar structure synthesis**.

---

## ✨ Features

- ⚡ **Multi-Algorithm Inpainting Suite**:
  1. **PatchMatch (Photoshop Engine)** *(Default)*: Randomized 2D patch propagation for natural textures (~0.3s – 0.7s).
  2. **Telea Fast Marching**: Instantaneous diffusion (<0.05s) for scratches, dust, text, watermarks, and skin blemishes.
  3. **Multi-Scale Pyramidal**: 2-level coarse-to-fine structure synthesis (~0.4s – 1.0s) for structured geometry and patterns.
  4. **Classic Criminisi**: Traditional isophote priority propagation for geometric edge continuation.
- 🎛️ **Intuitive UI**: Interactive Gtk3 dialog to adjust patch size, quality passes, and auto-deselection.
- 🚀 **Zero External Dependencies**: Pure Python implementation with `array.array` and `bytearray`—works out of the box with GIMP 3's bundled Python runtime.
- ↩️ **Full Undo Support**: Seamless single-step `Ctrl+Z` undo grouping.

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
4. Select your preferred algorithm and patch size, then click **Fill Selection**.

---

## ⚡ Algorithm Comparison

| Algorithm | Method | Speed | Best For |
| :--- | :--- | :--- | :--- |
| **⚡ PatchMatch** | Randomized 2D Patch Propagation | **~0.3s – 0.7s** | Complex textures, photos, large objects, foliage, backgrounds *(Photoshop's actual engine)* |
| **💨 Telea Fast Marching** | Fast Marching Distance Diffusion | **< 0.05s** *(Instant)* | Wires, scratches, dust, text, skin blemishes, smooth skies |
| **🎯 Multi-Scale Pyramidal** | Coarse-to-Fine 2-Level Synthesis | **~0.4s – 1.0s** | Structured geometry, walls, architectural lines |
| **🔬 Classic Criminisi** | Exhaustive Isophote Priority | **~2.0s – 5.0s** | High-precision geometric edge continuation |

---

## 📄 License

GPLv3. See [LICENSE](LICENSE) for details.
