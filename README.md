# Photoshop-Style Content-Aware Fill for GIMP 3

A state-of-the-art classical computer vision inpainting suite for GIMP 3, engineered to produce perceptually convincing, Photoshop-grade object removal and hole filling without neural networks or machine learning models.

---

## 🌟 Key Features

- **⚡ Full Modular Computer Vision Pipeline (Default Engine)**:
  - **Stage 1: Mask Analysis**: Distance transform, boundary geometry, and normal vectors.
  - **Stage 2: Structure Detection**: Scharr gradient operators, Structure Tensor ($J_0$), and local anisotropy / coherence field.
  - **Stage 3: Structure Propagation**: Criminisi-style priority queue ($P = C \cdot D$) and boundary isophote trajectory tracing.
  - **Stage 4: Multi-Scale Gaussian Pyramid**: Coarse-to-fine NNF projection ($1/4 \to 1/2 \to 1/1$).
  - **Stage 5: High-Performance Multi-Threaded PatchMatch**: Randomized search, spatial propagation, and exponential decay random window sampling across all CPU cores (`ThreadPoolExecutor`).
  - **Stage 6: Composite Perceptual Distance Metric**: YCbCr color separation (luminance vs chroma) + gradient magnitude & orientation + structure coherence alignment.
  - **Stage 7: Geometric Patch Adaptation**: Discrete candidate rotations ($\pm 30^\circ$), scales ($0.8\times - 1.25\times$), and horizontal/vertical mirroring.
  - **Stage 8: Contamination-Free Source Selection**: Dilation margins and directional sampling presets (`Auto`, `Right`, `Left`, `Above`, `Below`, `All Around`).
  - **Stage 9: Global MRF Patch Consistency**: Minimizes pairwise coordinate jumps between adjacent patches to eliminate salt-and-pepper artifacts.
  - **Stage 10: Minimum-Error Seam Optimization**: Multi-patch consensus blending along low-contrast boundaries.
  - **Stage 11: Local Color & Exposure Adaptation**: Matches local target boundary lighting and contrast.
  - **Stage 12: Gradient-Domain Poisson Residual Healing ($\nabla^2 \Delta = 0$)**: Diffuses boundary residual differences smoothly across the selection to eliminate all cut-off lines and collision steps.
  - **Stage 13: Confidence-Driven Iterative Refinement**: Tracks pixel confidence and re-evaluates low-confidence details.
- **Support for Both Opaque Selections and Transparent Holes (`alpha = 0`)**.
- **Fast Standalone Fallback Engines**:
  - **🎯 Structural Shift-Map**: Instantaneous ($<0.04\text{s}$) direct offset alignment.
  - **💨 Telea Fast Marching**: Instant diffusion for small scratches, wires, and text.
- **Zero External Dependencies**: Pure Python 3 + standard library (`math`, `array`, `concurrent.futures`, `heapq`), ensuring 100% plug-and-play compatibility on Windows, macOS, and Linux.

---

## 🚀 Installation

### Automated (Windows)
Run `install.bat`.

### Manual
Copy the `content-aware-fill` folder to your GIMP plug-ins directory:
- **Windows**: `%APPDATA%\GIMP\3.2\plug-ins\content-aware-fill\` (or `3.0`)
- **Linux**: `~/.config/GIMP/3.2/plug-ins/content-aware-fill/`
- **macOS**: `~/Library/Application Support/GIMP/3.2/plug-ins/content-aware-fill/`

---

## 📖 How to Use

1. Open an image in **GIMP 3**.
2. Select the object or region you want to remove using any selection tool (**Free Select / Lasso**, **Rectangle Select**, **Fuzzy Select**, etc.) or erase it to transparency (`alpha = 0`).
3. Open the menu: **`Edit` → `Content-Aware Fill...`** (or **`Filters` → `Enhance` → `Content-Aware Fill...`**).
4. Choose your desired **Sampling Area**, **Quality Preset**, and **Geometric Adaptation**, then click **Fill Selection**!

---

## 🔬 Algorithmic References

- **Barnes et al.**, *"PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing"*, ACM Transactions on Graphics (SIGGRAPH 2009).
- **Criminisi, Pérez and Toyama**, *"Region Filling and Object Removal by Exemplar-Based Image Inpainting"*, IEEE Transactions on Image Processing (TIP 2004).
- **Wexler, Shechtman and Irani**, *"Space-Time Video Completion"*, IEEE TPAMI (2007).
- **Pérez, Gangnet and Blake**, *"Poisson Image Editing"*, ACM Transactions on Graphics (SIGGRAPH 2003).
- **He and Sun**, *"Statistics of Patch Offsets for Image Completion"*, ECCV (2012).

---

## 📄 License
GPLv3+
