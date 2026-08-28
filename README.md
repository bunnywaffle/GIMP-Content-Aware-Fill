# Content-Aware Fill for GIMP 3

Photoshop-grade object removal for GIMP 3 — pure algorithms, no AI, no dependencies.

![GIMP 3.2+](https://img.shields.io/badge/GIMP-3.2%2B-blue) ![Python 3](https://img.shields.io/badge/Python-3-green) ![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow)

## Installation — Clone into GIMP's plug-ins folder

**Windows** (`%APPDATA%\GIMP\3.2\plug-ins`):
```powershell
cd $env:APPDATA\GIMP\3.2\plug-ins
git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git content-aware-fill
```

**Linux** (`~/.config/GIMP/3.2/plug-ins`):
```bash
cd ~/.config/GIMP/3.2/plug-ins
git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git content-aware-fill
```

**macOS** (`~/Library/Application Support/GIMP/3.2/plug-ins`):
```bash
cd ~/Library/Application\ Support/GIMP/3.2/plug-ins
git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git content-aware-fill
```

Restart GIMP. Find it at **Edit > Content-Aware Fill...** (also **Filters > Enhance**).

> The repo *is* the plugin. Cloning it as `content-aware-fill` inside `plug-ins` is all that is needed — no copy step, no build.

### Manual (ZIP download)
Download ZIP from GitHub → **Code > Download ZIP** → extract the `content-aware-fill.py` and `caf_engine/` folder into:
- Windows: `%APPDATA%\GIMP\3.2\plug-ins\content-aware-fill\`
- Linux: `~/.config/GIMP/3.2/plug-ins/content-aware-fill/`
- macOS: `~/Library/Application Support/GIMP/3.2/plug-ins/content-aware-fill/`

Or run `install.bat` (Windows) / `install.sh` (Linux/macOS) from the repo root.

### Updating
```bash
cd <plug-ins>/content-aware-fill
git pull
```
Restart GIMP.

---

## Use

1. Open image in GIMP 3.
2. Select the area to remove (**Free Select / Lasso**, **Rectangle**, **Fuzzy Select**, or paint an alpha hole).
3. **Edit > Content-Aware Fill...**
4. Choose **Patch Size** (5–21, 9 is default), **Quality** (`Balanced`/`High`/`Fast`), **Sampler Area** (`Auto` detects best side).
5. Open **Advanced Options** for blending control:
   - **Blending:** `Poisson Seamless` (harmonic diffusion, invisible seam), `Feather Only`, or `None`
   - **Poisson Band / Iters / Feather Width**
   - **Sampler Expand** `1.0x–3.0x` (frisket size around selection)
6. Click **Fill**.

---

## How it works

Classical computer vision pipeline, no neural network:

1. **Mask Analysis** — distance transform + boundary normals
2. **Frisket corpus** — source limited to `selection dilate 1.5×` (Heal Selection style)
3. **PatchMatch** — Cauchy robust distance `1-1/(1+ssd/2000)`, SAT O(1) source validity, fixed support sets, jittered dominant shifts, shotgun 9-NN for interior holes
4. **Vote** — overlap-weighted averaging
5. **Poisson band** — seamless harmonic blending (`16px` band, `80` SOR sweeps, `ω=1.4`) + cosine feather

Pure Python + stdlib (`math`, `array`, `heapq`, `random`) — works on Windows/macOS/Linux without extra packages.

---

## References

- Barnes et al., *PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing*, SIGGRAPH 2009
- Criminisi et al., *Region Filling and Object Removal by Exemplar-Based Image Inpainting*, TIP 2004
- Wexler et al., *Space-Time Video Completion*, TPAMI 2007
- Pérez et al., *Poisson Image Editing*, SIGGRAPH 2003
- He & Sun, *Statistics of Patch Offsets for Image Completion*, ECCV 2012

## License

GPLv3+
