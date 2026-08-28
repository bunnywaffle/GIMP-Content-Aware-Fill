#!/usr/bin/env bash
set -e
echo "GIMP 3 Content-Aware Fill — Installer"
echo "Repo is the plugin — clone it into plug-ins:"
echo "  cd ~/.config/GIMP/3.2/plug-ins && git clone https://github.com/bunnywaffle/GIMP-Content-Aware-Fill.git content-aware-fill"
echo ""
echo "Manual fallback: copying current folder..."
SRC="$(cd "$(dirname "$0")" && pwd)"
for VER in "3.2" "3.0"; do
  DST="$HOME/.config/GIMP/$VER/plug-ins/content-aware-fill"
  mkdir -p "$DST"
  cp -f "$SRC/content-aware-fill.py" "$DST/"
  cp -rf "$SRC/caf_engine" "$DST/"
  echo "[OK] $DST"
done
echo "Restart GIMP 3 — Edit > Content-Aware Fill..."
