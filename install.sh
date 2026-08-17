#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  GIMP 3 Content-Aware Fill Plugin Installer (Linux/macOS)"
echo "========================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$SCRIPT_DIR/content-aware-fill/content-aware-fill.py"

if [ ! -f "$PLUGIN_SRC" ]; then
    echo "[ERROR] Could not find content-aware-fill.py"
    exit 1
fi

chmod +x "$PLUGIN_SRC"

# Target directories
LINUX_DIR="$HOME/.config/GIMP/3.0/plug-ins/content-aware-fill"
MACOS_DIR="$HOME/Library/Application Support/GIMP/3.0/plug-ins/content-aware-fill"

if [[ "$OSTYPE" == "darwin"* ]]; then
    TARGET_DIR="$MACOS_DIR"
else
    TARGET_DIR="$LINUX_DIR"
fi

mkdir -p "$TARGET_DIR"
cp -f "$PLUGIN_SRC" "$TARGET_DIR/content-aware-fill.py"
chmod +x "$TARGET_DIR/content-aware-fill.py"

echo "[OK] Installed to $TARGET_DIR/content-aware-fill.py"
echo ""
echo "Installation Complete! Restart GIMP 3 and go to: Edit > Content-Aware Fill..."
