#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-Aware Fill for GIMP 3 - Clean Pipeline
==============================================
Non-AI, pure algorithm, no dependencies.
Pipeline: Mask Analysis -> Pyramid -> PatchMatch (Cauchy + frisket) -> Vote -> Poisson
"""

import sys
import os
import math
import time
import array
import random
import heapq
import traceback

plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

from caf_engine.engine import inpaint as engine_inpaint

import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk", "3.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gimp, GimpUi, Gegl, Gtk, GLib, GObject
Gegl.init(None)

def _(m): return m

class ContentAwareFillDialog(Gtk.Dialog):
    def __init__(self, image, drawable):
        super().__init__(title=_("Content-Aware Fill"), flags=Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT)
        self.set_default_size(540, 560)
        self.set_resizable(False)
        self.image = image
        self.drawable = drawable
        self.add_button(_("_Cancel"), Gtk.ResponseType.CANCEL)
        btn = self.add_button(_("_Fill"), Gtk.ResponseType.OK)
        btn.get_style_context().add_class("suggested-action")
        content = self.get_content_area()
        content.set_spacing(10)
        content.set_margin_start(16)
        content.set_margin_end(16)
        content.set_margin_top(14)
        content.set_margin_bottom(14)
        hb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        tl = Gtk.Label()
        tl.set_markup("<span size='large' weight='bold'>Content-Aware Fill</span>")
        tl.set_xalign(0.0)
        dl = Gtk.Label()
        dl.set_markup("<span size='small' color='#777'>Clean PatchMatch + Poisson — No AI, No Dependencies</span>")
        dl.set_xalign(0.0)
        hb.pack_start(tl, False, False, 0)
        hb.pack_start(dl, False, False, 0)
        content.pack_start(hb, False, False, 0)
        # Basic Settings
        frame = Gtk.Frame(label=_("Basic Settings"))
        grid = Gtk.Grid()
        grid.set_column_spacing(14)
        grid.set_row_spacing(10)
        grid.set_margin_start(12)
        grid.set_margin_end(12)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        frame.add(grid)
        content.pack_start(frame, False, False, 0)
        pl = Gtk.Label(label=_("Patch Size:"))
        pl.set_xalign(0.0)
        grid.attach(pl, 0, 0, 1, 1)
        self.adj = Gtk.Adjustment(value=9, lower=5, upper=21, step_increment=2, page_increment=4)
        self.scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.adj)
        self.scale.set_digits(0)
        self.scale.set_hexpand(True)
        self.scale.add_mark(5, Gtk.PositionType.BOTTOM, "5")
        self.scale.add_mark(9, Gtk.PositionType.BOTTOM, "9*")
        self.scale.add_mark(15, Gtk.PositionType.BOTTOM, "15")
        grid.attach(self.scale, 1, 0, 1, 1)
        ql = Gtk.Label(label=_("Quality:"))
        ql.set_xalign(0.0)
        grid.attach(ql, 0, 1, 1, 1)
        self.qual = Gtk.ComboBoxText()
        self.qual.append_text(_("Balanced (Recommended)"))
        self.qual.append_text(_("High (Slower, Sharper)"))
        self.qual.append_text(_("Fast"))
        self.qual.set_active(0)
        grid.attach(self.qual, 1, 1, 1, 1)
        sl = Gtk.Label(label=_("Sampler Area:"))
        sl.set_xalign(0.0)
        grid.attach(sl, 0, 2, 1, 1)
        self.src = Gtk.ComboBoxText()
        self.src.append_text(_("Auto (Smart)"))
        self.src.append_text(_("Right →"))
        self.src.append_text(_("Left ←"))
        self.src.append_text(_("Above ↓"))
        self.src.append_text(_("Below ↑"))
        self.src.append_text(_("All Around"))
        self.src.set_active(0)
        grid.attach(self.src, 1, 2, 1, 1)
        self.desel = Gtk.CheckButton(label=_("Deselect after fill"))
        grid.attach(self.desel, 0, 3, 2, 1)
        # Advanced expander
        self.expander = Gtk.Expander(label=_("Advanced Options — Blending & Sampling"))
        self.expander.set_expanded(False)
        content.pack_start(self.expander, False, False, 0)
        adv_grid = Gtk.Grid()
        adv_grid.set_column_spacing(14)
        adv_grid.set_row_spacing(10)
        adv_grid.set_margin_start(12)
        adv_grid.set_margin_end(12)
        adv_grid.set_margin_top(8)
        adv_grid.set_margin_bottom(8)
        self.expander.add(adv_grid)
        # Blending mode
        bl = Gtk.Label(label=_("Blending:"))
        bl.set_xalign(0.0)
        adv_grid.attach(bl, 0, 0, 1, 1)
        self.blend = Gtk.ComboBoxText()
        self.blend.append_text(_("Poisson Seamless (Recommended)"))
        self.blend.append_text(_("Feather Only"))
        self.blend.append_text(_("None (Hard)"))
        self.blend.set_active(0)
        self.blend.connect("changed", self._on_blend_changed)
        adv_grid.attach(self.blend, 1, 0, 1, 1)
        # Poisson band
        self.band_label = Gtk.Label(label=_("Poisson Band:"))
        self.band_label.set_xalign(0.0)
        adv_grid.attach(self.band_label, 0, 1, 1, 1)
        self.band_adj = Gtk.Adjustment(value=16, lower=4, upper=32, step_increment=4, page_increment=8)
        self.band_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.band_adj)
        self.band_scale.set_digits(0)
        self.band_scale.set_hexpand(True)
        self.band_scale.add_mark(8, Gtk.PositionType.BOTTOM, "8")
        self.band_scale.add_mark(16, Gtk.PositionType.BOTTOM, "16*")
        self.band_scale.add_mark(24, Gtk.PositionType.BOTTOM, "24")
        adv_grid.attach(self.band_scale, 1, 1, 1, 1)
        # Poisson iters
        self.iter_label = Gtk.Label(label=_("Poisson Iters:"))
        self.iter_label.set_xalign(0.0)
        adv_grid.attach(self.iter_label, 0, 2, 1, 1)
        self.iter_adj = Gtk.Adjustment(value=40, lower=10, upper=120, step_increment=10, page_increment=20)
        self.iter_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.iter_adj)
        self.iter_scale.set_digits(0)
        self.iter_scale.set_hexpand(True)
        self.iter_scale.add_mark(20, Gtk.PositionType.BOTTOM, "20")
        self.iter_scale.add_mark(40, Gtk.PositionType.BOTTOM, "40*")
        self.iter_scale.add_mark(80, Gtk.PositionType.BOTTOM, "80")
        adv_grid.attach(self.iter_scale, 1, 2, 1, 1)
        # Feather width
        self.feather_label = Gtk.Label(label=_("Feather Width:"))
        self.feather_label.set_xalign(0.0)
        adv_grid.attach(self.feather_label, 0, 3, 1, 1)
        self.feather_adj = Gtk.Adjustment(value=12, lower=2, upper=24, step_increment=2, page_increment=4)
        self.feather_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.feather_adj)
        self.feather_scale.set_digits(0)
        self.feather_scale.set_hexpand(True)
        self.feather_scale.add_mark(4, Gtk.PositionType.BOTTOM, "4")
        self.feather_scale.add_mark(12, Gtk.PositionType.BOTTOM, "12*")
        self.feather_scale.add_mark(20, Gtk.PositionType.BOTTOM, "20")
        adv_grid.attach(self.feather_scale, 1, 3, 1, 1)
        # Sampler expansion
        el = Gtk.Label(label=_("Sampler Expand:"))
        el.set_xalign(0.0)
        adv_grid.attach(el, 0, 4, 1, 1)
        self.expand_adj = Gtk.Adjustment(value=1.5, lower=1.0, upper=3.0, step_increment=0.5, page_increment=1.0)
        self.expand_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.expand_adj)
        self.expand_scale.set_digits(1)
        self.expand_scale.set_hexpand(True)
        self.expand_scale.add_mark(1.0, Gtk.PositionType.BOTTOM, "1.0")
        self.expand_scale.add_mark(1.5, Gtk.PositionType.BOTTOM, "1.5*")
        self.expand_scale.add_mark(3.0, Gtk.PositionType.BOTTOM, "3.0")
        adv_grid.attach(self.expand_scale, 1, 4, 1, 1)
        self._on_blend_changed(self.blend)
        self.show_all()
    def _on_blend_changed(self, combo):
        mode = combo.get_active()
        is_poisson = (mode == 0)
        is_feather = (mode == 1)
        self.band_label.set_sensitive(is_poisson)
        self.band_scale.set_sensitive(is_poisson)
        self.iter_label.set_sensitive(is_poisson)
        self.iter_scale.set_sensitive(is_poisson)
        self.feather_label.set_sensitive(is_feather or is_poisson)
        self.feather_scale.set_sensitive(is_feather or is_poisson)
    def get_settings(self):
        v = int(self.adj.get_value())
        if v%2==0: v+=1
        srcs = ["auto","right","left","above","below","all"]
        src = srcs[self.src.get_active()] if self.src.get_active() < len(srcs) else "auto"
        qual = ["balanced","high","fast"][self.qual.get_active()]
        blend = ["poisson","feather","none"][self.blend.get_active()]
        return {"radius": max(2, v//2), "quality": qual, "source": src, "deselect": self.desel.get_active(), "blend": blend, "poisson_band": int(self.band_adj.get_value()), "poisson_iters": int(self.iter_adj.get_value()), "feather_width": int(self.feather_adj.get_value()), "sampler_expand": float(self.expand_adj.get_value())}

class ContentAwareFillPlugin(Gimp.PlugIn):
    def do_set_i18n(self, procname):
        return True, "gimp30-python", None
    def do_query_procedures(self):
        return ["plug-in-content-aware-fill"]
    def do_create_procedure(self, name):
        if name == "plug-in-content-aware-fill":
            proc = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
            proc.set_image_types("RGB*, GRAY*")
            proc.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            proc.set_documentation(_("Content-Aware Fill"), _("Seamless hole filling via PatchMatch + Poisson"), name)
            proc.set_menu_label(_("Content-Aware Fill..."))
            proc.add_menu_path("<Image>/Edit/")
            proc.add_menu_path("<Image>/Filters/Enhance/")
            proc.set_attribution("bunnywaffle", "GPLv3+", "2026")
            return proc
        return None
    def run(self, procedure, run_mode, image, drawables, config, data):
        try:
            if not drawables or drawables[0] is None:
                Gimp.message(_("Select a layer"))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)
            drawable = drawables[0]
            if Gimp.Selection.is_empty(image):
                Gimp.message(_("Make a selection first"))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)
            bounds = Gimp.Selection.bounds(image)
            if len(bounds)==6:
                success, non_empty, x1, y1, x2, y2 = bounds
            else:
                success, non_empty, x1, y1, x2, y2 = bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
            if not non_empty or x2<=x1 or y2<=y1:
                Gimp.message(_("Empty selection"))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)
            settings = {"radius":4, "quality":"balanced", "source":"auto", "deselect":False}
            if run_mode == Gimp.RunMode.INTERACTIVE:
                GimpUi.init("content-aware-fill")
                dlg = ContentAwareFillDialog(image, drawable)
                resp = dlg.run()
                if resp != Gtk.ResponseType.OK:
                    dlg.destroy()
                    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, None)
                settings = dlg.get_settings()
                dlg.destroy()
            off = drawable.get_offsets()
            off_x, off_y = (off[1], off[2]) if len(off)==3 else (off[0], off[1])
            dw, dh = drawable.get_width(), drawable.get_height()
            lx1 = max(0, min(dw, x1 - off_x))
            ly1 = max(0, min(dh, y1 - off_y))
            lx2 = max(0, min(dw, x2 - off_x))
            ly2 = max(0, min(dh, y2 - off_y))
            if lx2<=lx1 or ly2<=ly1:
                Gimp.message(_("Selection outside layer"))
                return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, None)
            sel_w, sel_h = lx2-lx1, ly2-ly1
            margin = max(250, max(sel_w, sel_h))
            rx1 = max(0, lx1 - margin)
            ry1 = max(0, ly1 - margin)
            rx2 = min(dw, lx2 + margin)
            ry2 = min(dh, ly2 + margin)
            rw, rh = rx2-rx1, ry2-ry1
            if rw<=0 or rh<=0:
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)
            img_rect = Gegl.Rectangle.new(rx1+off_x, ry1+off_y, rw, rh)
            lay_rect = Gegl.Rectangle.new(rx1, ry1, rw, rh)
            has_alpha = drawable.has_alpha()
            fmt = "R'G'B'A u8" if has_alpha else "R'G'B' u8"
            ch = 4 if has_alpha else 3
            dbuf = drawable.get_buffer()
            sbuf = image.get_selection().get_buffer()
            img_raw = dbuf.get(lay_rect, 1.0, fmt, Gegl.AbyssPolicy.CLAMP)
            mask_raw = sbuf.get(img_rect, 1.0, "Y u8", Gegl.AbyssPolicy.CLAMP)
            img_bytes = bytearray(img_raw)
            mask_bytes = bytearray(mask_raw)
            Gimp.progress_init(_("Filling..."))
            def progress_cb(f,m):
                Gimp.progress_update(max(0.0, min(1.0, f)))
                return True
            image.undo_group_start()
            t0 = time.time()
            inpainted = engine_inpaint(img_bytes, mask_bytes, rw, rh, ch, patch_radius=settings["radius"], quality=settings["quality"], sample_source=settings["source"], progress_callback=progress_cb, blend_mode=settings["blend"], poisson_band=settings["poisson_band"], poisson_iters=settings["poisson_iters"], feather_width=settings["feather_width"], sampler_expand=settings["sampler_expand"])
            elapsed = time.time()-t0
            # Composite using selection anti-aliasing to guarantee zero edge steps
            final_bytes = bytearray(img_bytes)
            for idx in range(rw * rh):
                m_val = mask_bytes[idx]
                if m_val > 0:
                    alpha_m = m_val / 255.0
                    p = idx * ch
                    for c in range(min(3, ch)):
                        final_bytes[p + c] = max(0, min(255, int(round(inpainted[p + c] * alpha_m + img_bytes[p + c] * (1.0 - alpha_m)))))
                    if ch == 4:
                        final_bytes[p + 3] = 255
            dbuf.set(lay_rect, fmt, bytes(final_bytes))
            dbuf.flush()
            drawable.update(rx1, ry1, rw, rh)
            if settings["deselect"]:
                Gimp.Selection.none(image)
            image.undo_group_end()
            Gimp.displays_flush()
            Gimp.progress_end()
            print(f"[CAF] {rw}x{rh} in {elapsed:.2f}s")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, None)
        except Exception as e:
            try: image.undo_group_end()
            except: pass
            Gimp.progress_end()
            traceback.print_exc()
            try:
                import pathlib
                p = pathlib.Path(r"C:\Users\Hoptimizer\AppData\Local\Temp\caf_error.log")
                p.write_text(traceback.format_exc(), encoding="utf-8")
            except: pass
            Gimp.message(f"Content-Aware Fill Error: {e}\n{traceback.format_exc()[:800]}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(message=str(e)))

if __name__ == "__main__":
    Gimp.main(ContentAwareFillPlugin.__gtype__, sys.argv)
