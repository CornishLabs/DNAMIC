#!/usr/bin/env python3
# ARTIQ applet (PyQt6 + pyqtgraph) with:
# - ImageView + histogram panel
# - magma colormap (few control points => sane histogram handles)
# - ROI overlays (optional), counts labels (optional)
# - single top-row toolbar: [Autoscale] [Auto once]  <x,y,val>
# - autoscale uses min/max; histogram bounds stay in sync

import numpy as np
import PyQt6  # ensure pyqtgraph binds to Qt6
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')  # width = n_x, height = n_y

from artiq.applets.simple import SimpleApplet


def _simple_colormap(name="magma", stops=6) -> pg.ColorMap:
    """Build a ColorMap with only `stops` control points from a named pg colormap."""
    base = pg.colormap.get(name)  # modern API
    lut = base.getLookupTable(0.0, 1.0, stops)    # (stops x 3 or x4) uint8
    colors = lut[:, :3]                           # Nx3 RGB
    pos = np.linspace(0.0, 1.0, stops)
    return pg.ColorMap(pos, colors)


class ImageWithROIs(pg.ImageView):
    def __init__(self, args, req):
        super().__init__()
        self.args = args

        # Show histogram; hide extra buttons for a clean look.
        if getattr(self.ui, "menuBtn", None):
            self.ui.menuBtn.hide()
        if getattr(self.ui, "roiBtn", None):
            self.ui.roiBtn.hide()

        # NumPy-like coords: (x right, y down); origin top-left
        self.getView().setAspectLocked(True)
        self.getView().invertY(True)

        # Colormap (few control points => few histogram handles)
        self._cmap = _simple_colormap("magma", stops=6)
        self.setColorMap(self._cmap)   # updates both image & histogram

        self._roi_curves, self._roi_labels = [], []
        self._img_np = None
        self._autoscale = True  # autoscale ON by default

        # ---- Single top-row toolbar overlay: [Autoscale] [Auto once]  position ----
        vp = self.ui.graphicsView.viewport()
        self._toolbar = QtWidgets.QWidget(vp)
        layout = QtWidgets.QHBoxLayout(self._toolbar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._chk_autoscale = QtWidgets.QCheckBox("Autoscale", self._toolbar)
        self._chk_autoscale.setChecked(self._autoscale)

        self._btn_auto_once = QtWidgets.QPushButton("Auto once", self._toolbar)

        # Position label (monospace), fixed-ish width so the bar doesn't jump
        self._pos_label = QtWidgets.QLabel("", self._toolbar)
        self._pos_label.setStyleSheet("QLabel { font-family: monospace; }")
        self._pos_label.setMinimumWidth(200)

        layout.addWidget(self._chk_autoscale)
        layout.addWidget(self._btn_auto_once)
        layout.addWidget(self._pos_label)

        self._toolbar.setStyleSheet(
            "QWidget { background: rgba(0,0,0,120); color: white; border-radius: 4px; }"
            "QPushButton, QCheckBox, QLabel { color: white; }"
        )
        self._toolbar.move(6, 6)
        self._toolbar.adjustSize()

        # Hooks
        self._chk_autoscale.toggled.connect(self._on_autoscale_toggled)
        self._btn_auto_once.clicked.connect(self._apply_auto_levels_once)

        # Mouse move hook (throttled)
        self._mouse_proxy = pg.SignalProxy(
            self.getView().scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        # Keep toolbar pinned at top-left on viewport resize
        vp.installEventFilter(self)

        self.resize(900, 600)
        self.setWindowTitle("Tweezers Image")

    # Keep toolbar in the corner when the view resizes
    def eventFilter(self, obj, ev):
        if obj is self.ui.graphicsView.viewport() and ev.type() == QtCore.QEvent.Type.Resize:
            self._toolbar.move(6, 6)
            self._toolbar.adjustSize()
        return super().eventFilter(obj, ev)

    # ----- level helpers (sync image + histogram) ----------------------------
    def _set_levels(self, lo: float, hi: float) -> None:
        """Update both the ImageItem and the histogram widget region."""
        self.getImageItem().setLevels((lo, hi))
        if getattr(self.ui, "histogram", None) is not None:
            try:
                self.ui.histogram.setLevels(lo, hi)   # modern API
            except Exception:
                if hasattr(self.ui.histogram, "region"):
                    self.ui.histogram.region.setRegion((lo, hi))

    def _apply_levels_minmax(self, arr) -> bool:
        if arr is None:
            return False
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return False
        self._set_levels(lo, hi)
        return True

    def _on_autoscale_toggled(self, state: bool):
        self._autoscale = bool(state)
        if self._autoscale and self._img_np is not None:
            # Apply min/max on the current frame immediately
            self._apply_levels_minmax(self._img_np)

    def _apply_auto_levels_once(self):
        """One-shot min/max level set; leaves autoscale OFF afterward."""
        if self._apply_levels_minmax(self._img_np):
            self._chk_autoscale.setChecked(False)

    # ----- overlays: ROIs ----------------------------------------------------
    def _clear_rois(self):
        vb = self.getView()
        for it in self._roi_curves:
            vb.removeItem(it)
        for it in self._roi_labels:
            vb.removeItem(it)
        self._roi_curves.clear()
        self._roi_labels.clear()

    def _draw_rois(self, rois, counts=None):
        self._clear_rois()
        if rois is None:
            return
        rois = np.asarray(rois)
        if rois.ndim != 2 or rois.shape[1] != 4:
            return

        vb = self.getView()
        have_counts = counts is not None and len(counts) == len(rois)
        for i, (y0, y1, x0, x1) in enumerate(rois):
            xs = np.array([x0, x1, x1, x0, x0], float)
            ys = np.array([y0, y0, y1, y1, y0], float)
            curve = pg.PlotCurveItem(xs, ys, pen=pg.mkPen((255, 255, 255, 200), width=2))
            curve.setZValue(10)
            vb.addItem(curve)
            self._roi_curves.append(curve)
            if have_counts:
                cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                lbl = pg.TextItem(text=str(int(counts[i])), anchor=(0.5, 0.5))
                lbl.setPos(cx, cy)
                lbl.setZValue(11)
                vb.addItem(lbl)
                self._roi_labels.append(lbl)

    # ----- mouse readout (constant-size, top row) ----------------------------
    def _on_mouse_moved(self, evt):
        if self._img_np is None:
            self._pos_label.setText("")
            return
        pos = evt[0]
        vb = self.getView()
        if not vb.sceneBoundingRect().contains(pos):
            self._pos_label.setText("")
            return
        p = vb.mapSceneToView(pos)
        # floor so any position within [i, i+1) selects pixel i
        x = int(np.floor(p.x()))
        y = int(np.floor(p.y()))
        n_y, n_x = self._img_np.shape[:2]
        if 0 <= x < n_x and 0 <= y < n_y:
            val = self._img_np[y, x]
            self._pos_label.setText(f"x={x}, y={y}, val={val}")
        else:
            self._pos_label.setText("")

    # ----- ARTIQ hook --------------------------------------------------------
    def data_changed(self, value, metadata, persist, mods):
        img = value.get(self.args.image)
        if img is not None:
            arr = np.array(img)
            self._img_np = arr
            # Draw without autoLevels; apply min/max ourselves if autoscale is on.
            self.setImage(arr, autoLevels=False)
            if self._autoscale:
                self._apply_levels_minmax(arr)

        rois_name = getattr(self.args, "rois", None)
        counts_name = getattr(self.args, "counts", None)
        rois = value.get(rois_name) if rois_name else None
        counts = value.get(counts_name) if counts_name else None
        self._draw_rois(rois, counts)


def main():
    applet = SimpleApplet(ImageWithROIs)
    applet.add_dataset("image", "2D image dataset")
    applet.add_dataset("rois", "Optional ROI list: (y0,y1,x0,x1)", required=False)
    applet.add_dataset("counts", "Optional counts per ROI", required=False)
    applet.run()


if __name__ == "__main__":
    main()
