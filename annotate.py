#!/usr/bin/env python3
"""
Interactive UI to fix slide corners and re-flatten.

Two windows:
  - "edit":    the photo + auto-detected quad (green lines, red dots), with
               clickable Prev/Reset/Next/AR/Done buttons along the bottom, and a
               margin around the photo so corners that fall outside the image
               can still be grabbed and dragged back.
  - "preview": live flattened + auto-colored result, updating as you drag.

Behavior:
  - Dragging a red dot adjusts a corner; the preview updates live.
  - Navigating (Prev / Next / Done) implicitly saves the current result.
  - If you don't touch an image, its auto-detected result is saved as-is.
  - Prev/Next stop at the first/last image (no wrap-around).
  - The "AR" button cycles the output aspect ratio (16:9, 4:3, 3:2, 1:1, 2:3,
    3:4, 9:16, auto). The current ratio is shown on the button and in the title.

Usage:
    python annotate.py                 # review + fix every image in the current dir
    python annotate.py --aspect 4:3    # start with a different aspect ratio
    python annotate.py IMG_0847.jpg IMG_0860.jpg   # review a specific subset
"""

import argparse
import os
import sys

import cv2
import numpy as np

import flatten_slides as fs

EDIT_WIN = "edit"
PREVIEW_WIN = "preview"

DRAG_RADIUS = 20
EDIT_MAX = 1000
PREVIEW_MAX = 460
MARGIN = 100
BAR_H = 56
ASPECT_CYCLE = ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "auto"]
DEFAULT_ASPECT = "16:9"


def collect_sources(args):
    if args.images:
        return [f for f in args.images if os.path.exists(f)]

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted(f for f in os.listdir(".") if f.lower().endswith(exts))


def draw_button(canvas, x0, y0, x1, y1, text, fill=(70, 70, 70)):
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cx = x0 + (x1 - x0 - tw) // 2
    cy = y0 + (y1 - y0 + th) // 2
    cv2.putText(canvas, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


class Annotator:
    def __init__(self, sources, out_dir="flattened", aspect=DEFAULT_ASPECT):
        self.sources = sources
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.idx = 0
        self.full = None
        self.small = None
        self.scale = 1.0
        self.corners = None
        self.drag = None
        self.edited = False
        self.buttons = []  # (x0, y0, x1, y1, action)
        self.set_aspect(aspect)

        self.disp_w = 0
        self.disp_h = 0
        self.canvas_w = 0
        self.canvas_h = 0
        self.bar_top = 0

        cv2.namedWindow(EDIT_WIN, cv2.WINDOW_NORMAL)
        cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(EDIT_WIN, self.on_mouse)

        self.load(0)
        self.run()

    # ---- geometry ----
    def recompute_geometry(self):
        self.disp_w = self.small.shape[1]
        self.disp_h = self.small.shape[0]
        self.canvas_w = self.disp_w + 2 * MARGIN
        self.canvas_h = self.disp_h + 2 * MARGIN + BAR_H
        self.bar_top = self.disp_h + 2 * MARGIN

    # ---- aspect ratio ----
    def set_aspect(self, s):
        self.aspect_str = s
        self.aspect = fs.parse_aspect(s)

    def ratio_hint(self):
        if self.aspect is None:
            return 16.0 / 9.0
        return self.aspect[0] / self.aspect[1]

    def cycle_aspect(self):
        i = ASPECT_CYCLE.index(self.aspect_str) if self.aspect_str in ASPECT_CYCLE else 0
        self.set_aspect(ASPECT_CYCLE[(i + 1) % len(ASPECT_CYCLE)])
        self.redraw()

    # ---- data loading ----
    def load(self, i):
        if not self.sources:
            return
        self.idx = max(0, min(i, len(self.sources) - 1))
        path = self.sources[self.idx]

        self.full = cv2.imread(path)
        if self.full is None:
            print(f"[skip] cannot read {path}")
            return

        self.scale = EDIT_MAX / max(self.full.shape[:2])
        self.small = cv2.resize(self.full, None, fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
        self.recompute_geometry()

        corners = fs.detect_slide_quad(self.full, ratio_hint=self.ratio_hint())
        if corners is None:
            h, w = self.full.shape[:2]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        self.corners = fs.order_points(corners).astype("float32")

        self.drag = None
        self.edited = False
        self.redraw()

    # ---- drawing ----
    def redraw(self):
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 45, np.uint8)
        canvas[MARGIN:MARGIN + self.disp_h, MARGIN:MARGIN + self.disp_w] = self.small

        pts = (self.corners * self.scale).astype("int32") + MARGIN
        for i in range(4):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 3)
        for i in range(4):
            cv2.circle(canvas, tuple(pts[i]), 8, (0, 0, 255), -1)

        name = os.path.basename(self.sources[self.idx])
        title = f"{self.idx + 1}/{len(self.sources)}  {name}   AR {self.aspect_str}"
        if self.edited:
            title += "  (edited)"
        cv2.putText(canvas, title, (10, self.canvas_h - BAR_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # bottom button bar
        ar_label = "AR:" + self.aspect_str
        labels = ["PREV", "RESET", "NEXT", ar_label, "DONE"]
        actions = ["prev", "reset", "next", "ar", "done"]
        bw = 100
        gap = 10
        total = len(labels) * bw + (len(labels) - 1) * gap
        start_x = (self.canvas_w - total) // 2
        y0 = self.bar_top + 8
        y1 = self.canvas_h - 8
        self.buttons = []
        for k, (lab, act) in enumerate(zip(labels, actions)):
            x0 = start_x + k * (bw + gap)
            x1 = x0 + bw
            self.buttons.append((x0, y0, x1, y1, act))
            draw_button(canvas, x0, y0, x1, y1, lab)

        cv2.imshow(EDIT_WIN, canvas)
        self.update_preview()

    def update_preview(self):
        corners_small = self.corners * self.scale
        out_w, out_h = fs.output_size(corners_small, self.aspect)
        warped = fs.four_point_transform(self.small, corners_small, out_w, out_h)
        warped = fs.auto_color(warped)

        h, w = warped.shape[:2]
        k = PREVIEW_MAX / max(w, h)
        if k < 1.0:
            warped = cv2.resize(warped, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        cv2.imshow(PREVIEW_WIN, warped)

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y >= self.bar_top:
                self.handle_button(x, y)
                return
            self.drag = self.nearest_corner(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.corners[self.drag] = [(x - MARGIN) / self.scale, (y - MARGIN) / self.scale]
            self.edited = True
            self.redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def nearest_corner(self, x, y):
        pts = (self.corners * self.scale).astype("float32") + MARGIN
        d = np.linalg.norm(pts - np.array([x, y], dtype="float32"), axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= DRAG_RADIUS else None

    def handle_button(self, x, y):
        for (x0, y0, x1, y1, act) in self.buttons:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if act == "prev":
                    self.go_prev()
                elif act == "next":
                    self.go_next()
                elif act == "reset":
                    self.reset()
                elif act == "ar":
                    self.cycle_aspect()
                elif act == "done":
                    self.save_current()
                    self.quit()
                return

    # ---- actions ----
    def reset(self):
        path = self.sources[self.idx]
        corners = fs.detect_slide_quad(self.full, ratio_hint=self.ratio_hint())
        if corners is None:
            h, w = self.full.shape[:2]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        self.corners = fs.order_points(corners).astype("float32")
        self.edited = False
        self.redraw()

    def save_current(self):
        out_w, out_h = fs.output_size(self.corners, self.aspect)
        warped = fs.four_point_transform(self.full, self.corners, out_w, out_h)
        warped = fs.auto_color(warped)

        name = os.path.basename(self.sources[self.idx])
        base, ext = os.path.splitext(name)
        out_path = os.path.join(self.out_dir, base + "_flat" + ext)
        cv2.imwrite(out_path, warped)
        print(f"[saved] {out_path}")
        return out_path

    def go_next(self):
        self.save_current()
        if self.idx < len(self.sources) - 1:
            self.load(self.idx + 1)

    def go_prev(self):
        self.save_current()
        if self.idx > 0:
            self.load(self.idx - 1)

    def quit(self):
        self._running = False

    # ---- main loop ----
    def run(self):
        self._running = True
        print("Drag red dots. Buttons: PREV / RESET / NEXT / AR:<aspect> / DONE")
        while self._running:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # Esc
                self.save_current()
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Interactive slide-corner fixer.")
    ap.add_argument("images", nargs="*", help="source images to review (default: all images in cwd)")
    ap.add_argument("--out", default="flattened", help="output dir for fixed results")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT,
                    help="default output aspect ratio, e.g. 16:9, 4:3, 9:16, or 'auto'")
    args = ap.parse_args()

    sources = collect_sources(args)
    if not sources:
        print("No source images found.")
        sys.exit(1)

    Annotator(sources, out_dir=args.out, aspect=args.aspect)


if __name__ == "__main__":
    main()
