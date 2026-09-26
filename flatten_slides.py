#!/usr/bin/env python3
"""
Flatten perspective-distorted photos of slides into flat, straight-on screenshots.

Usage:
    python flatten_slides.py                      # auto-detect + flatten all images
    python flatten_slides.py --aspect 16:9        # force output aspect ratio (default)
    python flatten_slides.py --aspect auto        # use per-image estimate
    python flatten_slides.py --debug              # also save detection overlays for review
    python flatten_slides.py --manual IMG_0822.jpg  # click 4 corners by hand

Output is written to ./flattened/ by default; debug overlays go to ./debug/.
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

DEFAULT_ASPECT = "16:9"


def order_points(pts):
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def quad_dims(rect):
    """Return (max width, max height) of an ordered quad."""
    tl, tr, br, bl = rect
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br))
    return w, h


def four_point_transform(image, corners, target_w, target_h):
    rect = order_points(corners)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (target_w, target_h))


def _quad_from_contour(cnt, img_area, min_area_frac=0.05, ratio_hint=16.0 / 9.0):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.05, 0.10, 0.15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    quad = quad.reshape(4, 2).astype("float32")

    area = cv2.contourArea(quad)
    if area < min_area_frac * img_area:
        return None, 0.0

    w, h = quad_dims(order_points(quad))
    if h < 1:
        return None, 0.0
    ratio = w / h
    # Prefer candidates close to the expected slide ratio.
    score = area * math.exp(-((ratio - ratio_hint) / 0.35) ** 2)
    return quad, score


def detect_slide_quad(image, ratio_hint=16.0 / 9.0):
    """Return the best candidate slide quad in original-image coordinates, or None.

    Auto-detect the screen's color signature instead of assuming it is blue:

    1. Split the image into a bright region (screen) and a dark region (room)
       using Otsu on luminance.
    2. Among the three pairwise contrasts (B-R, B-G, G-R), pick the one that best
       separates the bright region from the dark region, and note its direction
       (the screen is higher *or* lower in that contrast).
    3. Threshold that single contrast in that direction (raw values, not
       z-scores) to segment the screen, then fit a quad.

    Brightness is used only as a fallback when no color contrast is found.
    """
    h, w = image.shape[:2]
    scale = 1600.0 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    img_area = H * W

    b = small[:, :, 0].astype("float32")
    g = small[:, :, 1].astype("float32")
    r = small[:, :, 2].astype("float32")
    lum = (b + g + r) / 3.0

    # Bright vs dark split (screen is bright, room is dark).
    ret, _ = cv2.threshold(lum.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = lum > ret
    dark = ~bright
    if bright.sum() < 0.02 * img_area or dark.sum() < 0.02 * img_area:
        bright = lum > np.percentile(lum, 60)
        dark = ~bright

    # Pick the dominant color contrast and its direction.
    features = {"B-R": b - r, "B-G": b - g, "G-R": g - r}
    dom = None
    dom_sign = 1
    dom_sep = -1.0
    for f in features.values():
        mb = f[bright].mean()
        md = f[dark].mean()
        sep = abs(mb - md) / (f.std() + 1e-6)
        if sep > dom_sep:
            dom_sep = sep
            dom = f
            dom_sign = 1 if mb > md else -1

    best = None
    best_score = 0.0

    # Threshold the dominant contrast in its direction with raw thresholds.
    if dom is not None:
        signed = dom * dom_sign
        for th in (40, 60, 80, 100, 120, 140, 160):
            mask = (signed > th).astype(np.uint8) * 255
            for ks in (21, 41):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
                closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    quad, score = _quad_from_contour(cnt, img_area, ratio_hint=ratio_hint)
                    if quad is not None and score > best_score:
                        best_score = score
                        best = quad

    # Fallback: brightness, only if no color contrast found the screen.
    if best is None:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        for t in (80, 100, 120, 140, 160):
            mask = (gray > t).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                quad, score = _quad_from_contour(cnt, img_area, ratio_hint=ratio_hint)
                if quad is not None and score > best_score:
                    best_score = score
                    best = quad

    if best is None:
        return None
    return best / scale


def parse_aspect(s):
    if s is None or s.lower() == "auto":
        return None
    a, b = s.split(":")
    return float(a), float(b)


def output_size(corners, aspect):
    """Choose an output size that preserves the forced aspect ratio and max detail."""
    w, h = quad_dims(order_points(corners))
    if aspect is None:
        return int(round(w)), int(round(h))
    aw, ah = aspect
    out_w = max(int(round(w)), int(round(h * aw / ah)))
    out_h = int(round(out_w * ah / aw))
    return out_w, out_h


def auto_color(image, shadow_clip=0.5, highlight_clip=0.5):
    """Photoshop-style Auto Color: white balance (gray-world) then contrast stretch.

    Removes the projector's color cast and makes whites white / blacks black while
    preserving the corrected color balance (stretches luminance, not each channel).
    """
    f = image.astype(np.float32)

    # 1. White balance via gray-world assumption: neutralize the mean of each channel.
    means = f.mean(axis=(0, 1))
    gray = means.mean()
    if gray < 1e-6:
        return image
    gains = np.clip(gray / np.maximum(means, 1e-6), 0.4, 2.5)
    f = f * gains

    # 2. Contrast stretch on luminance (uniform across channels, preserves color).
    lum = f.mean(axis=2)
    lo, hi = np.percentile(lum, (shadow_clip, 100.0 - highlight_clip))
    if hi - lo < 1.0:
        hi = lo + 1.0
    out = np.clip((f - lo) * (255.0 / (hi - lo)), 0.0, 255.0)

    return out.astype(np.uint8)


def draw_overlay(image, corners, out_path):
    disp = cv2.resize(image, None, fx=1600.0 / max(image.shape[:2]), fy=1600.0 / max(image.shape[:2]))
    q = (corners * (1600.0 / max(image.shape[:2]))).astype("int32").reshape(4, 2)
    for i in range(4):
        cv2.line(disp, tuple(q[i]), tuple(q[(i + 1) % 4]), (0, 255, 0), 4)
    for i in range(4):
        cv2.circle(disp, tuple(q[i]), 10, (0, 0, 255), -1)
    cv2.imwrite(out_path, disp)


def manual_select(image):
    """Click the 4 slide corners (any order). Esc cancels, Backspace undoes last point."""
    h, w = image.shape[:2]
    scale = 1200.0 / max(h, w)
    disp = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts = []  # stored in display coordinates

    window = "Click 4 slide corners (Esc=cancel, Backspace=undo, Enter=done)"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow(window, disp)

    cv2.imshow(window, disp)
    cv2.setMouseCallback(window, on_click)
    print("  Click the 4 slide corners in any order. Enter=done, Backspace=undo, Esc=cancel.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if key == 8 and pts:
            pts.pop()
            cv2.imshow(window, disp)
        if key in (13, 10) and len(pts) == 4:
            cv2.destroyAllWindows()
            corners = np.array(pts, dtype="float32") / scale
            return order_points(corners)


def main():
    ap = argparse.ArgumentParser(description="Flatten slide photos.")
    ap.add_argument("images", nargs="*", help="image files (default: all jpg/png in cwd)")
    ap.add_argument("--out", default="flattened", help="output directory")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, help="force aspect ratio e.g. 16:9 (default), or 'auto'")
    ap.add_argument("--manual", action="store_true", help="click 4 corners per image")
    ap.add_argument("--debug", action="store_true", help="save detection overlays to ./debug")
    ap.add_argument("--autocolor", dest="autocolor", action="store_true", default=True,
                    help="apply Auto Color after flattening (default)")
    ap.add_argument("--no-autocolor", dest="autocolor", action="store_false",
                    help="keep raw flattened colors")
    args = ap.parse_args()

    aspect = parse_aspect(args.aspect)
    ratio_hint = (aspect[0] / aspect[1]) if aspect is not None else (16.0 / 9.0)

    images = args.images
    if not images:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        images = sorted(f for f in os.listdir(".") if f.lower().endswith(exts))

    if not images:
        print("No images found.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    if args.debug:
        os.makedirs("debug", exist_ok=True)

    for path in images:
        print(path)
        img = cv2.imread(path)
        if img is None:
            print("  [skip] could not read")
            continue

        corners = None
        if args.manual:
            corners = manual_select(img.copy())
            if corners is None:
                print("  [skip] cancelled")
                continue
        else:
            corners = detect_slide_quad(img, ratio_hint=ratio_hint)

        if corners is None:
            print(f"  [fail] no quad found (try: python flatten_slides.py --manual {path})")
            continue

        out_w, out_h = output_size(corners, aspect)
        warped = four_point_transform(img, corners, out_w, out_h)
        if args.autocolor:
            warped = auto_color(warped)

        name, ext = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(args.out, name + "_flat" + ext)
        cv2.imwrite(out_path, warped)
        print(f"  [ok] {out_w}x{out_h} -> {out_path}")

        if args.debug:
            draw_overlay(img, corners, os.path.join("debug", name + "_overlay" + ext))


if __name__ == "__main__":
    main()
