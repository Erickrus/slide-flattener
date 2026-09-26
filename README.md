# Slide Flattener

Turn messy photos of projected slides into clean, straight-on screenshots —
automatically finding each slide's four corners, flattening the perspective, and
fixing the color.

<img src="https://github.com/Erickrus/slide-flattener/blob/main/screenshot.png?raw=true" >

---

## What this does, and why it's valuable

You photographed a presentation screen from a seat, holding your phone at an
angle. Every shot has the same problems:

- **Perspective distortion** — the slide is a skewed trapezoid, not a rectangle.
- **Blue cast** — the projector light tints everything blue.
- **Low contrast** — dim projection, washed-out whites, murky blacks.
- **Dead space** — dark room, ceiling, and desks around the slide.

Copying those into a document means cropping each one by hand in Photoshop and
fighting the color. This tool does it automatically, in one pass, for the whole
folder:

1. It **finds the slide's four corners** by recognizing the screen's "white +
   blue" signature.
2. It **flattens** the skewed quadrilateral into a perfect rectangle (16:9 by
   default).
3. It **Auto-Colors** the result — removes the blue cast, makes whites white and
   blacks black.

The result is a clean, readable, shareable copy of each slide — as if you had the
original deck exported to images, instead of a pile of phone photos.

The process is mostly automatic. For the handful of photos where auto-detection
guesses wrong (bad angle, partial screen, a hand in the way), you fix it in a few
seconds by **dragging the corners** in a small UI — no Photoshop required.

---

## Quick start

```bash
# one-time setup
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt

# use the tool (every time)
.venv/bin/python annotate.py
```

That's it. `annotate.py` opens a window for every image in the folder so you can
review them and fix the few bad ones. Output is written to `./flattened/`.

> On macOS with Homebrew Python, always run via `.venv/bin/python` (the system
> Python refuses `pip install`).

---

## Using `annotate.py`

This is the only command you need. It opens two windows:

- **`edit`** — the photo with the detected slide outlined (green lines, red corner
  dots). There is a gray margin around the photo so a corner that lands outside
  the image can still be grabbed and pulled back in.
- **`preview`** — the flattened + Auto-Colored result, updating live as you drag.

### Controls

| Action | How |
|--------|-----|
| Fix a corner | drag a red dot (preview updates live) |
| Previous slide | click `PREV` |
| Next slide | click `NEXT` |
| Redo auto-detection | click `RESET` |
| Change output shape | click `AR:16:9` — cycles `16:9 → 4:3 → 3:2 → 1:1 → 2:3 → 3:4 → 9:16 → auto` |
| Save & finish | click `DONE` (or press `Esc`) |

### How saving works

- **Implicit save** — moving to another slide (`PREV`/`NEXT`/`DONE`/`Esc`)
  automatically saves the current one. There is no separate "save" button.
- **Untouched slides are fine** — if you skip past a slide without dragging
  anything, its auto-detected result is saved as-is.
- **No wrap-around** — `PREV`/`NEXT` stop at the first/last slide.
- The current aspect ratio is shown on the `AR` button and in the title line.

Typical workflow: click `NEXT` through the deck, stop only where a slide looks
wrong, drag its corners, keep going, click `DONE` at the end.

To start with a different aspect ratio, or to review only some files:

```bash
.venv/bin/python annotate.py --aspect 4:3
.venv/bin/python annotate.py IMG_0847.jpg IMG_0880.jpg
```

---

## Aspect ratio

The output shape is controlled by the `AR` button (or `--aspect` on the command
line). Default is **16:9**. Choose whatever matches your slides — the same photos
can be exported as 4:3, 1:1, 9:16 (portrait), or `auto` (each image's own
measured shape).

---

## Auto Color

The flattened image still carries the projector's color cast (usually blue).
Auto Color fixes it in two steps (Photoshop-style):

1. **White balance (gray-world)** — scales the channels so the average color is
   neutral, removing the cast.
2. **Contrast stretch on luminance** — clips the 0.5% / 99.5% brightness
   percentiles and stretches to full range, making whites white and blacks black
   *without* re-introducing a color cast.

---

## How the 4 corners are found

The core of the tool is locating the four corners of the projection screen in a
photo, so it can be warped flat.

### The key insight: the screen is bright *and* has a distinctive tint

A projected screen is hard to find by brightness or edges — slide text cuts the
screen up, and rooms contain other bright objects. But it has two reliable
properties:

1. It is **bright** — lit by the projector, clearly brighter than the room.
2. It is **tinted** — its light has a color cast that differs from the room's
   ambient light. Usually that cast is cool/blue, but it can be warm or green
   depending on the projector and the room lighting.

So instead of hard-coding "blue", the detector **auto-discovers the screen's
color signature**: it finds whichever channel contrast best separates the bright
screen from the dark room, then segments on exactly that contrast.

### The pipeline (`detect_slide_quad`)

1. **Downscale** — resize so the longest side is 1600 px (for speed); coordinates
   are scaled back at the end.

2. **Split bright vs dark** — Otsu threshold on luminance separates the bright
   screen from the dark room (falling back to a percentile split if the histogram
   is not bimodal).

3. **Auto-detect the dominant contrast** — for each pairwise channel difference
   (`B − R`, `B − G`, `G − R`), measure how far apart the bright region's mean is
   from the dark region's mean (normalized by the contrast's spread). Keep the
   contrast with the largest separation, and remember its direction (the screen is
   higher *or* lower in that contrast):

   ```python
   separation = |mean(bright) − mean(dark)| / std        # per contrast
   dominant   = contrast with the largest separation
   ```

   For a blue screen `B − R` wins (screen high); for a warm projector `B − R` also
   wins but with the opposite sign; for a green tint `G − R` wins. No hard-coded
   color.

4. **Threshold** the dominant contrast, in its detected direction, with raw values
   (`40 … 160`). Using raw differences — not z-scores — matters: a dark room has
   all channels near zero, so its faint tint has a tiny magnitude and is naturally
   excluded, whereas z-scoring would wrongly amplify it.

5. **Morphological closing** — a square kernel (`21×21` and `41×41` are tried)
   fills the dark holes left by slide text and joins the screen into one solid
   blob.

6. **Find contours** — each external contour of the closed mask is a *candidate*
   screen.

7. **Fit a quadrilateral** (`_quad_from_contour`) to each candidate:
   - take the contour's **convex hull**, then
   - `cv2.approxPolyDP` with increasing epsilon (`0.02 … 0.15` of the perimeter)
     until a 4-point convex polygon appears, or
   - fall back to `cv2.minAreaRect` (the minimum rotated rectangle) otherwise.

8. **Filter & score** — discard candidates smaller than 5% of the image, then
   score the rest:

   ```
   score = area × exp( −((ratio − ratio_hint) / 0.35)² )
   ```

   where `ratio = width/height` and `ratio_hint` is the expected slide ratio
   (16/9 by default). This prefers candidates shaped like a slide, and rejects
   things like a wide desk or a strongly-colored object that is the wrong shape.

9. **Pick the best** — the highest-scoring candidate across all thresholds and
   kernel sizes wins.

10. **Brightness fallback** — if no color contrast finds the screen (e.g. a
    perfectly neutral screen), repeat the contour→quad pipeline on plain
    brightness (`gray > t`).

11. **Order the corners** — `order_points` sorts them into a fixed cyclic order
    **top-left, top-right, bottom-right, bottom-left**:

    ```python
    top_left      = point with smallest (x + y)
    bottom_right  = point with largest  (x + y)
    top_right     = point with smallest (y − x)
    bottom_left   = point with largest  (y − x)
    ```

12. **Scale back** — divide by the downscale factor to get full-resolution
    coordinates.

### Perspective transform

With the four corners ordered, `cv2.getPerspectiveTransform` maps them to the
four corners of the target rectangle and `cv2.warpPerspective` produces the flat
image. The target size is chosen to match the requested aspect ratio while
preserving the slide's detail.

### Why this beats simple edge/brightness detection

- The slide background is fragmented by content, but the screen's tint is uniform
  across the whole screen regardless of content.
- The color signature is auto-detected per photo, so it is not tied to a specific
  projector's color temperature.
- Other bright (even more strongly-colored) objects in the room are discarded by
  the ratio score and the bright/dark split.
- The sweep over thresholds and kernels absorbs the large exposure and
  white-balance differences between phones and shots.

---

## Will it work if I move seats or change projector?

Short answer: **changing position is always fine; a different projector is also
fine.** The detector does not assume a particular color — it auto-detects the
screen's tint per photo. And it does not care where the screen is, how big it is,
or how much perspective there is, so:

- **Moving seats / changing angle / distance** → still works. The screen is just
  a differently-shaped quadrilateral; the same bright, tinted region is found.
- **Zooming closer / screen touching the frame edge** → still detected, and the
  UI margin lets you pull a stray corner back in.
- **Cool/neutral, warm, or green projector** → works. The detector picks whichever
  of `B − R`, `B − G`, `G − R` best separates the screen from the room, in either
  direction.

The one case that still needs help is when the screen is **indistinguishable from
the room** — i.e. it is neither brighter nor a different color (for example a
very dim projector in a brightly lit room, a TV showing a nearly-black slide, or
a colored wall with no lit screen). Then auto-detection has nothing to key off and
you'll drag the corners manually in `annotate.py` (which always works regardless).

---

## `flatten_slides.py` (optional batch engine)

`annotate.py` is the only script you normally need; it imports
`flatten_slides.py` under the hood, so keep both files together.

If you ever want to process a folder with **no review step at all** (e.g. you
trust the auto-detection), you can run the engine directly:

```bash
.venv/bin/python flatten_slides.py            # all images in cwd
.venv/bin/python flatten_slides.py --debug    # also write detection overlays
.venv/bin/python flatten_slides.py --aspect 4:3
.venv/bin/python flatten_slides.py --manual IMG_0822.jpg   # click corners by hand
```

`--debug` writes `./debug/<name>_overlay.jpg` — the detected quad drawn on the
source photo — so you can verify detection at a glance.

---

## Troubleshooting

- **A slide came out wrong (wrong region, clipped, skewed).**
  Run `.venv/bin/python annotate.py`, drag the red corners on that slide, click
  `DONE`.

- **Text is sideways or upside-down.**
  The source photo may have a stale EXIF rotation flag. Rotate the source image
  first, or fix the corners in `annotate.py`.

- **The screen is indistinguishable from the room (no brightness/color contrast).**
  Auto-detection has nothing to key off. `annotate.py` still works — drag all
  four corners manually. (`flatten_slides.py --manual` also exists.)

---

## Project layout

```
annotate.py         # the review UI you run (imports flatten_slides)
flatten_slides.py   # core engine: corner detection, warp, Auto Color
requirements.txt
README.md
flattened/          # output (created on run)
debug/              # detection overlays (created with --debug)
```
