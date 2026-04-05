"""
detect_lines.py
---------------
Detects colored lines in a chart image.

Pipeline:
  1. Detect plot area via axis-line rules
  2. Collect non-background pixels inside the plot, grouped by quantized color
  3. Per (color, x): keep only the TOPMOST y
  4. Cluster nearby colors (union-find on RGB distance)
  5. Drop clusters with too few points (noise)
  6. Remove horizontal reference lines: y_std below threshold = flat line
  7. Save annotated image

Usage:
    python detect_lines.py chart.png

Requirements:
    pip install pillow numpy
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw


# ── Plot area detection ───────────────────────────────────────────────────────

def detect_plot_area(arr):
    h, w  = arr.shape[:2]
    rgb   = arr[:, :, :3].astype(int)
    dark  = rgb.max(axis=2) < 120   # catches gray axis lines (not just pure black)
    light = rgb.min(axis=2) > 200

    best_x, best_run = 0, 0
    for x in range(w // 4):
        run = cur = 0
        for y in range(h):
            if dark[y, x]: cur += 1; run = max(run, cur)
            else: cur = 0
        if run > best_run: best_run = run; best_x = x

    # Rule 1: top of y-axis
    y_top = next((y for y in range(1, h)
                  if dark[y, best_x] and light[y-1, best_x]), 0)

    # Rule 3: x_max and y_origin.
    # Step 1 — find a candidate axis row from the y-axis column
    # (bottommost dark pixel in that column).
    x_axis_bottom = next((y for y in range(h - 2, y_top, -1)
                          if dark[y, best_x]), h - 1)

    # Step 2 — find x_max on that candidate row
    x_max = next((x for x in range(w - 2, best_x, -1)
                  if dark[x_axis_bottom, x] and light[x_axis_bottom, x + 1]), w - 1)

    # Step 3 — sweep upward from x_max (NOT from best_x) until the pixel
    # above is no longer dark. At x_max there is no vertical axis above,
    # so the sweep stops at the top edge of the x-axis line only.
    x_axis_row = x_axis_bottom
    while x_axis_row > y_top + 1 and dark[x_axis_row - 1, x_max]:
        x_axis_row -= 1

    # Origin = (x of y-axis column, y of x-axis row)
    return {
        "x_min":    best_x,      # left edge: x coordinate of y-axis
        "x_max":    x_max,       # right edge: last dark pixel on x-axis row
        "y_top":    y_top,       # top of y-axis (highest price)
        "y_origin": x_axis_row,  # origin: y coordinate of x-axis row
    }


# ── Pixel collection ──────────────────────────────────────────────────────────

def quantize(r, g, b, bucket=16):
    return (r//bucket*bucket, g//bucket*bucket, b//bucket*bucket)

def is_background(r, g, b):
    if r > 200 and g > 200 and b > 200: return True   # white / near-white
    if (r+g+b)/3 < 80:                  return True   # black/dark
    if max(r,g,b) - min(r,g,b) < 30:   return True   # gray (unsaturated)
    return False

def collect_pixels(img_path):
    """
    Two-pass pixel collection:

    Pass 1 — collect ALL non-background topmost pixels, cluster them,
    identify which clusters are horizontal reference lines (y_std ≈ 0),
    and record the y-rows those reference lines occupy.

    Pass 2 — re-collect, but this time skip any pixel whose quantized
    color belongs to a reference-line cluster. This ensures reference-line
    pixels never hijack the topmost-y of a data-line column, even when
    the data line happens to pass through the same y position.

    Fringe filter applied in both passes: skip pixels whose pixel above
    AND pixel below are both background (antialiasing edge artifacts).
    """
    img  = Image.open(img_path).convert("RGB")
    arr  = np.array(img)
    h, w = arr.shape[:2]
    plot = detect_plot_area(arr)
    x_min, x_max = plot["x_min"], plot["x_max"]
    y_top, y_origin = plot["y_top"], plot["y_origin"]

    def is_bg(y, x):
        if y < 0 or y >= h: return True
        return is_background(int(arr[y,x,0]), int(arr[y,x,1]), int(arr[y,x,2]))

    # ── Pass 1: collect everything ────────────────────────────────────────
    top1 = defaultdict(dict)
    for y in range(y_top, y_origin + 1):
        for x in range(x_min, x_max + 1):
            r, g, b = int(arr[y,x,0]), int(arr[y,x,1]), int(arr[y,x,2])
            if is_background(r, g, b): continue
            if is_bg(y-1, x) and is_bg(y+1, x): continue   # fringe
            c = quantize(r, g, b)
            if x not in top1[c] or y < top1[c][x]:
                top1[c][x] = y

    colors1 = {c: sorted(xy.items()) for c, xy in top1.items()}

    # ── Identify reference-line y-rows ────────────────────────────────────
    # Look at RAW color buckets (before clustering) — each individual color
    # bucket that is flat (y_std < 1) and has enough points is a ref line.
    # This avoids the chain-linking problem where antialiasing bridges a
    # ref line color into the data line cluster, hiding its flatness.
    ref_rows = set()
    for c, pts in colors1.items():
        if len(pts) < 10: continue
        ys = np.array([y for _, y in pts], dtype=float)
        if ys.std() < 1.0:
            for _, y in pts:
                ref_rows.update([y-1, y, y+1])

    if ref_rows:
        row_ranges = []
        sorted_rows = sorted(ref_rows)
        start = sorted_rows[0]
        prev  = sorted_rows[0]
        for r2 in sorted_rows[1:]:
            if r2 > prev + 1:
                row_ranges.append(f"{start}-{prev}")
                start = r2
            prev = r2
        row_ranges.append(f"{start}-{prev}")
        print(f"  Reference-line rows excluded in pass 2: {', '.join(row_ranges)}")

    # ── Pass 2: re-collect, skipping reference-line y-rows ───────────────
    top2 = defaultdict(dict)
    for y in range(y_top, y_origin + 1):
        if y in ref_rows: continue                           # skip ref line rows
        for x in range(x_min, x_max + 1):
            r, g, b = int(arr[y,x,0]), int(arr[y,x,1]), int(arr[y,x,2])
            if is_background(r, g, b): continue
            if is_bg(y-1, x) and is_bg(y+1, x): continue   # fringe
            c = quantize(r, g, b)
            if x not in top2[c] or y < top2[c][x]:
                top2[c][x] = y

    colors2 = {c: sorted(xy.items()) for c, xy in top2.items()}
    return {"plot": plot, "colors": colors2, "img": img}


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_colors(colors, dist_thresh=48, min_points=20):
    """Merge similar colors; keep topmost y per x across members."""
    keys   = list(colors.keys())
    parent = {c: c for c in keys}

    def find(c):
        while parent[c] != c: parent[c] = parent[parent[c]]; c = parent[c]
        return c
    def union(a, b): parent[find(a)] = find(b)

    for i, c1 in enumerate(keys):
        for c2 in keys[i+1:]:
            if ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**.5 <= dist_thresh:
                union(c1, c2)

    groups = defaultdict(list)
    for c in keys: groups[find(c)].append(c)

    result = {}
    for root, members in groups.items():
        cr = cg = cb = tw = 0
        x_miny = {}
        for c in members:
            pts = colors[c]; w = len(pts)
            cr += c[0]*w; cg += c[1]*w; cb += c[2]*w; tw += w
            for x, y in pts:
                if x not in x_miny or y < x_miny[x]: x_miny[x] = y
        if tw == 0 or len(x_miny) < min_points: continue
        result[root] = {
            "center": (int(cr/tw), int(cg/tw), int(cb/tw)),
            "points": sorted(x_miny.items())
        }
    return result


# ── Horizontal line removal ───────────────────────────────────────────────────

def remove_horizontal_lines(clusters, y_std_thresh=3.0):
    """
    Drop clusters where y_std < threshold.
    A real data line varies in y; a horizontal reference line does not.
    """
    kept = {}
    for root, info in clusters.items():
        ys  = np.array([y for _, y in info["points"]], dtype=float)
        std = float(ys.std())
        if std >= y_std_thresh:
            kept[root] = info
        else:
            print(f"  Removed flat line : rgb{info['center']}  "
                  f"{len(info['points'])} pts  y_std={std:.2f}")
    return kept


# ── Merge spatially overlapping clusters ─────────────────────────────────────

def merge_overlapping_clusters(clusters, y_proximity=15):
    """
    Merge clusters that represent the same data line drawn in multiple colors
    (e.g. a red stroke on top of a green fill, or antialiasing variants).

    Two clusters are merged if their y positions at shared x columns are
    within y_proximity pixels of each other — they're tracing the same line.
    Keeps the topmost y per x across both clusters after merging.
    Runs iteratively until no further merges are possible.
    """
    changed = True
    while changed:
        changed = False
        keys = sorted(clusters.keys(), key=lambda k: -len(clusters[k]["points"]))
        if len(keys) < 2:
            break

        for i, k1 in enumerate(keys):
            merged = False
            ys1 = dict(clusters[k1]["points"])
            for k2 in keys[i+1:]:
                ys2 = dict(clusters[k2]["points"])

                # Check y proximity at shared x positions
                shared_x = set(ys1.keys()) & set(ys2.keys())
                if len(shared_x) >= 5:
                    diffs = [abs(ys1[x] - ys2[x]) for x in shared_x]
                    if np.median(diffs) > y_proximity:
                        continue
                else:
                    # No/few shared x — check nearest-neighbour proximity
                    if not ys1 or not ys2: continue
                    x1min, x1max = min(ys1), max(ys1)
                    x2min, x2max = min(ys2), max(ys2)
                    # Must have overlapping x ranges (within 5px)
                    if x2min > x1max + 5 or x2max < x1min - 5:
                        continue
                    diffs = []
                    for x in ys2:
                        nx = min(ys1.keys(), key=lambda rx: abs(rx-x))
                        diffs.append(abs(ys2[x] - ys1[nx]))
                    if not diffs or np.median(diffs) > y_proximity:
                        continue

                # Merge k2 into k1: keep topmost y per x
                merged_ys = dict(clusters[k1]["points"])
                for x, y in clusters[k2]["points"]:
                    if x not in merged_ys or y < merged_ys[x]:
                        merged_ys[x] = y

                n1 = len(clusters[k1]["points"])
                n2 = len(clusters[k2]["points"])
                total = n1 + n2
                cr = clusters[k1]["center"][0]*n1 + clusters[k2]["center"][0]*n2
                cg = clusters[k1]["center"][1]*n1 + clusters[k2]["center"][1]*n2
                cb = clusters[k1]["center"][2]*n1 + clusters[k2]["center"][2]*n2

                print(f"  Merged rgb{clusters[k2]['center']} "
                      f"({len(clusters[k2]['points'])} pts) into "
                      f"rgb{clusters[k1]['center']} "
                      f"({len(clusters[k1]['points'])} pts)")

                clusters[k1] = {
                    "center": (cr//total, cg//total, cb//total),
                    "points": sorted(merged_ys.items())
                }
                del clusters[k2]
                changed = True
                merged = True
                break
            if merged:
                break

    return clusters


# ── Visualization ─────────────────────────────────────────────────────────────

MARKERS = [
    (255, 50,  50), (50,  50, 255), (255,165,  0),
    (160, 32, 240), (  0,180, 180), (255, 20,147),
    (  0,160,   0), (255,215,  0),
]

def save_annotated(img, plot, clusters, out_path):
    canvas  = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    x0,x1   = plot["x_min"], plot["x_max"]
    y0,y1   = plot["y_top"], plot["y_origin"]

    draw.rectangle([x0,y0,x1,y1], outline=(255,140,0,180), width=2)
    for pt, col in [((x0,y0),(220,50,50,255)),
                    ((x0,y1),(50,50,220,255)),
                    ((x1,y1),(50,200,50,255))]:
        draw.ellipse([pt[0]-7,pt[1]-7,pt[0]+7,pt[1]+7], fill=col, outline=(0,0,0,255))

    for i, (_,info) in enumerate(sorted(clusters.items(), key=lambda c:-len(c[1]["points"]))):
        m = MARKERS[i % len(MARKERS)]
        for x, y in info["points"]:
            draw.ellipse([x-3,y-3,x+3,y+3], fill=(*m,220))

    Image.alpha_composite(canvas, overlay).convert("RGB").save(out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def process(img_path, y_std_thresh=3.0, dist_thresh=48, min_points=20):
    print(f"\nProcessing: {img_path}")
    data     = collect_pixels(img_path)
    plot     = data["plot"]
    colors   = data["colors"]

    print(f"  Plot area : x={plot['x_min']}..{plot['x_max']}  "
          f"y={plot['y_top']}..{plot['y_origin']}")
    print(f"  Raw color buckets : {len(colors)}")

    clusters = cluster_colors(colors, dist_thresh=dist_thresh, min_points=min_points)
    print(f"  After clustering  : {len(clusters)} cluster(s)")

    clusters = remove_horizontal_lines(clusters, y_std_thresh=y_std_thresh)
    print(f"  After horiz filter: {len(clusters)} cluster(s)")

    clusters = merge_overlapping_clusters(clusters, y_proximity=15)
    print(f"  After merge       : {len(clusters)} data line(s)\n")

    for i, (_,info) in enumerate(sorted(clusters.items(), key=lambda c:-len(c[1]["points"]))):
        pts = info["points"]
        ys  = np.array([y for _,y in pts])
        print(f"  Line {i+1}: rgb{info['center']}  {len(pts)} pts  "
              f"y={ys.mean():.0f}±{ys.std():.0f}  x={pts[0][0]}..{pts[-1][0]}")

    script_dir = Path(__file__).parent
    out = str(script_dir / (Path(img_path).stem + "_processed.png"))
    save_annotated(data["img"], plot, clusters, out)
    print(f"\n  Saved: {out}")
    data["clusters"] = clusters
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_lines.py image.png"); sys.exit(1)
    for path in sys.argv[1:]:
        process(path)
