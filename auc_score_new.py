"""
AUC-Judd (AUC-J) between the YOLOv8 segmentation saliency maps and gaze data.

=====================================================================
STEP 4 OF 4 - AUC-JUDD FOR EVERY FRAME
=====================================================================
Run this after step 1 (resample_gaze.py) and step 2 (yolo_heatmap_video.py).
It uses the same gaze files and the same saliency maps as step 3
(nss_score_new.py); the two scripts are independent, so run them in either
order, or only the one you need.

COMMANDS

    # every participant of one task
    python auc_score_new.py --gaze_dir Resample_Output\\blockTowerR --saliency_dir saliency_frames\\blockTowerR

    # many tasks at once: the map folder is picked per file from its name
    python auc_score_new.py --gaze_dir Resample_Output --recursive --saliency_root saliency_frames

    # a single gaze file
    python auc_score_new.py --gaze_file x.csv --saliency_dir saliency_frames\\blockTowerR --output result.csv

    # try it on the first three files first
    python auc_score_new.py --gaze_dir Resample_Output\\blockTowerR --saliency_dir saliency_frames\\blockTowerR --limit 3

OUTPUT (batch mode) - four files per task

    auc_Output\\<task>\\<gaze file name>              per-frame AUC-J for that
                                                      participant (frame,x,y,auc_j)
    auc_Output\\<task>\\<task>_auc_summary.csv        ONE ROW PER PARTICIPANT
                                                      (mean_auc_j, std, min/max,
                                                      rows scored, skips)
    auc_Output\\<task>\\<task>_auc_pooled_by_frame.csv   AUC-J per frame, using
                                                      every participant's gaze
                                                      at once (frame,n_fixations,auc_j)
    auc_Output\\<task>\\<task>_auc_pooled_summary.csv    one-row summary of the
                                                      pooled series

    For a "one number for the whole video" figure, use mean_auc_j of the
    pooled summary; for statistics over participants, use the auc_summary.csv
    (also a plain CSV, so it opens in Excel/SPSS/R directly).
    --output_dir DIR changes the root (default auc_Output), --group NAME the
    subfolder, and single-file mode with --output FILE writes just one CSV.

HOW TO READ THE SCORE

    0.5 = chance (the gaze region is no more salient than a random pixel)
    1.0 = the gaze sits on the most salient pixels of the frame
    <0.5 = the gaze sits on less salient pixels than average (a real result,
           not an error; the value is never clamped)

    The pooled numbers are only meaningful when the batch really contains all
    participants of that task - --limit makes them partial by definition.

HOW THE GAZE POINT IS USED

    Same model as in step 3: a small region around the recorded point rather
    than one pixel (default: Gaussian, sigma = 1 degree), and the ROC is
    weighted by that belief, so a region that half covers an object
    contributes half of its mass instead of an all-or-nothing decision.
    --kernel point gives the single-pixel reference behaviour, --kernel disk
    --radius 82 a hard 2 degree disk, and --fp_scope picks whether the
    false-positive pool excludes the gaze region (default) or not.

REQUIREMENTS

    Python 3 with pandas, numpy and opencv-python. gaze_kernel.py must be in
    the same folder as this script.

IF SOMETHING GOES WRONG

    - "no frame/x/y columns"  : pass --frame_col / --x_col / --y_col
    - "rows had no saliency map file" : the frame numbering of the gaze file
      does not match the maps - check that step 1 used the video's frame rate
    - "No saliency folder '<task>' under ..." : the task name inside the gaze
      file name has no matching folder in --saliency_root
=====================================================================

WHAT THIS DOES
--------------
AUC-Judd (Judd et al., 2012) treats the saliency map as a classifier of the
question "is this pixel fixated?" and reports the area under its ROC curve:

  1. Normalise the map (min-max; only the ranking matters)
  2. Sweep a threshold down over the saliency values that carry gaze evidence
  3. At each threshold t:
       TP(t) = gaze belief on pixels with value >= t / total belief
       FP(t) = non-fixated pixels with value >= t / non-fixated pixel count
  4. Anchor the curve at (0,0) and (1,1) and integrate it:
       AUC = trapz(FP, TP)

This follows the reference implementation behind the MIT saliency benchmark
(AUC_Judd.m by Judd / Bylinskii), including its treatment of the fixated region
as absent from the false-positive pool - which is also what the benchmark does
when it blurs the fixation map into a density and counts every non-zero pixel
as fixated:
https://raw.githubusercontent.com/cvzoya/saliency/master/code_forMetrics/AUC_Judd.m

Two deliberate differences, both about quantised saliency maps (which have
large flat regions and therefore many tied values):

  - the threshold is swept down over every 8-bit level instead of only over the
    fixation values, which is the exact staircase ROC of a quantised classifier
    and equals the standard rank-based AUC (sklearn.metrics.roc_auc_score) when
    one fixation is used. The reference's coarser sweep treats the jump across
    a group of tied values as vertical and therefore overestimates the area by
    half the mass of those ties;
  - ties are counted exactly instead of being broken with `rand/1e7` noise, so
    a run is reproducible to the last digit and needs no seed. The reference
    subtracts each fixation's *rank*, which makes FP move backwards inside a
    tied group and the ROC non-monotone; here the exact count is subtracted.

GAZE MODEL - WHAT ONE RECORDED POINT STANDS FOR
-----------------------------------------------
An eye tracker is only accurate to about 2 degrees of visual angle, and every
recording of this study carries its own measured calibration error
(calib_acc_raw / calib_acc_nudged: 0.77 - 4.40 degrees across participants).
A recorded gaze point is therefore modelled as a region of belief, and the ROC
is weighted by that belief:

  --kernel gaussian (default)  anisotropic Gaussian, --sigma_deg in degrees of
                               visual angle (default 1.0; 86.5% of the mass
                               then lies inside 2 degrees)
  --kernel disk                uniform over --radius pixels
  --kernel point               a single pixel: the reference behaviour

A kernel that half-covers a salient region therefore contributes half of its
mass, instead of the all-or-nothing "more than half the circle" rule, which
would add a second arbitrary threshold and make the score jump as the gaze
moves across it.

Degrees are converted to pixels with the measured screen geometry
(--screen_width_cm 52.0, --screen_height_cm 29.0, --viewing_distance_cm 60.0):
the screen subtends 46.86 degrees horizontally and 27.17 degrees vertically, so
one degree is 41.0 px horizontally and 39.7 px vertically in a 1920x1080 video
(23.0 and 26.5 in a 1080x720 one). The same sigma therefore means the same
physical region at any video resolution. --px_per_deg overrides all of it.

--fp_scope picks the false-positive pool. The default ("auto" -> "nonfixation")
leaves the fixated region out of it, as the reference implementation does and
as the benchmark does once the fixation map is blurred into a density; with a
single fixation that makes the score identical to the rank-based definition of
AUC. "all" instead ranks the gaze belief against every pixel of the map. The
pooled series always uses "all", because a density summed over observers covers
much of the frame and has no fixed support to exclude.

INTERPRETING THE RESULT
-----------------------
  0.5  = chance  (gaze belief sits no higher than a random pixel)
  1.0  = perfect (all belief on pixels more salient than every other pixel)
  <0.5 = the gaze belief sits on LESS salient pixels than average

A completely flat map (nothing detected in that frame) scores exactly 0.5.

With --kernel point and ONE fixation the score is the standard rank-based AUC
of a single positive pixel: 0.5 means the gaze landed on an average pixel, ~1
means it landed on one of the most salient pixels of the frame, and ~0 means it
landed on one of the least salient ones. The full [0, 1] range is used.

(The reference implementation sweeps only the fixation values, so the jump
across the tied pixels of an 8-bit map is treated as vertical; for a single
fixation that anchors FP at the value just below the gaze and compresses the
score into [0.5, 1]. Sweeping every level - the exact staircase, which is what
the rank-based definition gives - restores the full range, at the cost of
differing from the reference by half the mass of the tied pixels.)

TWO GRANULARITIES - BOTH ARE WRITTEN IN ONE RUN
-----------------------------------------------
  per participant  each frame uses only that participant's own gaze point
                   -> one CSV per gaze file, columns frame,x,y,auc_j
  pooled per frame each frame sums the gaze belief of ALL participants in the
                   batch into one fixation density map (the standard way video
                   saliency is scored, far less noisy than a single point)
                   -> one CSV for the task, columns frame,n_fixations,auc_j

The pooled numbers are only meaningful when the batch really contains all
participants of that task - --limit makes them partial by definition.

EXPECTED INPUT FORMAT
-----------------------
Gaze: a CSV/Excel file with a frame column and an x/y pair. These layouts are
detected automatically:
  frame, x, y                            generic / older exports
  videoFrame, gaze_x, gaze_y             resample_gaze.py output
Override with --frame_col / --x_col / --y_col when yours differ.

By default x/y are NORMALIZED to [0, 1] ((0,0) = top-left, (1,1) =
bottom-right) and converted with each saliency map's own width/height, so it
stays correct for any resolution. Use --coords pixel for absolute pixels.

Saliency maps: image files named with the frame number, e.g. frame_0001.png
(change with --pattern). This is what yolo_heatmap_video.py writes.

OUTPUT
------
  <output_dir>\\<group>\\<gaze file name>              per-frame AUC-J, one row
                                                       per scored frame
  <output_dir>\\<group>\\<group>_auc_summary.csv       one row per gaze file
  <output_dir>\\<group>\\<group>_auc_pooled_by_frame.csv   pooled AUC-J per frame
  <output_dir>\\<group>\\<group>_auc_pooled_summary.csv    pooled aggregate

"<group>" is the saliency folder name, or the task parsed from each gaze file
name when --saliency_root is used; --group overrides it.

USAGE
------
Single gaze file (writes only the per-frame CSV):

    python auc_score_new.py --gaze_file gaze.csv --saliency_dir saliency_frames\\myvideo \\
                            --output myvideo_auc.csv

Batch - all participants of one task:

    python auc_score_new.py --gaze_dir Resample_Output\\blockTowerR \\
                            --saliency_dir saliency_frames\\blockTowerR

Batch - several tasks at once:

    python auc_score_new.py --gaze_dir Resample_Output --recursive \\
                            --saliency_root saliency_frames

Reproduce the single-pixel model (the reference setup, and the same number the
rank-based AUC gives):

    ... --kernel point

Optional:
    --output_dir auc_Output      where results are written
    --group NAME                 force the output subfolder name
    --limit 3                    only the first 3 gaze files (quick check)
    --recursive                  also look in subfolders of --gaze_dir
    --kernel gaussian|disk|point gaze model (see above)
    --sigma_deg 1.0              Gaussian sigma in degrees of visual angle
    --radius PIXELS              disk kernel radius in pixels
    --fp_scope auto|all|nonfixation   false-positive pool
    --screen_width_cm / --screen_height_cm / --viewing_distance_cm / --px_per_deg   geometry
    --frame_col/--x_col/--y_col  override the auto-detected columns
    --pattern "frame_{frame:04d}.png"   saliency map file name pattern
    --coords normalized|pixel    how to read x and y
"""

import argparse
import os
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import gaze_kernel
from gaze_kernel import GazeKernel


# numpy 2.x renamed trapz to trapezoid; keep working on both
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Gaze tables use one of these column layouts; the first complete match wins
# unless the matching --frame_col / --x_col / --y_col are given explicitly.
COLUMN_PRESETS = (
    ("frame", "x", "y"),                    # generic / older exports
    ("videoFrame", "gaze_x", "gaze_y"),     # resample_gaze.py output
)

GAZE_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Names these scripts write themselves; never treat them as input gaze files.
RESULT_NAME_PATTERNS = ("*_nss_summary.csv", "*_auc_summary.csv",
                        "*_auc_pooled_by_frame.csv", "*_auc_pooled_summary.csv")


def load_gaze_data(gaze_file):
    gaze_file = str(gaze_file)
    if gaze_file.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(gaze_file)
    return pd.read_csv(gaze_file)


def resolve_columns(df, frame_col, x_col, y_col):
    """Work out which columns hold the frame number and the gaze position."""
    if frame_col and x_col and y_col:
        if all(c in df.columns for c in (frame_col, x_col, y_col)):
            return frame_col, x_col, y_col
        return None

    for preset in COLUMN_PRESETS:
        candidate = (
            frame_col or preset[0],
            x_col or preset[1],
            y_col or preset[2],
        )
        if all(c in df.columns for c in candidate):
            return candidate

    return None


def build_frame_entries(df, cols):
    """Group the gaze table into {frame: [(row, x, y), ...]}.

    The row number is kept so every output keeps the row order of its input
    file. Rows without a usable frame number or without both coordinates are
    counted and skipped.
    """
    frame_col, x_col, y_col = cols

    frames = pd.to_numeric(df[frame_col], errors="coerce")
    xs = pd.to_numeric(df[x_col], errors="coerce")
    ys = pd.to_numeric(df[y_col], errors="coerce")

    entries = defaultdict(list)
    skipped_bad_frame = 0
    skipped_nan = 0

    for row_number, (frame, x, y) in enumerate(zip(frames, xs, ys)):
        if pd.isna(frame):
            skipped_bad_frame += 1
            continue
        if pd.isna(x) or pd.isna(y):
            skipped_nan += 1
            continue
        entries[int(frame)].append((row_number, float(x), float(y)))

    return entries, skipped_bad_frame, skipped_nan


def to_pixel(x, y, width, height, coords):
    """Gaze position -> pixel index, or None when it falls outside the map."""
    if coords == "normalized":
        px, py = x * width, y * height
    else:
        px, py = x, y

    ix, iy = int(round(px)), int(round(py))

    if ix < 0 or ix >= width or iy < 0 or iy >= height:
        return None

    return ix, iy


class FrameLookup:
    """Counts of map pixels with a value >= v, for one saliency map.

    The map is 8-bit, so one histogram answers every threshold at once. Ties
    are therefore handled exactly, which is what the reference implementation
    approximates by adding rand/1e7 noise to the map.
    """

    def __init__(self, saliency_map):
        self.n_pixels = int(saliency_map.size)
        self.map = saliency_map
        counts = np.bincount(saliency_map.ravel(), minlength=256)
        self.count_suffix = np.cumsum(counts[::-1])[::-1].astype(np.float64)


def roc_auc_from_mass(mass_hist, count_suffix, n_pixels, support_counts=None, support_pixels=0):
    """AUC-Judd from the belief mass of the gaze kernel.

    mass_hist[v]     belief mass placed on map pixels whose saliency value is v
    count_suffix[v]  number of map pixels with a saliency value >= v
    support_counts   the kernel's own pixel histogram, when the false-positive
                     rate should be computed over non-kernel pixels only (the
                     convention of the reference implementation); None means
                     FP is taken over all map pixels

    The threshold is swept down over every 8-bit level, which makes this the
    exact staircase ROC of a quantised classifier - the same number the
    standard trapezoid AUC gives (sklearn.metrics.roc_auc_score), and the same
    convention for all three kernels:

        TP(t) = mass with value >= t / total mass
        FP(t) = pixels with value >= t (outside the kernel, if asked) / count

    TP and FP are both non-decreasing as t drops, so the ROC is monotone even
    when the saliency map has large flat regions. The reference implementation
    instead sweeps only the fixation values and subtracts each fixation's
    *rank*, which both ignores the levels in between (slightly overestimating
    the area, by half the mass of the tied pixels) and makes FP move backwards
    inside a group of tied values.
    """
    total_mass = float(mass_hist.sum())
    if total_mass <= 0:
        return None

    mass_above = np.cumsum(mass_hist[::-1])[::-1]
    tp = mass_above[::-1] / total_mass                  # high threshold -> low

    if support_counts is None:
        fp = count_suffix[::-1] / n_pixels
    else:
        denominator = n_pixels - support_pixels
        if denominator <= 0:
            return None
        support_above = np.cumsum(support_counts[::-1])[::-1]
        fp = (count_suffix[::-1] - support_above[::-1]) / denominator

    tp_curve = np.concatenate(([0.0], tp, [1.0]))
    fp_curve = np.concatenate(([0.0], fp, [1.0]))

    return float(_trapz(tp_curve, fp_curve))


def task_name_from_file(path):
    """'1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv' -> 'bananaQantH'."""
    stem = Path(path).stem
    for suffix in (".mp4_gaze", "_gaze"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def collect_gaze_files(args):
    """All gaze tables the run should score, in a stable order."""
    if args.gaze_file:
        return [Path(args.gaze_file)]

    root = Path(args.gaze_dir)
    pattern = "**/*" if args.recursive else "*"
    files = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in GAZE_EXTENSIONS
        and not any(fnmatch(path.name, result) for result in RESULT_NAME_PATTERNS)
    ]

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    return files


def saliency_dir_for(path, args):
    """Which map folder belongs to this gaze file, and the output group name."""
    if args.saliency_root:
        task = task_name_from_file(path)
        folder = os.path.join(args.saliency_root, task)
        if not os.path.isdir(folder):
            return None, task, f"no saliency folder '{task}' under {args.saliency_root}"
        return folder, task, None

    group = args.group or os.path.basename(os.path.normpath(args.saliency_dir))
    return args.saliency_dir, group, None


def main():
    parser = argparse.ArgumentParser(
        description="AUC-Judd for one gaze file or a whole folder of them."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gaze_file", help="Single CSV/Excel file with gaze data")
    source.add_argument("--gaze_dir", help="Folder with gaze files to process in one run")

    maps = parser.add_mutually_exclusive_group(required=True)
    maps.add_argument("--saliency_dir", help="Folder with the per-frame saliency maps")
    maps.add_argument("--saliency_root",
                      help="Folder with one saliency map subfolder per task; the "
                           "task is read from each gaze file name")

    parser.add_argument("--recursive", action="store_true",
                        help="With --gaze_dir: also look in subfolders")
    parser.add_argument("--limit", type=int, default=0,
                        help="With --gaze_dir: stop after this many gaze files (0 = all)")
    parser.add_argument("--output_dir", default="auc_Output",
                        help="Where per-file results and summaries are written")
    parser.add_argument("--group", default=None,
                        help="Force the output subfolder name (default: task name)")
    parser.add_argument("--output", default=None,
                        help="Single file only: exact output path for the per-frame CSV")

    parser.add_argument("--frame_col", default=None)
    parser.add_argument("--x_col", default=None)
    parser.add_argument("--y_col", default=None)
    parser.add_argument("--pattern", default="frame_{frame:04d}.png",
                        help="Filename pattern for saliency maps, using {frame} as placeholder")
    parser.add_argument("--coords", choices=["normalized", "pixel"], default="normalized",
                        help="'normalized' (default): x,y in [0,1], (0,0)=top-left, (1,1)=bottom-right. "
                             "'pixel': x,y are already absolute pixel coordinates.")

    gaze = parser.add_argument_group(
        "gaze model",
        "How much of the saliency map a recorded gaze point stands for. The eye "
        "tracker is accurate to roughly 2 degrees of visual angle, so the default "
        "spreads the gaze over a Gaussian of --sigma_deg instead of a single pixel.")
    gaze.add_argument("--kernel", choices=list(gaze_kernel.KERNELS), default="gaussian",
                      help="gaussian (default), disk (uniform, --radius in pixels), "
                           "or point (single pixel, the reference behaviour)")
    gaze.add_argument("--sigma_deg", type=float, default=1.0,
                      help="Gaussian sigma in degrees of visual angle (default 1.0; "
                           "86.5%% of the mass then falls inside 2 degrees)")
    gaze.add_argument("--radius", type=int, default=20,
                      help="With --kernel disk: radius in pixels (default 20)")
    gaze.add_argument("--truncate_sigma", type=float, default=3.0,
                      help="With --kernel gaussian: cut the kernel at this many "
                           "sigmas (default 3.0)")
    gaze.add_argument("--screen_width_cm", type=float, default=gaze_kernel.DEFAULT_SCREEN_WIDTH_CM,
                      help="Physical screen width, for degrees -> pixels "
                           "(default %.1f, the measured setup)" % gaze_kernel.DEFAULT_SCREEN_WIDTH_CM)
    gaze.add_argument("--screen_height_cm", type=float, default=gaze_kernel.DEFAULT_SCREEN_HEIGHT_CM,
                      help="Physical screen height, for degrees -> pixels "
                           "(default %.1f, the measured setup)" % gaze_kernel.DEFAULT_SCREEN_HEIGHT_CM)
    gaze.add_argument("--viewing_distance_cm", type=float,
                      default=gaze_kernel.DEFAULT_VIEWING_DISTANCE_CM,
                      help="Eye-to-screen distance, for degrees -> pixels (default %.1f)"
                           % gaze_kernel.DEFAULT_VIEWING_DISTANCE_CM)
    gaze.add_argument("--px_per_deg", type=float, default=None,
                      help="Override the computed pixels-per-degree (skips the geometry)")
    gaze.add_argument("--fp_scope", choices=["auto", "all", "nonfixation"], default="auto",
                      help="Which pixels form the false-positive pool: 'nonfixation' "
                           "(outside the gaze kernel - the reference convention, and "
                           "what makes a single fixation identical to the rank-based "
                           "AUC), 'all' (every map pixel), or 'auto' (nonfixation; the "
                           "pooled series always uses 'all', since a density summed "
                           "over observers has no fixed support to exclude)")

    args = parser.parse_args()

    if args.gaze_dir and not os.path.isdir(args.gaze_dir):
        print(f"ERROR: gaze folder does not exist: {args.gaze_dir}")
        sys.exit(1)
    if args.gaze_file and not os.path.isfile(args.gaze_file):
        print(f"ERROR: gaze file does not exist: {args.gaze_file}")
        sys.exit(1)
    if args.saliency_dir and not os.path.isdir(args.saliency_dir):
        print(f"ERROR: saliency folder does not exist: {args.saliency_dir}")
        sys.exit(1)
    if args.saliency_root and not os.path.isdir(args.saliency_root):
        print(f"ERROR: saliency root does not exist: {args.saliency_root}")
        sys.exit(1)
    if args.output and args.gaze_dir:
        print("ERROR: --output writes a single file; use --output_dir with --gaze_dir")
        sys.exit(1)

    if args.fp_scope == "auto":
        # The reference implementation, and the benchmark practice of blurring
        # the fixation map and treating its non-zero pixels as fixated, both
        # exclude the fixated region from the false-positive pool. With one
        # fixation this makes the score identical to the rank-based definition
        # of AUC (sklearn.metrics.roc_auc_score).
        fp_scope = "nonfixation"
    else:
        fp_scope = args.fp_scope

    gaze_files = collect_gaze_files(args)
    if not gaze_files:
        print(f"ERROR: no gaze files found in {args.gaze_dir}")
        sys.exit(1)

    if args.gaze_dir and os.path.abspath(args.gaze_dir) == os.path.abspath(args.output_dir):
        print("WARNING: --gaze_dir and --output_dir are the same folder, so the "
              "per-frame results written there look exactly like inputs. Use a "
              "different --output_dir, or --group NAME, to keep them apart.")
        print()

    print("================================")
    print("AUC-JUDD SCORING")
    print("================================")
    print(f"Gaze files : {len(gaze_files)}")
    print(f"Saliency   : {args.saliency_dir or (args.saliency_root + ' (per task)')}")
    print(f"Coords     : {args.coords}")
    print(f"Gaze model : {args.kernel}")
    print(f"Screen     : {args.screen_width_cm:.1f} x {args.screen_height_cm:.1f} cm, "
          f"{args.viewing_distance_cm:.1f} cm viewing distance"
          + (f" (px/deg forced to {args.px_per_deg:.1f})" if args.px_per_deg else ""))
    print(f"FP pool    : {'all map pixels' if fp_scope == 'all' else 'non-kernel pixels'}"
          f"{' (auto)' if args.fp_scope == 'auto' else ''}")
    print()

    # The kernel depends on the frame size, so build one per size and reuse it.
    kernel_cache = {}

    def kernel_for(shape):
        key = shape
        kernel = kernel_cache.get(key)
        if kernel is None:
            kernel = GazeKernel(
                shape[1], shape[0],
                kind=args.kernel,
                sigma_deg=args.sigma_deg,
                radius_px=args.radius,
                screen_width_cm=args.screen_width_cm,
                screen_height_cm=args.screen_height_cm,
                viewing_distance_cm=args.viewing_distance_cm,
                truncate_sigma=args.truncate_sigma,
                px_per_deg=args.px_per_deg,
            )
            kernel_cache[key] = kernel
            print(f"  gaze kernel for {shape[1]}x{shape[0]} frames: {kernel.describe()}")
        return kernel

    # ----------------------------------------------------------------
    # 1. Read every gaze table and group its rows by frame
    # ----------------------------------------------------------------

    states = []

    for path in gaze_files:
        state = {
            "path": str(path),
            "name": path.name,
            "sal_dir": None,
            "group": None,
            "entries": {},
            "results": {},
            "rows_scored": 0,
            "skipped_missing_file": 0,
            "skipped_out_of_bounds": 0,
            "skipped_nan": 0,
            "skipped_bad_frame": 0,
            "error": None,
        }
        states.append(state)

        sal_dir, group, error = saliency_dir_for(path, args)
        state["group"] = group
        if error:
            state["error"] = error
            continue
        state["sal_dir"] = sal_dir

        try:
            df = load_gaze_data(path)
        except Exception as exc:
            state["error"] = f"could not read the file: {exc}"
            continue

        cols = resolve_columns(df, args.frame_col, args.x_col, args.y_col)
        if cols is None:
            state["error"] = (
                "no frame/x/y columns (looked for "
                + " | ".join(",".join(p) for p in COLUMN_PRESETS)
                + f"; columns found: {list(df.columns)})"
            )
            continue

        entries, skipped_bad_frame, skipped_nan = build_frame_entries(df, cols)
        state["entries"] = entries
        state["skipped_bad_frame"] = skipped_bad_frame
        state["skipped_nan"] = skipped_nan

    if not any(s["sal_dir"] for s in states):
        print("Nothing to score:")
        for state in states:
            print(f"  {state['name']}: {state['error']}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Score frame by frame, one pass per saliency folder
    #    Each map image is read once and reused for every gaze file,
    #    and the per-frame histogram is built once for all of them.
    # ----------------------------------------------------------------

    buckets = defaultdict(lambda: defaultdict(list))
    for index, state in enumerate(states):
        if not state["sal_dir"]:
            continue
        for frame, rows in state["entries"].items():
            for row_number, x, y in rows:
                buckets[state["sal_dir"]][frame].append((index, row_number, x, y))

    pooled = {}          # sal_dir -> list of (frame, n_fixations, auc)

    for sal_dir in sorted(buckets):
        frames = buckets[sal_dir]
        pooled[sal_dir] = []
        names = set(os.listdir(sal_dir)) if os.sep not in args.pattern else None
        total = len(frames)

        print(f"Saliency folder: {sal_dir}")
        print(f"  frames to score: {total}")

        for done, frame in enumerate(sorted(frames), start=1):
            filename = args.pattern.format(frame=frame)
            filepath = os.path.join(sal_dir, filename)

            found = (filename in names) if names is not None else os.path.exists(filepath)
            saliency_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) if found else None

            if saliency_map is None:
                for index, _, _, _ in frames[frame]:
                    states[index]["skipped_missing_file"] += 1
                continue

            height, width = saliency_map.shape
            lookup = FrameLookup(saliency_map)
            kernel = kernel_for(saliency_map.shape)

            # Sum of the participants' belief mass per saliency value = the
            # fixation density map of the whole group for this frame.
            pooled_mass = np.zeros(256)
            pooled_pixels = 0

            for index, row_number, x, y in frames[frame]:
                pixel = to_pixel(x, y, width, height, args.coords)
                if pixel is None:
                    states[index]["skipped_out_of_bounds"] += 1
                    continue

                want_counts = (fp_scope == "nonfixation")
                mass, support_counts = kernel.weighted_histograms(
                    saliency_map, pixel[0], pixel[1], with_counts=want_counts)

                score = roc_auc_from_mass(
                    mass, lookup.count_suffix, lookup.n_pixels,
                    support_counts=support_counts,
                    support_pixels=int(support_counts.sum()) if support_counts is not None else 0,
                )
                if score is None:
                    states[index]["skipped_out_of_bounds"] += 1
                    continue

                states[index]["results"][row_number] = (frame, x, y, score)
                states[index]["rows_scored"] += 1

                pooled_mass += mass
                pooled_pixels += 1

            # The pooled series always scores FP over all map pixels: a summed
            # density has no fixed support to exclude.
            if pooled_pixels:
                pooled_score = roc_auc_from_mass(
                    pooled_mass, lookup.count_suffix, lookup.n_pixels)
                if pooled_score is not None:
                    pooled[sal_dir].append((frame, pooled_pixels, pooled_score))

            if done % 500 == 0 or done == total:
                print(f"    {done}/{total} frames scored")

        print()

    # ----------------------------------------------------------------
    # 3. Write the results
    # ----------------------------------------------------------------

    def frame_rows(state):
        return [
            {"frame": frame, "x": x, "y": y, "auc_j": score}
            for row_number, (frame, x, y, score) in sorted(state["results"].items())
        ]

    # Single file with an explicit destination: only the per-frame CSV.
    if args.output:
        state = states[0]
        if not state["results"]:
            print(f"No AUC-J could be computed for {state['name']}.")
            if state["error"]:
                print(f"  {state['error']}")
            else:
                print(f"  {state['skipped_missing_file']} rows had no saliency map file")
            sys.exit(1)

        results_df = pd.DataFrame(frame_rows(state), columns=["frame", "x", "y", "auc_j"])
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        results_df.to_csv(args.output, index=False)

        print(f"\nDone. Computed AUC-J for {len(results_df)} frames.")
        if state["skipped_missing_file"]:
            print(f"  Skipped {state['skipped_missing_file']} rows (saliency map file not found)")
        if state["skipped_out_of_bounds"]:
            print(f"  Skipped {state['skipped_out_of_bounds']} rows (gaze point outside frame bounds)")
        if state["skipped_nan"]:
            print(f"  Skipped {state['skipped_nan']} rows (missing/NaN x or y value)")
        if state["skipped_bad_frame"]:
            print(f"  Skipped {state['skipped_bad_frame']} rows (missing/NaN frame number)")

        print(f"\nMean AUC-J: {results_df['auc_j'].mean():.4f}")
        print(f"Std AUC-J:  {results_df['auc_j'].std():.4f}")
        print(f"Min/Max:    {results_df['auc_j'].min():.4f} / {results_df['auc_j'].max():.4f}")
        print(f"\nPer-frame results saved to: {args.output}")
        return

    summaries = defaultdict(list)
    files_with_scores = 0

    for state in states:
        if not state["results"]:
            continue

        results_df = pd.DataFrame(frame_rows(state), columns=["frame", "x", "y", "auc_j"])

        group_dir = os.path.join(args.output_dir, state["group"]) if state["group"] else args.output_dir
        os.makedirs(group_dir, exist_ok=True)
        results_df.to_csv(os.path.join(group_dir, state["name"]), index=False)
        files_with_scores += 1

        summaries[state["group"] or ""].append({
            "gaze_file": state["name"],
            "task": state["group"] or "",
            "rows_in_file": len(state["results"]) + state["skipped_missing_file"]
                            + state["skipped_out_of_bounds"] + state["skipped_nan"]
                            + state["skipped_bad_frame"],
            "rows_scored": len(results_df),
            "mean_auc_j": results_df["auc_j"].mean(),
            "std_auc_j": results_df["auc_j"].std(),
            "min_auc_j": results_df["auc_j"].min(),
            "max_auc_j": results_df["auc_j"].max(),
            "skipped_no_map": state["skipped_missing_file"],
            "skipped_out_of_bounds": state["skipped_out_of_bounds"],
            "skipped_missing_xy": state["skipped_nan"],
            "skipped_bad_frame": state["skipped_bad_frame"],
            "status": "ok",
        })

    summary_paths = {}
    pooled_paths = {}

    # which saliency folders feed which output group
    sal_dirs_per_group = defaultdict(set)
    for state in states:
        if state["sal_dir"]:
            sal_dirs_per_group[state["group"] or ""].add(state["sal_dir"])

    for group, rows in summaries.items():
        group_dir = os.path.join(args.output_dir, group) if group else args.output_dir
        os.makedirs(group_dir, exist_ok=True)
        label = group or "auc"

        summary_path = os.path.join(group_dir, f"{label}_auc_summary.csv")
        pd.DataFrame(rows).sort_values("gaze_file").to_csv(summary_path, index=False)
        summary_paths[group] = summary_path

        sal_dirs = sorted(sal_dirs_per_group.get(group, ()))

        for sal_dir in sal_dirs:
            entries = pooled.get(sal_dir) or []
            if not entries:
                continue

            # several tasks sharing one output group must not be merged frame by
            # frame, so each saliency folder gets its own series in that case
            series_label = label if len(sal_dirs) == 1 else \
                f"{label}_{os.path.basename(os.path.normpath(sal_dir))}"

            by_frame = pd.DataFrame(
                sorted(entries, key=lambda item: item[0]),
                columns=["frame", "n_fixations", "auc_j"],
            )
            by_frame_path = os.path.join(group_dir, f"{series_label}_auc_pooled_by_frame.csv")
            by_frame.to_csv(by_frame_path, index=False)

            aggregate = pd.DataFrame([{
                "task": group or "",
                "saliency_dir": sal_dir,
                "n_frames": len(by_frame),
                "mean_auc_j": by_frame["auc_j"].mean(),
                "std_auc_j": by_frame["auc_j"].std(),
                "min_auc_j": by_frame["auc_j"].min(),
                "max_auc_j": by_frame["auc_j"].max(),
                "mean_n_fixations": by_frame["n_fixations"].mean(),
            }])
            aggregate_path = os.path.join(group_dir, f"{series_label}_auc_pooled_summary.csv")
            aggregate.to_csv(aggregate_path, index=False)

            pooled_paths.setdefault(group, []).append((by_frame_path, aggregate_path, by_frame))

    # ----------------------------------------------------------------
    # 4. Report
    # ----------------------------------------------------------------

    print("================================")
    print("RESULTS")
    print("================================")

    for group, rows in summaries.items():
        label = group or "(no group)"
        per_file_means = [r["mean_auc_j"] for r in rows]
        scored = sum(r["rows_scored"] for r in rows)
        print(f"\n{label}: {len(rows)} files scored, {scored} gaze frames (per participant)")
        print(f"  mean of per-file mean AUC-J: {np.mean(per_file_means):.4f}")
        print(f"  spread of per-file means   : "
              f"{np.min(per_file_means):.4f} .. {np.max(per_file_means):.4f}")
        print(f"  saved to: {os.path.join(args.output_dir, group) if group else args.output_dir}")
        print(f"  summary : {summary_paths.get(group)}")

        if group in pooled_paths:
            for by_frame_path, aggregate_path, by_frame in pooled_paths[group]:
                print(f"  pooled AUC-J (all participants per frame, "
                      f"{len(by_frame)} frames, mean {by_frame['n_fixations'].mean():.1f} fixations): "
                      f"{by_frame['auc_j'].mean():.4f} "
                      f"(std {by_frame['auc_j'].std():.4f})")
                print(f"  pooled by frame  : {by_frame_path}")
                print(f"  pooled aggregate : {aggregate_path}")

    failed = [s for s in states if s["error"]]
    if failed:
        print(f"\nSkipped {len(failed)} file(s):")
        for state in failed:
            print(f"  {state['name']}: {state['error']}")

    no_scores = [s for s in states if not s["error"] and not s["results"]]
    if no_scores:
        print(f"\nNo AUC-J could be computed for {len(no_scores)} file(s) "
              f"(no matching saliency map for their frames):")
        for state in no_scores[:10]:
            print(f"  {state['name']}: {state['skipped_missing_file']} frames without a map")
        if len(no_scores) > 10:
            print(f"  ... and {len(no_scores) - 10} more")

    total_oob = sum(s["skipped_out_of_bounds"] for s in states)
    total_nan = sum(s["skipped_nan"] for s in states)
    if total_oob:
        print(f"\nSkipped {total_oob} rows in total (gaze point outside frame bounds)")
    if total_nan:
        print(f"Skipped {total_nan} rows in total (missing/NaN x or y value)")

    print()
    if files_with_scores == 0:
        print("No AUC-J scores could be computed. Check your file paths and column names.")
        sys.exit(1)

    print(f"Done. {files_with_scores} per-frame CSV file(s) written, "
          f"{len(gaze_files)} gaze file(s) considered.")


if __name__ == "__main__":
    main()
