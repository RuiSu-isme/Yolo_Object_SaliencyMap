"""
Resample raw eye-tracking samples onto the video frame grid.

=====================================================================
STEP 1 OF 4 - RUN THIS FIRST
=====================================================================
THE WHOLE PIPELINE, IN ORDER

    Step 1  resample_gaze.py       raw tracker export -> one row per video frame
    Step 2  yolo_heatmap_video.py  stimulus video     -> grayscale saliency maps
    Step 3  nss_score_new.py       maps x gaze        -> per-frame NSS
    Step 4  auc_score_new.py       maps x gaze        -> per-frame AUC-Judd

FOLDERS THIS PIPELINE USES (names are examples, change them freely)

    gazedata\\<name>.csv                       raw exports, one row per sample
    video\\<task>.mp4                          the stimulus videos
    Resample_Output\\<task>\\<name>.csv        written by step 1
    saliency_frames\\<task>\\frame_0001.png    written by step 2
    output\\<task>.mp4                         written by step 2 (heatmap video)
    nss_Output\\<task>\\...                    written by step 3
    auc_Output\\<task>\\...                    written by step 4

WHAT THIS STEP DOES, AND WHY THE FRAME RATE MATTERS

The tracker exports one row per sample (about 120 Hz). Steps 3 and 4 need one
row per VIDEO FRAME instead, so that row `videoFrame` k lines up with the
saliency map `frame_000k.png`. This script builds that grid and keeps, for each
video frame time, the nearest valid gaze sample.

The grid must be built at the frame rate of the STIMULUS VIDEO. If it is not,
frame k of the output no longer corresponds to video frame k and every score
downstream compares a fixation with the wrong picture.

COMMANDS

    # normal case: read the rate from video\\<task>.mp4 for every file
    python resample_gaze.py gazedata Resample_Output

    # one task only, sorted into its own subfolder
    python resample_gaze.py gazedata Resample_Output --only "*srt7Aud*" --by_task

    # force one rate for every file of this run (e.g. 29.97 counted as 30)
    python resample_gaze.py gazedata Resample_Output --fps 30

    # see what would happen, writing nothing
    python resample_gaze.py gazedata Resample_Output --dry_run

OUTPUT

    without --by_task : <output_folder>\\<same file name>.csv      one flat folder
    with    --by_task : <output_folder>\\<task>\\<same file name>.csv

    columns: videoFrame, videoTime, gaze_x, gaze_y, calib_winner, gaze_missing

    The task (video) name is read from the gaze file name, so --by_task is how
    you get one folder per video. Run --dry_run first: it prints the task, the
    frame rate and the exact destination path of every file without writing
    anything, and ends with a per-task count. A normal run prints the same
    per-task count at the end.

CHECKS AFTER RUNNING

    - the row count should be close to the number of saliency maps of that task
      (a few frames fewer is normal if the recording is shorter than the video)
    - the script warns by itself when the frames it wrote disagree with the
      frame count of the stimulus video
    - "ERROR ... could not read the frame rate" means the stimulus video was
      not found: pass --video_dir, or the rate with --fps N
=====================================================================

For every video frame time it keeps the nearest valid gaze sample (within
MAX_TIME_DIFF). That is exactly what nss_score_new.py / auc_score_new.py
expect: one row per video frame, so `videoFrame` k lines up with the saliency
map `frame_000k.png`.

FRAME RATE
----------
The timeline must be built at the frame rate of the STIMULUS VIDEO, not at a
fixed 60 Hz. With the wrong rate the resampled frame k no longer corresponds to
video frame k, so every downstream metric would compare a fixation against the
wrong picture (and any frames beyond the end of the video have no map at all).

--fps therefore defaults to "auto": the rate is read from
<video_dir>/<task>.mp4, where <task> is taken from the gaze file name
(1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv -> bananaQantH). A rate within 1%
of an integer is rounded to it, so 29.97 is used as 30 and 23.976 as 24; the
resulting drift is a fraction of one frame, far below the 1/120 s sampling
tolerance. Pass --fps 60 to restore the old fixed behaviour for a 60 fps batch.

Every file is checked after writing: the number of rows produced is compared
with the frame count of the stimulus video, and a warning is printed when they
disagree.

USAGE
-----
    python resample_gaze.py "input_folder" "output_folder" [options]

    # one task, at the rate of its own video
    python resample_gaze.py gazedata\\srt14Aud Resample_Output\\srt14Aud

    # only the 25/30 fps tasks, mirroring the per-task layout, nothing written
    python resample_gaze.py gazedata Resample_Output --only "*srt*Aud*" --by_task --dry_run

OPTIONS
-------
    --fps auto|N      frame rate of the stimulus video (default: auto)
    --video_dir DIR   folder holding <task>.mp4 (default: video)
    --only PATTERN    only files whose name matches (repeatable)
    --by_task         write each result into <output_folder>/<task>/
    --dry_run         print the frame rate each file would use; write nothing
"""

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Used only with --fps 60 / when a rate cannot be read from a video.
DEFAULT_VIDEO_FPS = 60

# Maximum time difference between a video frame and
# the nearest valid eye-tracking sample.
MAX_TIME_DIFF = 1 / 120

# A rate this close (relatively) to an integer is treated as that integer,
# so 29.97 -> 30, 23.976 -> 24, 59.94 -> 60.
FPS_SNAP_TOLERANCE = 0.01


# ============================================================
# HELPERS
# ============================================================

def task_name_from_file(path):
    """'1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv' -> 'bananaQantH'."""
    stem = Path(path).stem
    for suffix in (".mp4_gaze", "_gaze"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def snap_fps(fps):
    """Round a measured frame rate to the nearest integer when it is close."""
    snapped = round(fps)
    if snapped > 0 and abs(fps - snapped) / snapped <= FPS_SNAP_TOLERANCE:
        return float(snapped), True
    return float(fps), False


def read_video_info(video_path):
    """(fps, frame_count) of a video, or (None, None) when it cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not fps or fps <= 0:
        return None, None
    return float(fps), frames


def resolve_fps(fps_arg, input_file, video_dir):
    """Frame rate to use for one gaze file.

    Returns (fps, source, video_frames, task). `source` is a short label for
    the printout; video_frames is None when no video was read.
    """
    if fps_arg != "auto":
        return float(fps_arg), f"--fps {fps_arg}", None, task_name_from_file(input_file)

    task = task_name_from_file(input_file)
    video_path = video_dir / f"{task}.mp4"
    measured, frames = read_video_info(video_path)

    if measured is None:
        raise RuntimeError(
            f"could not read the frame rate of the stimulus video for this file: "
            f"{video_path} (task '{task}'). Pass --fps N to set it explicitly, "
            f"or --video_dir DIR if the videos live elsewhere.")

    fps, snapped = snap_fps(measured)
    label = f"{video_path.name}: {measured:.2f} fps"
    if snapped and abs(measured - fps) > 0.005:
        label += f" -> using {fps:.0f}"
    return fps, label, frames, task


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Resample raw gaze samples onto the video frame grid, at the "
                "frame rate of the stimulus video by default.")
parser.add_argument("input_folder")
parser.add_argument("output_folder")
parser.add_argument("--fps", default="auto",
                    help="'auto' (default) reads the rate from the stimulus video, "
                         "otherwise a number, e.g. --fps 30")
parser.add_argument("--video_dir", default="video",
                    help="Folder holding <task>.mp4 for --fps auto (default: video)")
parser.add_argument("--only", action="append", default=[], metavar="PATTERN",
                    help="Only process files whose name matches this pattern "
                         "(repeatable), e.g. --only \"*srt14Aud*\"")
parser.add_argument("--by_task", action="store_true",
                    help="Write every result into <output_folder>/<task>/ instead "
                         "of straight into <output_folder>/")
parser.add_argument("--dry_run", action="store_true",
                    help="Print the frame rate each file would use and exit")
args = parser.parse_args()

INPUT_FOLDER = Path(args.input_folder)
OUTPUT_FOLDER = Path(args.output_folder)
VIDEO_DIR = Path(args.video_dir)


# ============================================================
# CHECK INPUT / CREATE OUTPUT FOLDER
# ============================================================

if not INPUT_FOLDER.exists():
    print("ERROR: Input folder does not exist:")
    print(INPUT_FOLDER)
    sys.exit(1)

if not INPUT_FOLDER.is_dir():
    print("ERROR: Input path is not a folder:")
    print(INPUT_FOLDER)
    sys.exit(1)

if not args.dry_run:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND ALL CSV FILES
# ============================================================

csv_files = sorted(INPUT_FOLDER.glob("*.csv"))

if args.only:
    csv_files = [path for path in csv_files
                 if any(fnmatch(path.name, pattern) for pattern in args.only)]

if len(csv_files) == 0:
    print("No CSV files found in:")
    print(INPUT_FOLDER)
    if args.only:
        print("(after applying --only:", ", ".join(args.only), ")")
    sys.exit(1)

print("================================")
print("BATCH GAZE RESAMPLING")
print("================================")
print("Input folder:")
print(INPUT_FOLDER)
print()
print("Output folder:")
print(OUTPUT_FOLDER)
print()
print("Frame rate:")
print("  " + (args.fps if args.fps == "auto"
               else f"{args.fps} (fixed, --fps)"))
if args.fps == "auto":
    print("  video folder:", VIDEO_DIR)
print()
print("CSV files found:", len(csv_files))
print()


# ============================================================
# DRY RUN: WHICH FRAME RATE WOULD EACH FILE USE?
# ============================================================

if args.dry_run:
    print("================================")
    print("DRY RUN (nothing is written)")
    print("================================")
    planned = {}
    for input_file in csv_files:
        try:
            fps, source, frames, task = resolve_fps(args.fps, input_file, VIDEO_DIR)
            file_output_dir = OUTPUT_FOLDER / task if args.by_task else OUTPUT_FOLDER
            planned[task] = planned.get(task, 0) + 1
            print(f"  {input_file.name}")
            print(f"      task '{task}' -> {source}")
            if frames:
                print(f"      stimulus video frames: {frames} "
                      f"(about {frames / fps:.1f} s at {fps:g} fps)")
            print(f"      would write: {file_output_dir / input_file.name}")
        except Exception as exc:
            print(f"  {input_file.name}")
            print(f"      ERROR: {exc}")

    print()
    print("Files per task:")
    for task in sorted(planned):
        destination = OUTPUT_FOLDER / task if args.by_task else OUTPUT_FOLDER
        print(f"  {task:<28} {planned[task]:>4} file(s)  ->  {destination}")

    sys.exit(0)


# ============================================================
# PROCESS EACH FILE
# ============================================================

successful = 0
failed = 0
warnings = []
written_per_task = {}

for file_number, input_file in enumerate(csv_files, start=1):

    print("--------------------------------")
    print(f"[{file_number}/{len(csv_files)}] Processing:")
    print(input_file.name)

    try:

        # ====================================================
        # 0. FRAME RATE OF THIS FILE'S STIMULUS VIDEO
        # ====================================================

        fps, fps_source, video_frames, task = resolve_fps(args.fps, input_file, VIDEO_DIR)
        print("Frame rate:", f"{fps:g}", f"({fps_source})")


        # ====================================================
        # 1. LOAD DATA
        # ====================================================

        df = pd.read_csv(input_file)

        print("Original rows:", len(df))


        # ====================================================
        # 2. VIDEO TIME
        # ====================================================

        df["videoTime"] = pd.to_numeric(
            df["videoTime"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["videoTime"]
        ).copy()


        # ====================================================
        # 3. GAZE COLUMNS
        # ====================================================

        for column in [
            "eyeXmf",
            "eyeYmf",
            "nudged_eyeX",
            "nudged_eyeY"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        # ====================================================
        # 4. SELECT EFFECTIVE GAZE
        # ====================================================

        calib = (
            df["calib_winner"]
            .astype(str)
            .str.lower()
        )

        df["gaze_x"] = np.where(
            calib == "raw",
            df["eyeXmf"],
            np.where(
                calib == "nudged",
                df["nudged_eyeX"],
                np.nan
            )
        )

        df["gaze_y"] = np.where(
            calib == "raw",
            df["eyeYmf"],
            np.where(
                calib == "nudged",
                df["nudged_eyeY"],
                np.nan
            )
        )


        # ====================================================
        # 5. SORT BY TIME
        # ====================================================

        df = df.sort_values(
            "videoTime"
        ).reset_index(drop=True)


        # ====================================================
        # 6. KEEP VALID GAZE
        # ====================================================

        valid_gaze = df.dropna(
            subset=["gaze_x", "gaze_y"]
        ).copy()

        if len(valid_gaze) == 0:
            raise ValueError(
                "No valid gaze observations found."
            )

        print("Valid gaze observations:", len(valid_gaze))


        # ====================================================
        # 7. CREATE THE VIDEO TIMELINE AT THE VIDEO'S OWN RATE
        # ====================================================

        video_end = df["videoTime"].max()

        frame_interval = 1 / fps

        n_frames = (
            int(np.floor(
                video_end / frame_interval
            )) + 1
        )

        video_times = (
            np.arange(n_frames)
            * frame_interval
        )


        # ====================================================
        # 8. CREATE VIDEO FRAME DATAFRAME
        # ====================================================

        video_df = pd.DataFrame({

            "videoFrame":
                np.arange(
                    1,
                    n_frames + 1
                ),

            "videoTime":
                video_times
        })


        # ====================================================
        # 9. MATCH NEAREST GAZE SAMPLE
        # ====================================================

        valid_gaze = valid_gaze[
            [
                "videoTime",
                "gaze_x",
                "gaze_y",
                "calib_winner"
            ]
        ].sort_values(
            "videoTime"
        )

        matched = pd.merge_asof(
            video_df,
            valid_gaze,
            on="videoTime",
            direction="nearest",
            tolerance=MAX_TIME_DIFF
        )


        # ====================================================
        # 10. IDENTIFY MISSING GAZE
        # ====================================================

        matched["gaze_missing"] = (
            matched["gaze_x"].isna()
            |
            matched["gaze_y"].isna()
        )

        missing_frames = (
            matched["gaze_missing"].sum()
        )

        total_frames = len(matched)

        missing_percentage = (
            missing_frames
            / total_frames
            * 100
        )


        # ====================================================
        # 11. SAVE WITH ORIGINAL FILENAME
        # ====================================================

        output_dir = OUTPUT_FOLDER / task if args.by_task else OUTPUT_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_dir
            / input_file.name
        )

        matched.to_csv(
            output_file,
            index=False
        )


        # ====================================================
        # 12. CHECK AGAINST THE STIMULUS VIDEO
        # ====================================================

        if video_frames:
            difference = total_frames - video_frames
            if abs(difference) > 2:
                message = (f"{input_file.name}: wrote {total_frames} frames but "
                           f"{VIDEO_DIR / (task + '.mp4')} has {video_frames} "
                           f"({difference:+d}) - the frame rate or the video may "
                           f"not match this recording")
                warnings.append(message)
                print("WARNING:", message)


        # ====================================================
        # 13. PRINT SUMMARY
        # ====================================================

        print(
            "Video frames:",
            total_frames
        )

        print(
            "Missing gaze:",
            missing_frames,
            f"({missing_percentage:.2f}%)"
        )

        print("Saved to:")
        print(output_file)

        successful += 1
        written_per_task[task] = written_per_task.get(task, 0) + 1


    except Exception as e:

        failed += 1

        print("ERROR processing this file:")
        print(e)

        print("Skipping this file and continuing...")
        print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("================================")
print("BATCH PROCESSING COMPLETE")
print("================================")

print("Total files:", len(csv_files))
print("Successfully processed:", successful)
print("Failed:", failed)

if written_per_task:
    print()
    print("Files written per task:")
    for task in sorted(written_per_task):
        destination = OUTPUT_FOLDER / task if args.by_task else OUTPUT_FOLDER
        print(f"  {task:<28} {written_per_task[task]:>4}  ->  {destination}")

if warnings:
    print()
    print(f"Warnings ({len(warnings)}):")
    for message in warnings:
        print("  -", message)

print()
print("Output folder:")
print(OUTPUT_FOLDER)
if not args.by_task:
    print("(all files are in this one folder; add --by_task to sort them into")
    print(" one subfolder per task/video instead)")
