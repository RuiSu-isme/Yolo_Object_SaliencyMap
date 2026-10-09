"""
Turn YOLOv8 instance segmentation into a colorful heatmap-style video,
AND (optionally) save the raw per-frame grayscale saliency maps as
individual images -- needed for downstream analysis (NSS / AUC scripts).

=====================================================================
STEP 2 OF 4 - STIMULUS VIDEO -> SALIENCY MAPS
=====================================================================
Run this after step 1 (resample_gaze.py) and before the scoring scripts. It is
independent of the gaze data: it only looks at the stimulus videos, so it can
be run once per task and reused for every participant.

COMMANDS

    # every stimulus video of a folder -> maps + heatmap videos
    python yolo_heatmap_video.py --video_dir video --exclude "*_flipped*" --exclude "*_with_ROIs*"

    # a single video (writes the visualization next to you unless you name it)
    python yolo_heatmap_video.py video\\blockTowerR.mp4

    # see the plan without loading the model, writing nothing
    python yolo_heatmap_video.py --video_dir video --dry_run

    # only some videos, or only the first frames of each (quick test)
    python yolo_heatmap_video.py --video_dir video --only "blockTowerR.mp4" --max_frames 50

OUTPUT

    saliency_frames\\<name>\\frame_0001.png ...   one grayscale map per frame
    output\\<name>.mp4                           side-by-side heatmap video
    (single-video mode: --frames_dir DIR for the maps, and the output video is
     the second positional argument, default yolo_heatmap_output.mp4. Note that
     --frames_dir is the single-video option and --frames_root the batch one.)

REQUIREMENTS

    Python 3 with ultralytics + torch (CPU is fine: about 0.2 s per 1920x1080
    frame) and opencv-python. yolov8n-seg.pt must sit next to this script; it
    is downloaded automatically the first time if it is missing.

NOTES

    - the map has the same pixel size as the video, which is the coordinate
      space the gaze coordinates of step 1 refer to
    - videos that already have results are skipped, so an interrupted batch can
      simply be started again (--overwrite recomputes them)
    - only the NON-mirrored, ROI-free videos should be processed for the
      analysis: *_flipped and *_with_ROIs are the same content again
=====================================================================

For every frame, this script:
  1. Runs YOLOv8-seg to detect all objects and their pixel-level masks
  2. Combines every detected object's mask into a single map
     (brighter/more red = detected as an object, blue = background)
  3. Colorizes it like a saliency heatmap for the side-by-side video
  4. (optional) Saves the RAW grayscale map (same resolution as the
     original video, no colorization, no side-by-side stitching) as a
     standalone image per frame -- this is the format nss_score.py /
     auc_score.py expect, and it stays in the same pixel coordinate
     space as your gaze data.

SINGLE VIDEO (unchanged)
------------------------
    python yolo_heatmap_video.py <input_video> [output_video] [--frames_dir DIR]
    python yolo_heatmap_video.py my_video.mp4 result.mp4 --frames_dir ./saliency_frames

BATCH - A WHOLE FOLDER
----------------------
    # basic videos only (skips the mirrored *_flipped and the *_with_ROIs
    # exports, whose content would otherwise be processed twice)
    python yolo_heatmap_video.py --video_dir video --exclude "*_flipped*" --exclude "*_with_ROIs*"

    # every video in the folder
    python yolo_heatmap_video.py --video_dir video

For a video called NAME.mp4 the batch writes
    saliency_frames\\NAME\\frame_0001.png ...   raw grayscale maps for NSS / AUC
    output\\NAME.mp4                            side-by-side visualization video

Batch behaviour
    - the YOLO model is loaded once and reused for every video
    - a video whose output video AND frame folder already look complete is
      skipped, so an interrupted batch can simply be started again
      (--overwrite recomputes it, --dry_run only lists what would happen)
    - one unreadable video does not stop the batch; it is reported at the end
"""

import argparse
import os
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
FRAME_PREFIX = "frame_"

# The model sits next to this script, so the batch works from any directory.
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-seg.pt")


def load_model(model_path):
    """Load the YOLO segmentation model.

    Imported here rather than at the top of the file so that --help and
    --dry_run work before ultralytics (and torch) are installed.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ERROR: this step needs the 'ultralytics' package (which brings torch):\n"
            "    pip install ultralytics\n"
            f"(the import failed with: {exc})")

    print(f"Loading YOLOv8 segmentation model: {model_path}")
    return YOLO(model_path)


def video_info(video_path):
    """Frame count / size / fps of a video, or None when it cannot be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def count_saved_frames(frames_dir):
    if not frames_dir or not os.path.isdir(frames_dir):
        return 0
    return sum(
        1
        for name in os.listdir(frames_dir)
        if name.startswith(FRAME_PREFIX) and name.lower().endswith(".png")
    )


def is_complete(out_video, frames_dir, expected_frames):
    """True when this video already has a full-looking result on disk."""
    if not os.path.isfile(out_video):
        return False
    if not frames_dir:
        return True

    saved = count_saved_frames(frames_dir)
    if expected_frames and expected_frames > 0:
        return saved >= expected_frames
    return saved > 0


def clear_saved_frames(frames_dir):
    """Drop old frame_*.png so a recomputed video cannot leave stale frames."""
    if not os.path.isdir(frames_dir):
        return 0

    removed = 0
    for name in os.listdir(frames_dir):
        if name.startswith(FRAME_PREFIX) and name.lower().endswith(".png"):
            os.remove(os.path.join(frames_dir, name))
            removed += 1
    return removed


def process_video(model, video_path, out_video, frames_dir, max_frames=0):
    """Run detection over one video. Returns (stats, error_message)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, "could not open the video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "could not read the first frame"

    height, width = frame.shape[0], frame.shape[1]

    writer = None
    if out_video:
        out_dir = os.path.dirname(os.path.abspath(out_video))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width * 2, height))
        if not writer.isOpened():
            cap.release()
            return None, f"could not open the output video for writing: {out_video}"

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0

    try:
        while ret:
            # Run detection + segmentation on this frame
            results = model(frame, verbose=False)
            result = results[0]

            # Start with an all-black (zero) map the size of the frame
            combined_mask = np.zeros((height, width), dtype=np.float32)

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

                for i in range(masks.shape[0]):
                    mask = masks[i]
                    mask_resized = cv2.resize(mask, (width, height))
                    weight = float(confidences[i]) if confidences is not None else 1.0
                    combined_mask = np.maximum(combined_mask, mask_resized * weight)

            # Normalize to 0-255
            combined_mask_8u = (np.clip(combined_mask, 0, 1) * 255).astype(np.uint8)
            combined_mask_8u = cv2.GaussianBlur(combined_mask_8u, (3, 3), 1)

            # --- Save the raw grayscale map for analysis, if requested ---
            if frames_dir:
                frame_num = frame_count + 1  # start numbering at 1, matches nss_score.py default pattern
                raw_filename = f"{FRAME_PREFIX}{frame_num:04d}.png"
                cv2.imwrite(os.path.join(frames_dir, raw_filename), combined_mask_8u)

            # --- Build the colorful side-by-side video for viewing ---
            if writer is not None:
                heatmap = cv2.applyColorMap(combined_mask_8u, cv2.COLORMAP_JET)
                combined_frame = np.hstack((frame, heatmap))
                writer.write(combined_frame)

            frame_count += 1
            if frame_count % 20 == 0:
                print(f"    ...{frame_count} frames processed")

            if max_frames and frame_count >= max_frames:
                print(f"    stopped after {frame_count} frames (--max_frames)")
                break

            ret, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {"frames": frame_count, "width": width, "height": height, "fps": fps}, None


def select_videos(video_dir, only, exclude):
    """Videos of a folder, after applying --only / --exclude name patterns."""
    videos = [
        path
        for path in sorted(Path(video_dir).iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if only:
        videos = [p for p in videos if any(fnmatch(p.name, pattern) for pattern in only)]
    if exclude:
        videos = [p for p in videos if not any(fnmatch(p.name, pattern) for pattern in exclude)]

    return videos


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 segmentation heatmap video, one file or a whole folder."
    )

    parser.add_argument("input_video", nargs="?",
                        help="Single video to process (omit when using --video_dir)")
    parser.add_argument("output_video", nargs="?",
                        help="Single video: where to write the visualization video")

    parser.add_argument("--video_dir", default=None,
                        help="Batch mode: folder whose videos are all processed")
    parser.add_argument("--frames_dir", default=None,
                        help="Single video: folder for the raw grayscale saliency maps")
    parser.add_argument("--frames_root", default="saliency_frames",
                        help="Batch mode: root folder holding one frame subfolder per video")
    parser.add_argument("--output_dir", default="output",
                        help="Batch mode: folder for the visualization videos")
    parser.add_argument("--group", default=None,
                        help="Batch mode: optional subfolder under both roots "
                             "(e.g. --group test)")
    parser.add_argument("--only", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: process only videos matching this name pattern "
                             "(repeatable)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: skip videos matching this name pattern "
                             "(repeatable), e.g. --exclude \"*_flipped*\"")
    parser.add_argument("--overwrite", action="store_true",
                        help="Batch mode: recompute videos that already have results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Batch mode: only list what would be processed")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many frames per video (0 = whole video)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO segmentation weights (default: yolov8n-seg.pt next to this script)")

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Which mode are we in?
    # ----------------------------------------------------------------

    if args.video_dir and (args.input_video or args.output_video):
        print("ERROR: use either --video_dir (batch) or <input_video> (single), not both")
        sys.exit(1)

    if not args.video_dir and not args.input_video:
        print("ERROR: give a video file, or --video_dir for batch mode")
        sys.exit(1)

    if args.video_dir and not os.path.isdir(args.video_dir):
        print(f"ERROR: video folder does not exist: {args.video_dir}")
        sys.exit(1)

    # ================================================================
    # BATCH MODE
    # ================================================================

    if args.video_dir:
        videos = select_videos(args.video_dir, args.only, args.exclude)

        if not videos:
            print(f"ERROR: no videos matched in {args.video_dir}")
            sys.exit(1)

        jobs = []
        for video_path in videos:
            stem = video_path.stem
            frames_dir = os.path.join(args.frames_root, stem)
            out_video = os.path.join(args.output_dir, f"{stem}.mp4")
            if args.group:
                frames_dir = os.path.join(args.frames_root, args.group, stem)
                out_video = os.path.join(args.output_dir, args.group, f"{stem}.mp4")

            info = video_info(video_path)
            jobs.append({
                "path": video_path,
                "info": info,
                "frames_dir": frames_dir,
                "out_video": out_video,
                "skip": False,
                "error": None,
                "done_frames": 0,
            })

            if info is None:
                jobs[-1]["error"] = "could not read the video header"
                continue

            if not args.overwrite and is_complete(out_video, frames_dir, info["frames"]):
                jobs[-1]["skip"] = True

        print("================================")
        print("BATCH HEATMAP + SALIENCY FRAMES")
        print("================================")
        print(f"Video folder   : {args.video_dir}")
        print(f"Videos matched : {len(jobs)}")
        print(f"Frame folders  : {args.frames_root}")
        print(f"Output videos  : {args.output_dir}")
        if args.group:
            print(f"Group          : {args.group}")
        if args.exclude:
            print(f"Excluded       : {', '.join(args.exclude)}")
        print()

        todo = [job for job in jobs if not job["skip"] and not job["error"]]
        skipped = [job for job in jobs if job["skip"]]
        broken = [job for job in jobs if job["error"]]

        print(f"To process : {len(todo)}")
        print(f"Already done (skipped) : {len(skipped)}")
        if broken:
            print(f"Unreadable : {len(broken)}")
        print()

        if args.dry_run:
            print("Plan (--dry_run, nothing is written):")
            for job in jobs:
                state = "SKIP (already done)" if job["skip"] else ("ERROR: " + job["error"] if job["error"] else "process")
                frames = job["info"]["frames"] if job["info"] else 0
                print(f"  {job['path'].name:32s} {frames:6d} frames  [{state}]")
                print(f"      frames -> {job['frames_dir']}")
                print(f"      video  -> {job['out_video']}")
            sys.exit(0)

        if not todo:
            print("Nothing to do - every matched video already has results.")
            print("Use --overwrite to recompute them.")
            sys.exit(0)

        model = load_model(args.model)
        print()

        started = time.time()
        processed = 0
        failed = 0
        total_frames = 0

        for number, job in enumerate(todo, start=1):
            name = job["path"].name
            info = job["info"]
            print(f"[{number}/{len(todo)}] {name}  "
                  f"({info['frames']} frames, {info['width']}x{info['height']}, {info['fps']:.0f} fps)")
            print(f"    frames -> {job['frames_dir']}")
            print(f"    video  -> {job['out_video']}")

            if args.overwrite:
                removed = clear_saved_frames(job["frames_dir"])
                if removed:
                    print(f"    cleared {removed} old frame image(s)")

            video_started = time.time()
            stats, error = process_video(
                model,
                job["path"],
                job["out_video"],
                job["frames_dir"],
                max_frames=args.max_frames,
            )

            if error:
                failed += 1
                job["error"] = error
                print(f"    ERROR: {error} (continuing with the next video)\n")
                continue

            processed += 1
            total_frames += stats["frames"]
            elapsed = time.time() - video_started
            print(f"    done: {stats['frames']} frames in {elapsed/60:.1f} min "
                  f"({elapsed/max(stats['frames'], 1)*1000:.0f} ms/frame)\n")

        elapsed = time.time() - started
        print("================================")
        print("BATCH COMPLETE")
        print("================================")
        print(f"Processed        : {processed}")
        print(f"Skipped (done)   : {len(skipped)}")
        print(f"Failed           : {failed}")
        print(f"Frames processed : {total_frames}")
        print(f"Elapsed          : {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
        print(f"Frame folders    : {args.frames_root}")
        print(f"Output videos    : {args.output_dir}")

        problems = [job for job in jobs if job["error"]]
        if problems:
            print("\nVideos with problems:")
            for job in problems:
                print(f"  {job['path'].name}: {job['error']}")

        if processed == 0:
            sys.exit(1)

        return

    # ================================================================
    # SINGLE VIDEO (original behaviour)
    # ================================================================

    input_video = args.input_video
    output_video = args.output_video or "yolo_heatmap_output.mp4"

    if not os.path.isfile(input_video):
        print(f"ERROR: video does not exist: {input_video}")
        sys.exit(1)

    model = load_model(args.model)

    if args.frames_dir:
        print(f"Raw per-frame saliency maps will be saved to: {args.frames_dir}")

    print("Processing... this may take a while, especially on CPU.")

    stats, error = process_video(model, input_video, output_video, args.frames_dir,
                                 max_frames=args.max_frames)
    if error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print(f"Done. Processed {stats['frames']} frames.")
    print(f"Visualization video saved to: {output_video}")
    if args.frames_dir:
        print(f"Raw per-frame saliency maps saved to: {args.frames_dir}")


if __name__ == "__main__":
    main()
