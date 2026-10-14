r"""
FACE saliency maps for the study's folder layout, in the exact shape the
existing NSS/AUC scripts already expect.

=====================================================================
WHAT IT WRITES (this is the whole point of the script)
=====================================================================
For a stimulus video called NAME.mp4 (NAME is the task name):

    output\\NAME_face.mp4                  side-by-side video: original | face map
    saliency_frames_face\\NAME\\frame_0001.png ...
                                           one grayscale map per frame, for
                                           nss_score_new.py / auc_score_new.py

The frame folders are byte-for-byte the same contract as the object maps in
saliency_frames\\NAME\\: single channel, same pixel size as the video (the
coordinate space the gaze data refers to), numbered from 0001 with no gaps.
object maps                            face maps
    saliency_frames\\look4CatR\\frame_0001.png      saliency_frames_face\\look4CatR\\frame_0001.png
    output\\look4CatR.mp4                          output\\look4CatR_face.mp4

=====================================================================
WHY "_face" AND NOT "_head"
=====================================================================
The map is the face-shaped outline: the convex hull of MediaPipe
FaceLandmarker's 478 landmarks, fitted inside each face that YuNet finds.
Rectangles are the fallback.

That outline is the FACIAL OVAL. It does NOT include hair: MediaPipe has no
hair landmarks at all, so a name containing "head" would promise something
the map does not deliver. "_face" describes what is actually in the pixels.

If you do want the whole head including hair, --include_hair unions in the
hair segmenter output. It is off by default on purpose: it costs about 15x
more per frame (~120 ms) and it over-segments on some subjects (on our test
clip it also classified a necklace as hair).

Independent of the name: mediapipe is unreliable on small faces. The
outline fits ~100% of faces on the close-up H/R stimuli, but on the srt
clips, where faces are ~14-24 px, it fits none and every region becomes a
rectangle. The script prints the fit rate per video and warns when it drops
below 50%, so a rectangle-shaped run is never mistaken for a shaped one.

=====================================================================
RUNNING THIS FROM ANOTHER FOLDER
=====================================================================
Two files are needed, and the third line is only for the assets:

    yolo_face_saliency.py        this script
    face_hand_core.py            the shared engine it imports
    models\\ + _vendor\\            only if you do not point FACE_HAND_HOME
                                 at an existing copy of them

    set FACE_HAND_HOME=D:\\PhD\\Thesis\\Object level\\Yolo
    python D:\\my\\new\\folder\\yolo_face_saliency.py --video_dir D:\\my\\videos

The maps and videos are written relative to the CURRENT DIRECTORY, not to the
script, so run it from the folder you want saliency_frames_face\\ and output\\
to appear in. Both folders are created if missing.

=====================================================================
COMMANDS
=====================================================================
    # a single stimulus -> saliency_frames_face\NAME\ + output\NAME_face.mp4
    python yolo_face_saliency.py video\look4CupH.mp4

    # a single stimulus, maps only, no video
    python yolo_face_saliency.py video\look4CupH.mp4 --no_video

    # the whole stimulus set (mirrored and ROI exports skipped by default,
    # because they are the same content again and would double-count)
    python yolo_face_saliency.py --video_dir video

    # include everything, mirrored copies and all
    python yolo_face_saliency.py --video_dir video --no_default_excludes

    # what would happen, writing nothing
    python yolo_face_saliency.py --video_dir video --dry_run

    # rectangles instead of shapes, for the ablation
    python yolo_face_saliency.py --video_dir video --mask_shape box ^
        --frames_root saliency_frames_face_box --suffix _face_box

    # red test of the contract (naming, sizes, frames); no video needed
    python yolo_face_saliency.py --self_test

=====================================================================
NOTES
=====================================================================
    - an interrupted batch can simply be restarted: a video whose frames and
      video already look complete is skipped (--overwrite recomputes it).
    - the maps are what nss_score_new.py / auc_score_new.py read, e.g.
        python nss_score_new.py --gaze_dir gazedata --saliency_root saliency_frames_face
    - the engine lives in face_hand_core.py, shared with
      yolo_face_hand_saliency.py, so the two cannot drift apart.
    - models/ and _vendor/ are found next to this script, or via
      FACE_HAND_HOME (see face_hand_core.py).
"""

import argparse
import os
import shutil
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np

# face_hand_core.py normally sits next to this script (then Python finds it by
# itself). If this script was copied somewhere on its own, look for the engine
# in $FACE_HAND_HOME or in the current directory before giving up.
try:
    import face_hand_core as core
except ImportError:                                              # pragma: no cover
    _candidates = []
    _home = os.environ.get("FACE_HAND_HOME", "").strip()
    if _home:
        _candidates.append(_home)
    _candidates.append(os.getcwd())
    for _candidate in _candidates:
        if os.path.isfile(os.path.join(_candidate, "face_hand_core.py")):
            sys.path.insert(0, _candidate)
            break
    else:
        raise SystemExit(
            "ERROR: face_hand_core.py was not found.\n"
            "It is the shared engine this script imports. Either copy it next to\n"
            "this script, or point FACE_HAND_HOME at the folder that has it:\n"
            "    set FACE_HAND_HOME=D:\\PhD\\Thesis\\Object level\\Yolo")
    import face_hand_core as core

from face_hand_core import (
    ASSET_HOME_ENV,
    DEFAULT_FACE_MODEL,
    DEFAULT_MESH_MODEL,
    FRAME_PREFIX,
    MASK_SHAPES,
    VIDEO_EXTENSIONS,
    FaceDetector,
    FaceMeshSegmenter,
    FaceRegion,
    empty_map,
    face_boxes,
    faces_for,
    finish_map,
)

# tell the core where this script lives, so a copy of it in another folder
# still finds the shared models/ and _vendor/
core.set_caller_dir(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FRAMES_ROOT = "saliency_frames_face"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SUFFIX = "_face"

# the mirrored and ROI exports are the same content again; counting them
# would duplicate every stimulus in the analysis
DEFAULT_EXCLUDES = ("*_flipped*", "*_with_ROIs*")


# =====================================================================
# One video
# =====================================================================

def process_video(face_detector, mesh_segmenter, video_path, out_video, frames_dir,
                  args, max_frames=0):
    """Write the map frames and (optionally) the side-by-side video.

    Returns (stats, error_message).
    """
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
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width * 2, height))
        if not writer.isOpened():
            print(f"    DEBUG writer failed: path={out_video!r} fps={fps} "
                  f"size=({width * 2}, {height}) exists={os.path.exists(str(out_video))} "
                  f"dir_exists={os.path.isdir(os.path.dirname(os.path.abspath(out_video)))}")
            cap.release()
            return None, f"could not open the output video for writing: {out_video}"

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    frames_done = 0
    frames_with_face = 0
    face_regions = 0
    mesh_regions = 0
    box_regions = 0
    small_face_frames = 0
    peak_sum = 0.0

    try:
        while ret:
            regions, _ = faces_for(frame, face_detector, mesh_segmenter,
                                   args.mask_shape, args.box_scale)
            if regions:
                frames_with_face += 1
                face_regions += len(regions)
                mesh_regions += sum(1 for r in regions if r.kind == "mesh")
                box_regions += sum(1 for r in regions if r.kind == "box")
                if min(r.area for r in regions) < 32 * 32:
                    small_face_frames += 1

            map_ = empty_map(height, width)
            for region in regions:
                region.paint(map_, region.score if args.weight == "conf" else 1.0)

            peak_sum += float(map_.max())
            map_8u = finish_map(map_, args.blur_sigma, args.normalize)

            if frames_dir:
                name = f"{FRAME_PREFIX}{frames_done + 1:04d}.png"
                cv2.imwrite(os.path.join(frames_dir, name), map_8u)

            if writer is not None:
                heat = cv2.applyColorMap(map_8u, cv2.COLORMAP_JET)
                label = f"head map (shape={args.mask_shape}, weight={args.weight})"
                cv2.putText(heat, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(np.hstack((frame, heat)))

            frames_done += 1
            if frames_done % 200 == 0:
                print(f"    ...{frames_done} frames processed")

            if max_frames and frames_done >= max_frames:
                print(f"    stopped after {frames_done} frames (--max_frames)")
                break

            ret, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    stats = {
        "frames": frames_done,
        "width": width,
        "height": height,
        "fps": fps,
        "face_frames": frames_with_face,
        "face_regions": face_regions,
        "mesh_regions": mesh_regions,
        "box_regions": box_regions,
        "small_face_frames": small_face_frames,
        "mean_peak": (peak_sum / frames_done) if frames_done else 0.0,
    }
    return stats, None


# =====================================================================
# Batch bookkeeping
# =====================================================================

def video_info(video_path):
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
    return sum(1 for name in os.listdir(frames_dir)
               if name.startswith(FRAME_PREFIX) and name.lower().endswith(".png"))


def is_complete(out_video, frames_dir, expected_frames, want_video):
    if want_video and not os.path.isfile(out_video):
        return False
    if not frames_dir:
        return True
    saved = count_saved_frames(frames_dir)
    if expected_frames and expected_frames > 0:
        return saved >= expected_frames
    return saved > 0


def clear_saved_frames(frames_dir):
    if not os.path.isdir(frames_dir):
        return 0
    removed = 0
    for name in os.listdir(frames_dir):
        if name.startswith(FRAME_PREFIX) and name.lower().endswith(".png"):
            os.remove(os.path.join(frames_dir, name))
            removed += 1
    return removed


def select_videos(video_dir, only, exclude):
    videos = [p for p in sorted(Path(video_dir).iterdir())
              if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if only:
        videos = [p for p in videos if any(fnmatch(p.name, pat) for pat in only)]
    if exclude:
        videos = [p for p in videos if not any(fnmatch(p.name, pat) for pat in exclude)]
    return videos


# =====================================================================
# Self test: the file contract, on synthetic data
# =====================================================================

def self_test(args):
    print("Self test: naming, folder layout and map format (no video needed).")
    ok = True

    def check(label, condition):
        nonlocal ok
        ok = ok and bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    # a workspace-relative scratch folder: the OS temp dir is not writable
    # under this environment's sandbox
    tmp = os.path.join(HERE, "_self_test_tmp")
    try:
        frames_dir = os.path.join(tmp, DEFAULT_FRAMES_ROOT, "taskX")
        out_video = os.path.join(tmp, DEFAULT_OUTPUT_DIR, f"taskX{DEFAULT_SUFFIX}.mp4")

        height, width = 64, 96
        saved = []
        for index in range(1, 4):
            region = FaceRegion((10, 20, 50, 60), 0.9, "box")
            map_ = empty_map(height, width)
            region.paint(map_, region.score)
            map_8u = finish_map(map_, 0.0, False)
            os.makedirs(frames_dir, exist_ok=True)
            name = f"{FRAME_PREFIX}{index:04d}.png"
            cv2.imwrite(os.path.join(frames_dir, name), map_8u)
            saved.append(name)

        check("frames root folder name", DEFAULT_FRAMES_ROOT == "saliency_frames_face")
        check("frame naming frame_0001.png", saved[0] == "frame_0001.png")
        check("frame naming is zero-padded to 4", saved[-1] == "frame_0003.png")

        read_back = cv2.imread(os.path.join(frames_dir, saved[0]), cv2.IMREAD_UNCHANGED)
        check("one channel", read_back.ndim == 2)
        check("same size as the source frame", read_back.shape == (height, width))
        check("uint8", read_back.dtype == np.uint8)
        check("inside the region is painted", read_back[40, 30] == int(0.9 * 255))
        check("outside the region is zero", read_back[5, 5] == 0)

        check("count_saved_frames sees them", count_saved_frames(frames_dir) == 3)
        check("output video path is NAME_face.mp4",
              out_video.endswith(os.path.join(DEFAULT_OUTPUT_DIR, "taskX_face.mp4")))
        check("suffix matches the folder naming", DEFAULT_SUFFIX == "_face")

        # completeness: 3 frames on disk, 3 expected -> done
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        with open(out_video, "wb") as handle:
            handle.write(b"")
        check("complete when frames and video exist",
              is_complete(out_video, frames_dir, 3, True))
        check("incomplete when fewer frames than expected",
              not is_complete(out_video, frames_dir, 4, True))
        check("incomplete when the video is missing",
              not is_complete(out_video + ".missing", frames_dir, 3, True))

        removed = clear_saved_frames(frames_dir)
        check("clear_saved_frames removes exactly the maps", removed == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nSelf test:", "all checks passed" if ok else "FAILURES above")
    return 0 if ok else 1


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Head (face-shaped) saliency maps in the study's folder layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input_video", nargs="?",
                        help="Single video to process (omit when using --video_dir)")
    parser.add_argument("--video_dir", default=None,
                        help="Batch mode: folder whose videos are all processed")

    parser.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT,
                        help=f"Root folder for the per-frame maps (default: {DEFAULT_FRAMES_ROOT})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder for the side-by-side videos (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX,
                        help=f"Suffix appended to the video name (default: {DEFAULT_SUFFIX}, "
                             "so NAME.mp4 -> NAME_head.mp4)")
    parser.add_argument("--group", default=None,
                        help="Optional subfolder under both roots (e.g. --group test)")
    parser.add_argument("--no_video", action="store_true",
                        help="Write only the maps, not the side-by-side video")

    parser.add_argument("--only", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: process only videos matching this name pattern")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: skip videos matching this name pattern "
                             "(repeatable; added on top of the default excludes)")
    parser.add_argument("--no_default_excludes", action="store_true",
                        help="Do not skip *_flipped and *_with_ROIs (which duplicate content)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute videos that already have results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list what would be processed")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many frames per video (0 = whole video)")

    parser.add_argument("--mask_shape", choices=list(MASK_SHAPES), default="mesh",
                        help="mesh (default) = face-shaped outline; box = rectangle. "
                             "Small faces fall back to a rectangle either way, and the "
                             "fit rate is reported so this is never silent")
    parser.add_argument("--face_model", default=DEFAULT_FACE_MODEL,
                        help="YuNet ONNX weights for finding faces")
    parser.add_argument("--mesh_model", default=DEFAULT_MESH_MODEL,
                        help="MediaPipe face landmarker .task file")
    parser.add_argument("--det_size", type=int, default=640,
                        help="YuNet input size (default: 640)")
    parser.add_argument("--face_threshold", type=float, default=0.5,
                        help="YuNet score threshold (default: 0.5)")
    parser.add_argument("--mesh_confidence", type=float, default=0.5,
                        help="MediaPipe face threshold (default: 0.5; keep it there)")
    parser.add_argument("--mesh_dilate", type=float, default=0.0,
                        help="Grow the outline by this fraction of its size")
    parser.add_argument("--include_hair", action="store_true",
                        help="mesh mode: also cover the hair via the hair segmenter, making "
                             "the map a whole-head region. ~15x slower and it over-segments "
                             "on some subjects; off by default so the map stays a face region")
    parser.add_argument("--hair_model", default=core.DEFAULT_HAIR_MODEL,
                        help="MediaPipe hair segmenter .tflite, only used by --include_hair")
    parser.add_argument("--num_faces", type=int, default=5,
                        help="Maximum faces per frame (default: 5)")

    parser.add_argument("--weight", choices=["conf", "1"], default="conf",
                        help="Pixel value inside the region: detector confidence, or a flat 1.0")
    parser.add_argument("--blur_sigma", type=float, default=0.0,
                        help="Gaussian softening in pixels (0 = hard edges)")
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Scale every detection about its centre")
    parser.add_argument("--normalize", action="store_true",
                        help="Rescale each frame so its maximum becomes 255 "
                             "(off by default: it destroys frame-to-frame comparability)")

    parser.add_argument("--self_test", action="store_true",
                        help="Check the naming/format contract on synthetic data and exit")

    return parser


def main():
    args = build_parser().parse_args()

    if args.self_test:
        sys.exit(self_test(args))

    if args.video_dir and args.input_video:
        print("ERROR: use either --video_dir (batch) or <input_video> (single), not both")
        sys.exit(1)
    if not args.video_dir and not args.input_video:
        print("ERROR: give a video file, or --video_dir for batch mode")
        sys.exit(1)
    if args.video_dir and not os.path.isdir(args.video_dir):
        print(f"ERROR: video folder does not exist: {args.video_dir}")
        sys.exit(1)

    want_video = not args.no_video

    # the core handles the mechanics; only load what the shape mode needs
    face_detector = FaceDetector(args.face_model, det_size=args.det_size,
                                 score_threshold=args.face_threshold)
    mesh_segmenter = None
    if args.mask_shape == "mesh":
        mesh_segmenter = FaceMeshSegmenter(
            args.mesh_model,
            face_detector=face_detector,
            include_hair=args.include_hair,
            hair_model_path=args.hair_model,
            num_faces=args.num_faces,
            min_confidence=args.mesh_confidence,
            dilate=args.mesh_dilate,
            box_scale=args.box_scale)

    shape_words = "face-shaped outline"
    if args.mask_shape == "mesh":
        shape_words += " + hair (whole head)" if args.include_hair else ", no hair"
    print(f"Shape: {args.mask_shape} ({shape_words})"
          if args.mask_shape == "mesh" else f"Shape: {args.mask_shape} (rectangles)")
    print(f"Maps  -> {args.frames_root}\\<name>\\{FRAME_PREFIX}0001.png ...")
    print(f"Video -> {args.output_dir}\\<name>{args.suffix}.mp4"
          if want_video else "Video -> (disabled by --no_video)")

    # ----------------------------------------------------------------
    # BATCH
    # ----------------------------------------------------------------
    if args.video_dir:
        excludes = list(args.exclude)
        if not args.no_default_excludes:
            excludes += [p for p in DEFAULT_EXCLUDES if p not in excludes]

        videos = select_videos(args.video_dir, args.only, excludes)
        if not videos:
            print(f"ERROR: no videos matched in {args.video_dir}")
            sys.exit(1)

        jobs = []
        for video_path in videos:
            frames_dir = os.path.join(args.frames_root, video_path.stem)
            out_video = os.path.join(args.output_dir, f"{video_path.stem}{args.suffix}.mp4")
            if args.group:
                frames_dir = os.path.join(args.frames_root, args.group, video_path.stem)
                out_video = os.path.join(args.output_dir, args.group,
                                         f"{video_path.stem}{args.suffix}.mp4")

            info = video_info(video_path)
            job = {"path": video_path, "info": info, "frames_dir": frames_dir,
                   "out_video": out_video if want_video else None, "skip": False,
                   "error": None, "stats": None}
            if info is None:
                job["error"] = "could not read the video header"
            elif not args.overwrite and is_complete(job["out_video"], frames_dir,
                                                    info["frames"], want_video):
                job["skip"] = True
            jobs.append(job)

        todo = [j for j in jobs if not j["skip"] and not j["error"]]
        skipped = [j for j in jobs if j["skip"]]
        broken = [j for j in jobs if j["error"]]

        print()
        print("================================")
        print("HEAD SALIENCY - BATCH")
        print("================================")
        print(f"Video folder   : {args.video_dir}")
        print(f"Videos matched : {len(jobs)}")
        print(f"To process     : {len(todo)}")
        print(f"Already done   : {len(skipped)}")
        if broken:
            print(f"Unreadable     : {len(broken)}")
        if excludes:
            print(f"Excluded       : {', '.join(excludes)}")
        print()

        if args.dry_run:
            print("Plan (--dry_run, nothing is written):")
            for job in jobs:
                state = ("SKIP (already done)" if job["skip"]
                         else ("ERROR: " + job["error"] if job["error"] else "process"))
                frames = job["info"]["frames"] if job["info"] else 0
                print(f"  {job['path'].name:32s} {frames:6d} frames  [{state}]")
                print(f"      maps  -> {job['frames_dir']}")
                if job["out_video"]:
                    print(f"      video -> {job['out_video']}")
            sys.exit(0)

        if not todo:
            print("Nothing to do - every matched video already has results.")
            print("Use --overwrite to recompute them.")
            sys.exit(0)

        started = time.time()
        processed = failed = total_frames = 0
        total_regions = total_mesh = total_box = small_total = 0

        for number, job in enumerate(todo, start=1):
            info = job["info"]
            print(f"[{number}/{len(todo)}] {job['path'].name}  "
                  f"({info['frames']} frames, {info['width']}x{info['height']}, {info['fps']:.0f} fps)")
            print(f"    maps  -> {job['frames_dir']}")
            if job["out_video"]:
                print(f"    video -> {job['out_video']}")

            if args.overwrite:
                removed = clear_saved_frames(job["frames_dir"])
                if removed:
                    print(f"    cleared {removed} old frame image(s)")

            video_started = time.time()
            stats, error = process_video(face_detector, mesh_segmenter, job["path"],
                                        job["out_video"], job["frames_dir"], args,
                                        max_frames=args.max_frames)
            if error:
                failed += 1
                job["error"] = error
                print(f"    ERROR: {error} (continuing with the next video)\n")
                continue

            job["stats"] = stats
            processed += 1
            total_frames += stats["frames"]
            total_regions += stats["face_regions"]
            total_mesh += stats["mesh_regions"]
            total_box += stats["box_regions"]
            small_total += stats["small_face_frames"]

            elapsed = time.time() - video_started
            percent_face = 100.0 * stats["face_frames"] / max(stats["frames"], 1)
            print(f"    done: {stats['frames']} frames in {elapsed/60:.1f} min "
                  f"({elapsed/max(stats['frames'], 1)*1000:.0f} ms/frame)")
            print(f"    faces in {stats['face_frames']}/{stats['frames']} frames "
                  f"({percent_face:.1f}%), {stats['face_regions']} regions")

            if args.mask_shape == "mesh":
                regions = max(stats["face_regions"], 1)
                fit = 100.0 * stats["mesh_regions"] / regions
                print(f"    shapes: {stats['mesh_regions']}/{stats['face_regions']} regions "
                      f"got a mesh outline ({fit:.0f}%)")
                if fit < 50:
                    print(f"    WARNING: mostly rectangles -- faces too small for MediaPipe")
            if stats["small_face_frames"]:
                print(f"    note: {stats['small_face_frames']} frames have a face under 32x32 px")
            print()

        elapsed = time.time() - started
        print("================================")
        print("BATCH COMPLETE")
        print("================================")
        print(f"Processed        : {processed}")
        print(f"Skipped (done)   : {len(skipped)}")
        print(f"Failed           : {failed}")
        print(f"Frames written   : {total_frames}")
        print(f"Face regions     : {total_regions}")
        if args.mask_shape == "mesh":
            regions = max(total_regions, 1)
            print(f"Mesh outlines    : {total_mesh}/{total_regions} "
                  f"({100.0 * total_mesh / regions:.0f}%)"
                  + (f", {total_box} fell back to a rectangle" if total_box else ""))
        if small_total:
            print(f"Small-face frames: {small_total} (< 32x32 px)")
        print(f"Elapsed          : {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
        print(f"Maps             : {args.frames_root}\\<name>\\")
        if want_video:
            print(f"Videos           : {args.output_dir}\\<name>{args.suffix}.mp4")
        print()
        print("Next: score these maps against the gaze data, e.g.")
        print(f"    python nss_score_new.py --gaze_dir gazedata --saliency_root {args.frames_root}")

        problems = [j for j in jobs if j["error"]]
        if problems:
            print("\nVideos with problems:")
            for job in problems:
                print(f"  {job['path'].name}: {job['error']}")

        if processed == 0:
            sys.exit(1)
        return

    # ----------------------------------------------------------------
    # SINGLE VIDEO
    # ----------------------------------------------------------------
    input_video = args.input_video
    if not os.path.isfile(input_video):
        print(f"ERROR: video does not exist: {input_video}")
        sys.exit(1)

    stem = Path(input_video).stem
    frames_dir = os.path.join(args.frames_root, stem)
    if args.group:
        frames_dir = os.path.join(args.frames_root, args.group, stem)
    out_video = None
    if want_video:
        out_video = os.path.join(args.output_dir, f"{stem}{args.suffix}.mp4")

    print(f"    maps  -> {frames_dir}")
    if out_video:
        print(f"    video -> {out_video}")

    stats, error = process_video(face_detector, mesh_segmenter, input_video, out_video,
                                frames_dir, args, max_frames=args.max_frames)
    if error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print(f"Done. {stats['frames']} frames "
          f"({stats['width']}x{stats['height']}, {stats['fps']:.0f} fps).")
    percent_face = 100.0 * stats["face_frames"] / max(stats["frames"], 1)
    print(f"Faces in {stats['face_frames']}/{stats['frames']} frames ({percent_face:.1f}%), "
          f"{stats['face_regions']} regions.")
    if args.mask_shape == "mesh":
        regions = max(stats["face_regions"], 1)
        fit = 100.0 * stats["mesh_regions"] / regions
        print(f"Shapes: {stats['mesh_regions']}/{stats['face_regions']} regions got a real "
              f"mesh outline ({fit:.0f}%); {stats['box_regions']} fell back to a RECTANGLE.")
        if fit < 50:
            print("        WARNING: most or all faces were too small for MediaPipe to fit "
                  "landmarks, so these maps are effectively box maps.")
            print("        Compare with --mask_shape box before interpreting them.")
    if stats["small_face_frames"]:
        print(f"WARNING: {stats['small_face_frames']} frames have a face smaller than 32x32 px.")
    print(f"Maps written to: {frames_dir}")
    if out_video:
        print(f"Video written to: {out_video}")


if __name__ == "__main__":
    main()
