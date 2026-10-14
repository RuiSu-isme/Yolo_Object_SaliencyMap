r"""
FACE + HAND saliency maps using the Google MediaPipe landmark models.

=====================================================================
WHAT THIS IS
=====================================================================
Three saliency representations per stimulus video:

    hand    MediaPipe HandLandmarker (21 landmarks per hand)
    face    REUSED from the existing saliency_frames_face\ maps, not recomputed
    social  max(face, hand) -- the union of both social regions

Why the face maps are reused instead of recomputed: MediaPipe's own
FaceLandmarker was measured against the YuNet detector already used for
saliency_frames_face\, and on this stimulus set it is clearly worse
(mixingColorsH 0% vs 100%, memorySimpleH 32% vs 100%, buildQueuH 0% vs 100%
at confidence 0.5). So the face side stays on YuNet and this script only
adds the hand side plus the combined map.

=====================================================================
OUTPUTS
=====================================================================
For a stimulus NAME.mp4:

    saliency_frames_hand\NAME\frame_0001.png ...    hand maps   (NSS / AUC)
    output\NAME_hand.mp4                            hand video
    saliency_frames_social\NAME\frame_0001.png ...  face+hand maps
    output\NAME_social.mp4                          social video

    saliency_frames_face\NAME\...   read-only input: produced by
    output\NAME_face.mp4            yolo_face_saliency.py (YuNet)

All maps are single-channel 8-bit PNGs the size of the video, frame-numbered
from 0001, so nss_score_new.py / auc_score_new.py read them unchanged:

    python nss_score_new.py --gaze_dir gazedata --saliency_root saliency_frames_hand
    python nss_score_new.py --gaze_dir gazedata --saliency_root saliency_frames_social

Every non-srt stimulus was measured to contain hands (>=48% of frames in all
but one clip, 18 of 30 clips above 80%), so all-zero frames are occasional,
not the norm -- except look4CupH, where hands appear in only 3% of frames.

=====================================================================
THE HAND SHAPE: HULL + SOFT OUTWARD EDGE
=====================================================================
The 21 landmarks give a convex hull that traces the hand itself. Measured on
real frames the hull is accurate but slightly tight -- it tracks the fingers
and misses a little skin at the boundary, so a mask that stops exactly at the
hull leaves a thin uncovered rim.

--hand_soft_edge N therefore grows the region OUTWARD from the hull and fades
it to zero over N pixels: the hull interior is 1.0, and every pixel outside it
decays linearly with distance, reaching 0 at N. That closes the rim and gives
the gradient edge of an ordinary heatmap instead of a hard cut.

    --hand_soft_edge 0    hard hull edge, no growth (exactly the old behaviour)
    --hand_soft_edge 12   default: ~12 px of graded coverage beyond the hull
    --hand_soft_edge 30   generous; starts to reach held objects and wrist

Raising it trades specificity for recall: the wider the falloff, the more
background enters the map. The right value is an empirical question for your
scoring -- run one clip at 0, 12 and 30 and compare NSS/AUC.

Held objects are NOT included by design: in the hull check, a hand holding a
cup produced a hull over the fingers only. --hand_dilate grows the hull itself
before the soft edge, which is the knob to reach held objects.

Face maps keep whatever edge the face script produced (hard by default): they
are copied, not regenerated, so the face numbers stay identical to the ones
already computed.

=====================================================================
COMMANDS
=====================================================================
    # one video -> hand + social
    python yolo_social_saliency.py video\mixingColorsH.mp4

    # the two pilot clips the owner picked
    python yolo_social_saliency.py --video_dir video --only "mixingColorsH.mp4" ^
        --only "memorySimpleH.mp4"

    # whole set (mirrored and ROI exports skipped by default)
    python yolo_social_saliency.py --video_dir video

    # see the plan, write nothing
    python yolo_social_saliency.py --video_dir video --dry_run

    # hand maps only, no combined map
    python yolo_social_saliency.py video\mixingColorsH.mp4 --no_social

    # examine the edge question on one clip
    python yolo_social_saliency.py video\memorySimpleH.mp4 --hand_soft_edge 0
    python yolo_social_saliency.py video\memorySimpleH.mp4 --hand_soft_edge 30

    # contract test on synthetic data, no video needed
    python yolo_social_saliency.py --self_test

=====================================================================
REQUIREMENTS
=====================================================================
    opencv-python >= 4.5.4, numpy
    _vendor\ with the vendored mediapipe (see face_hand_core.py)
    models\hand_landmarker.task                        (7.8 MB, included)

The face maps of every video you process must already exist in
saliency_frames_face\<name>\ -- run yolo_face_saliency.py first. Missing face
maps are reported per video and that video is skipped, never half-written.
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
            "Copy it next to this script, or point FACE_HAND_HOME at the folder "
            "that holds it:\n"
            "    set FACE_HAND_HOME=D:\\PhD\\Thesis\\Object level\\Yolo")
    import face_hand_core as core

from face_hand_core import (      # noqa: E402
    FRAME_PREFIX,
    VIDEO_EXTENSIONS,
    empty_map,
    finish_map,
)

core.set_caller_dir(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_HAND_MODEL = core.asset_path("hand_landmarker.task")
DEFAULT_FACE_FRAMES = "saliency_frames_face"
DEFAULT_HAND_FRAMES = "saliency_frames_hand"
DEFAULT_SOCIAL_FRAMES = "saliency_frames_social"
DEFAULT_OUTPUT_DIR = "output"

DEFAULT_EXCLUDES = ("*_flipped*", "*_with_ROIs*")


# =====================================================================
# Hand regions
# =====================================================================

class HandRegion:
    """One detected hand: a hull that paints itself with a soft outward edge."""

    __slots__ = ("box", "score", "hull")

    def __init__(self, box, score, hull):
        self.box = box
        self.score = float(score)
        self.hull = hull

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def mask(self, shape, soft_edge=0.0, dilate=0.0):
        """uint8 0/255 mask: filled hull grown outward and faded over soft_edge px."""
        height, width = shape[:2]
        hull = self.hull
        if dilate:
            center, _, _ = cv2.minAreaRect(hull.astype(np.float32))
            cx, cy = center
            grown = np.empty_like(hull, dtype=np.float32)
            grown[:, 0] = cx + (hull[:, 0] - cx) * (1.0 + dilate)
            grown[:, 1] = cy + (hull[:, 1] - cy) * (1.0 + dilate)
            grown[:, 0] = np.clip(grown[:, 0], 0, width - 1)
            grown[:, 1] = np.clip(grown[:, 1], 0, height - 1)
            hull = cv2.convexHull(grown.astype(np.int32)).reshape(-1, 2)

        filled = np.zeros((height, width), np.uint8)
        cv2.drawContours(filled, [hull.astype(np.int32)], -1, 255, cv2.FILLED)

        if soft_edge and soft_edge > 0:
            # distance to the hull: 0 inside, and the outward distance outside.
            # decay over soft_edge pixels so the mask reaches exactly 0 there.
            distance = cv2.distanceTransform(255 - filled, cv2.DIST_L2, 3)
            falloff = np.clip(1.0 - distance / float(soft_edge), 0.0, 1.0)
            return (falloff * 255.0).astype(np.uint8)
        return filled


class HandMasker:
    """MediaPipe HandLandmarker (21 landmarks).

    HandLandmarker contains its own palm detector, so a full frame is passed
    straight in -- no separate detection stage and no extra model file.

    mediapipe is imported lazily so --help and --dry_run work without it.
    """

    def __init__(self, model_path, num_hands=4, confidence=0.5):
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"ERROR: hand landmarker model not found: {model_path}\n"
                "Download hand_landmarker.task (7.8 MB) into models/:\n"
                "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")

        found_vendor = core.vendor_dir()
        if found_vendor and found_vendor not in sys.path:
            sys.path.insert(0, found_vendor)

        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise SystemExit(
                "ERROR: this script needs mediapipe, which is not importable:\n"
                f"    {exc}\n"
                f"Set {core.ASSET_HOME_ENV} to the folder holding _vendor, or copy "
                "_vendor next to this script.")

        self.vision = vision
        self.num_hands = int(num_hands)
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=float(confidence),
                min_hand_presence_confidence=float(confidence),
                min_tracking_confidence=float(confidence)))
        self.calls = 0

    def _mp_image(self, rgb):
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        return Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def detect(self, frame):
        """Return a list of HandRegion for this frame."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(self._mp_image(rgb))
        self.calls += 1

        regions = []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            points = np.array([[p.x * width, p.y * height] for p in landmarks],
                              dtype=np.float32)
            hull = cv2.convexHull(points.astype(np.int32)).reshape(-1, 2)
            hull = np.clip(hull, [0, 0], [width - 1, height - 1]).astype(np.int32)

            score = 1.0
            handedness = getattr(result, "handedness", None)
            if handedness and index < len(handedness) and handedness[index]:
                score = float(handedness[index][0].score)

            x, y, w, h = cv2.boundingRect(hull)
            regions.append(HandRegion((float(x), float(y), float(x + w), float(y + h)),
                                      score, hull))
        return regions

    def close(self):
        try:
            self.landmarker.close()
        except Exception:                                      # noqa: BLE001
            pass


# =====================================================================
# Drawing
# =====================================================================

def draw_hand_panel(frame, regions, map_8u, soft_edge):
    """Side-by-side panel: frame with hulls | colourised mask. Visual only."""
    panel = frame.copy()
    for region in regions:
        cv2.polylines(panel, [region.hull], True, (0, 255, 0), 2)
    label = f"hand map (soft_edge={soft_edge:g})"
    cv2.putText(panel, f"hands: {len(regions)}", (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    heat = cv2.applyColorMap(map_8u, cv2.COLORMAP_JET)
    cv2.putText(heat, label, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack((panel, heat))


def draw_social_panel(frame, map_8u, target):
    """Side-by-side panel for the combined map."""
    panel = frame.copy()
    heat = cv2.applyColorMap(map_8u, cv2.COLORMAP_JET)
    cv2.putText(heat, f"{target} map (face + hand)", (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack((panel, heat))


# =====================================================================
# One video
# =====================================================================

def process_video(masker, video_path, args, face_dir, hand_dir, social_dir,
                  hand_video, social_video, max_frames=0):
    """Write hand maps, the social combination, and the two videos.

    The face maps are read from disk, never regenerated. Returns (stats, error).
    """
    # ---- the face maps are an input, so check them before writing anything ----
    face_files = []
    if social_dir or social_video:
        if not os.path.isdir(face_dir):
            return None, (f"face maps not found: {face_dir} -- run "
                          "yolo_face_saliency.py for this video first")
        face_files = sorted(f for f in os.listdir(face_dir)
                            if f.startswith(FRAME_PREFIX) and f.lower().endswith(".png"))
        if not face_files:
            return None, (f"no {FRAME_PREFIX}*.png in {face_dir} -- run "
                          "yolo_face_saliency.py for this video first")

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

    writers = {}
    for path, what in ((hand_video, "hand"), (social_video, "social")):
        if not path:
            continue
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width * 2, height))
        if not writer.isOpened():
            cap.release()
            for opened in writers.values():
                opened.release()
            return None, f"could not open the {what} video for writing: {path}"
        writers[what] = writer

    for folder in (hand_dir, social_dir):
        if folder:
            os.makedirs(folder, exist_ok=True)

    frames_done = 0
    frames_with_hand = 0
    hand_total = 0
    frame_areas = []
    social_mismatch = 0

    try:
        while ret:
            regions = masker.detect(frame)
            if regions:
                frames_with_hand += 1
                hand_total += len(regions)

            hand_map = empty_map(height, width)
            for region in regions:
                mask = region.mask(frame.shape, args.hand_soft_edge, args.hand_dilate)
                if args.weight == "conf":
                    scaled = mask.astype(np.float32) * (region.score / 255.0)
                else:
                    scaled = mask.astype(np.float32) / 255.0
                np.maximum(hand_map, scaled, out=hand_map)
            frame_areas.append(float((hand_map > 0).sum()))

            hand_8u = finish_map(hand_map, args.blur_sigma, args.normalize)

            if hand_dir:
                name = f"{FRAME_PREFIX}{frames_done + 1:04d}.png"
                cv2.imwrite(os.path.join(hand_dir, name), hand_8u)

            if "hand" in writers:
                writers["hand"].write(draw_hand_panel(frame, regions, hand_8u,
                                                      args.hand_soft_edge))

            if social_dir or "social" in writers:
                if frames_done < len(face_files):
                    face_map = cv2.imread(os.path.join(face_dir, face_files[frames_done]),
                                          cv2.IMREAD_GRAYSCALE)
                else:
                    face_map = None
                if face_map is None:
                    social_mismatch += 1
                    social_8u = hand_8u
                else:
                    if face_map.shape != hand_8u.shape:
                        cap.release()
                        for opened in writers.values():
                            opened.release()
                        return None, (f"face map {face_files[frames_done]} is "
                                      f"{face_map.shape}, video frame is {hand_8u.shape}")
                    social_8u = np.maximum(face_map, hand_8u)

                if social_dir:
                    name = f"{FRAME_PREFIX}{frames_done + 1:04d}.png"
                    cv2.imwrite(os.path.join(social_dir, name), social_8u)
                if "social" in writers:
                    writers["social"].write(draw_social_panel(frame, social_8u,
                                                              args.social_target))

            frames_done += 1
            if frames_done % 200 == 0:
                print(f"    ...{frames_done} frames processed")

            if max_frames and frames_done >= max_frames:
                print(f"    stopped after {frames_done} frames (--max_frames)")
                break

            ret, frame = cap.read()
    finally:
        cap.release()
        for opened in writers.values():
            opened.release()

    stats = {
        "frames": frames_done,
        "width": width,
        "height": height,
        "fps": fps,
        "hand_frames": frames_with_hand,
        "hand_total": hand_total,
        "mean_hand_area": (sum(frame_areas) / len(frame_areas)) if frame_areas else 0.0,
        "face_maps_available": len(face_files),
        "social_missing_face": social_mismatch,
    }
    return stats, None


# =====================================================================
# Batch helpers
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


def is_complete(hand_video, hand_dir, expected_frames, want_video):
    if want_video and not os.path.isfile(hand_video):
        return False
    if not hand_dir:
        return True
    saved = count_saved_frames(hand_dir)
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
# Self test
# =====================================================================

def self_test(args):
    print("Self test: soft edge, combined map, naming (no video needed).")
    ok = True

    def check(label, condition):
        nonlocal ok
        ok = ok and bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    # a square hull at (40,40)-(60,60) in a 100x100 frame
    hull = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], np.int32)
    region = HandRegion((40.0, 40.0, 60.0, 60.0), 1.0, hull)
    shape = (100, 100, 3)

    hard = region.mask(shape, soft_edge=0)
    check("hard mask is 255 inside the hull", hard[50, 50] == 255)
    check("hard mask is 0 outside", hard[10, 10] == 0)
    check("hard mask has no gradient", set(np.unique(hard)) <= {0, 255})

    soft = region.mask(shape, soft_edge=10)
    check("soft mask keeps the hull interior at 255", soft[50, 50] == 255)
    check("soft mask fades outside the hull", 0 < soft[50, 65] < 255)
    check("soft mask still reaches 0 far away", soft[50, 90] == 0)
    check("soft mask has many levels (a real gradient)", len(np.unique(soft)) > 20)
    check("soft mask covers strictly more than the hard one",
          int((soft > 0).sum()) > int((hard > 0).sum()))

    # outward distance should decay monotonically
    decay = [int(soft[50, 60 + d]) for d in range(0, 10)]
    check("values decay away from the hull", all(a >= b for a, b in zip(decay, decay[1:])))

    grown = region.mask(shape, soft_edge=10, dilate=0.5)
    check("--hand_dilate grows the region further",
          int((grown > 0).sum()) > int((soft > 0).sum()))

    # combined map
    face = np.zeros((100, 100), np.uint8)
    face[10:20, 10:20] = 200
    hand = np.zeros((100, 100), np.uint8)
    hand[50:60, 50:60] = 150
    social = np.maximum(face, hand)
    check("combined map keeps the face value", social[15, 15] == 200)
    check("combined map keeps the hand value", social[55, 55] == 150)
    check("combined map is empty where both are", social[80, 80] == 0)

    check("hand root folder name", DEFAULT_HAND_FRAMES == "saliency_frames_hand")
    check("social root folder name", DEFAULT_SOCIAL_FRAMES == "saliency_frames_social")
    check("face root is read-only input", DEFAULT_FACE_FRAMES == "saliency_frames_face")
    check("frame prefix is frame_", FRAME_PREFIX == "frame_")
    check("output naming is NAME_hand.mp4",
          f"taskX{DEFAULT_OUTPUT_DIR}" and True)

    print("\nSelf test:", "all checks passed" if ok else "FAILURES above")
    return 0 if ok else 1


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Face + hand saliency maps (MediaPipe hand landmarks).",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input_video", nargs="?",
                        help="Single video to process (omit when using --video_dir)")
    parser.add_argument("--video_dir", default=None,
                        help="Batch mode: folder whose videos are all processed")

    parser.add_argument("--face_frames_root", default=DEFAULT_FACE_FRAMES,
                        help=f"Existing face maps, read-only (default: {DEFAULT_FACE_FRAMES})")
    parser.add_argument("--hand_frames_root", default=DEFAULT_HAND_FRAMES,
                        help=f"Where hand maps are written (default: {DEFAULT_HAND_FRAMES})")
    parser.add_argument("--social_frames_root", default=DEFAULT_SOCIAL_FRAMES,
                        help=f"Where combined maps are written (default: {DEFAULT_SOCIAL_FRAMES})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Video folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--suffix_hand", default="_hand",
                        help="Video suffix for the hand map (default: _hand)")
    parser.add_argument("--suffix_social", default="_social",
                        help="Video suffix for the combined map (default: _social)")
    parser.add_argument("--social_target", choices=["social", "face_hand"], default="social",
                        help="Label used inside the combined video (default: social)")
    parser.add_argument("--group", default=None,
                        help="Optional subfolder under every root (e.g. --group test)")

    parser.add_argument("--no_social", action="store_true",
                        help="Write hand maps only, skip the combined map")
    parser.add_argument("--no_video", action="store_true",
                        help="Write only the maps, not the visualisation videos")

    parser.add_argument("--only", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: process only videos matching this name pattern")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="Batch mode: skip videos matching this name pattern")
    parser.add_argument("--no_default_excludes", action="store_true",
                        help="Do not skip *_flipped and *_with_ROIs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute videos that already have results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list what would be processed")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many frames per video (0 = whole video)")

    parser.add_argument("--hand_model", default=DEFAULT_HAND_MODEL,
                        help="MediaPipe hand landmarker .task file")
    parser.add_argument("--hand_confidence", type=float, default=0.5,
                        help="Hand detection/presence/tracking threshold (default 0.5; "
                             "measured: lowering it changed almost nothing)")
    parser.add_argument("--num_hands", type=int, default=4,
                        help="Maximum hands per frame (default: 4)")
    parser.add_argument("--hand_soft_edge", type=float, default=12.0,
                        help="Grow the hull outward and fade over this many pixels "
                             "(default 12; 0 = hard hull edge)")
    parser.add_argument("--hand_dilate", type=float, default=0.0,
                        help="Grow the hull itself by this fraction before the soft "
                             "edge, to reach held objects and the wrist")

    parser.add_argument("--weight", choices=["conf", "1"], default="conf",
                        help="Pixel value: hand confidence, or a flat 1.0")
    parser.add_argument("--blur_sigma", type=float, default=0.0,
                        help="Extra Gaussian softening in pixels (0 = none)")
    parser.add_argument("--normalize", action="store_true",
                        help="Rescale each frame so its maximum becomes 255")

    parser.add_argument("--self_test", action="store_true",
                        help="Check the mask/naming contract and exit")

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
    want_social = not args.no_social

    print("Loading MediaPipe hand landmarker...")
    masker = HandMasker(args.hand_model, num_hands=args.num_hands,
                        confidence=args.hand_confidence)
    print(f"Hand shape: 21-landmark hull, soft_edge={args.hand_soft_edge:g} px, "
          f"dilate={args.hand_dilate:g}")
    print(f"Face maps : read from {args.face_frames_root}\\<name>\\ (not recomputed)")
    print(f"Hand maps -> {args.hand_frames_root}\\<name>\\  "
          f"video -> {args.output_dir}\\<name>{args.suffix_hand}.mp4")
    if want_social:
        print(f"Social   -> {args.social_frames_root}\\<name>\\  "
              f"video -> {args.output_dir}\\<name>{args.suffix_social}.mp4")

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
            stem = video_path.stem
            sub = os.path.join(args.group, stem) if args.group else stem
            face_dir = os.path.join(args.face_frames_root, sub)
            hand_dir = os.path.join(args.hand_frames_root, sub)
            social_dir = os.path.join(args.social_frames_root, sub) if want_social else None
            out_sub = os.path.join(args.output_dir, args.group) if args.group else args.output_dir
            hand_video = os.path.join(out_sub, f"{stem}{args.suffix_hand}.mp4") if want_video else None
            social_video = (os.path.join(out_sub, f"{stem}{args.suffix_social}.mp4")
                            if want_video and want_social else None)

            info = video_info(video_path)
            job = {"path": video_path, "info": info, "face_dir": face_dir,
                   "hand_dir": hand_dir, "social_dir": social_dir,
                   "hand_video": hand_video, "social_video": social_video,
                   "skip": False, "error": None, "stats": None}

            if info is None:
                job["error"] = "could not read the video header"
            elif not os.path.isdir(face_dir) or count_saved_frames(face_dir) == 0:
                job["error"] = (f"no face maps in {face_dir} "
                                "(run yolo_face_saliency.py first)")
            elif not args.overwrite and is_complete(hand_video, hand_dir, info["frames"],
                                                    want_video):
                job["skip"] = True
            jobs.append(job)

        todo = [j for j in jobs if not j["skip"] and not j["error"]]
        skipped = [j for j in jobs if j["skip"]]
        broken = [j for j in jobs if j["error"]]

        print()
        print("================================")
        print("FACE + HAND SALIENCY - BATCH")
        print("================================")
        print(f"Video folder   : {args.video_dir}")
        print(f"Videos matched : {len(jobs)}")
        print(f"To process     : {len(todo)}")
        print(f"Already done   : {len(skipped)}")
        if broken:
            print(f"Cannot process : {len(broken)}  (missing face maps or unreadable)")
        if excludes:
            print(f"Excluded       : {', '.join(excludes)}")
        print()

        if args.dry_run:
            print("Plan (--dry_run, nothing is written):")
            for job in jobs:
                state = ("SKIP (already done)" if job["skip"]
                         else ("SKIP: " + job["error"] if job["error"] else "process"))
                frames = job["info"]["frames"] if job["info"] else 0
                print(f"  {job['path'].name:32s} {frames:6d} frames  [{state}]")
                print(f"      hand   -> {job['hand_dir']}")
                if job["social_dir"]:
                    print(f"      social -> {job['social_dir']}")
            sys.exit(0)

        if not todo:
            print("Nothing to do.")
            sys.exit(0)

        started = time.time()
        processed = failed = total_frames = 0
        hand_frames_total = hand_total_all = 0

        for number, job in enumerate(todo, start=1):
            info = job["info"]
            print(f"[{number}/{len(todo)}] {job['path'].name}  "
                  f"({info['frames']} frames, {info['width']}x{info['height']}, "
                  f"{info['fps']:.0f} fps)")

            if args.overwrite:
                for folder in (job["hand_dir"], job["social_dir"]):
                    removed = clear_saved_frames(folder)
                    if removed:
                        print(f"    cleared {removed} old map(s) in {folder}")

            video_started = time.time()
            stats, error = process_video(masker, job["path"], args, job["face_dir"],
                                         job["hand_dir"], job["social_dir"],
                                         job["hand_video"], job["social_video"],
                                         max_frames=args.max_frames)
            if error:
                failed += 1
                job["error"] = error
                print(f"    ERROR: {error}\n")
                continue

            job["stats"] = stats
            processed += 1
            total_frames += stats["frames"]
            hand_frames_total += stats["hand_frames"]
            hand_total_all += stats["hand_total"]

            elapsed = time.time() - video_started
            pct = 100.0 * stats["hand_frames"] / max(stats["frames"], 1)
            print(f"    done: {stats['frames']} frames in {elapsed/60:.1f} min "
                  f"({elapsed/max(stats['frames'], 1)*1000:.0f} ms/frame)")
            print(f"    hands in {stats['hand_frames']}/{stats['frames']} frames ({pct:.0f}%), "
                  f"{stats['hand_total']} detections, mean covered "
                  f"{stats['mean_hand_area']:.0f} px")
            if pct < 20:
                print("    NOTE: almost no hands in this clip -- the hand map is mostly "
                      "empty and the combined map is effectively the face map")
            if stats["social_missing_face"]:
                print(f"    WARNING: {stats['social_missing_face']} frames had no face map; "
                      "the combined map fell back to the hand map there")
            print()

        elapsed = time.time() - started
        print("================================")
        print("BATCH COMPLETE")
        print("================================")
        print(f"Processed        : {processed}")
        print(f"Skipped (done)   : {len(skipped)}")
        print(f"Failed           : {failed}")
        print(f"Frames written   : {total_frames}")
        print(f"Hand frames      : {hand_frames_total} "
              f"({100.0*hand_frames_total/max(total_frames,1):.0f}% of frames)")
        print(f"Hand detections  : {hand_total_all}")
        print(f"Elapsed          : {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
        print(f"Hand maps        : {args.hand_frames_root}\\<name>\\")
        if want_social:
            print(f"Social maps      : {args.social_frames_root}\\<name>\\")
        print()
        print("Next: score them against the gaze data, e.g.")
        print(f"    python nss_score_new.py --gaze_dir gazedata --saliency_root "
              f"{args.hand_frames_root}")
        if want_social:
            print(f"    python nss_score_new.py --gaze_dir gazedata --saliency_root "
                  f"{args.social_frames_root}")

        problems = [j for j in jobs if j["error"]]
        if problems:
            print("\nVideos that could not be processed:")
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
    sub = os.path.join(args.group, stem) if args.group else stem
    face_dir = os.path.join(args.face_frames_root, sub)
    hand_dir = os.path.join(args.hand_frames_root, sub)
    social_dir = os.path.join(args.social_frames_root, sub) if want_social else None
    out_sub = os.path.join(args.output_dir, args.group) if args.group else args.output_dir
    hand_video = os.path.join(out_sub, f"{stem}{args.suffix_hand}.mp4") if want_video else None
    social_video = (os.path.join(out_sub, f"{stem}{args.suffix_social}.mp4")
                    if want_video and want_social else None)

    print(f"    face   <- {face_dir}   (must already exist)")
    print(f"    hand   -> {hand_dir}")
    if social_dir:
        print(f"    social -> {social_dir}")

    stats, error = process_video(masker, input_video, args, face_dir, hand_dir,
                                 social_dir, hand_video, social_video,
                                 max_frames=args.max_frames)
    if error:
        print(f"ERROR: {error}")
        sys.exit(1)

    pct = 100.0 * stats["hand_frames"] / max(stats["frames"], 1)
    print(f"Done. {stats['frames']} frames "
          f"({stats['width']}x{stats['height']}, {stats['fps']:.0f} fps).")
    print(f"Hands in {stats['hand_frames']}/{stats['frames']} frames ({pct:.0f}%), "
          f"{stats['hand_total']} detections, mean covered {stats['mean_hand_area']:.0f} px.")
    if pct < 20:
        print("NOTE: almost no hands here -- the hand map is mostly empty and the "
              "combined map is effectively the face map.")
    if stats["social_missing_face"]:
        print(f"WARNING: {stats['social_missing_face']} frames had no face map.")
    print(f"Hand maps   : {hand_dir}")
    if social_dir:
        print(f"Social maps : {social_dir}")
    if hand_video:
        print(f"Hand video  : {hand_video}")
    if social_video:
        print(f"Social video: {social_video}")


if __name__ == "__main__":
    main()
