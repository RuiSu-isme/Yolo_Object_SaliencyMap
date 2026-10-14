"""
Shared engine for face/hand saliency maps.

This module holds the parts that both map generators need -- the YuNet and
MediaPipe face detectors, the optional hand detector, the shape painters
and the debug renderer:

    yolo_face_hand_saliency.py   configurable generator (box or mesh, +hands)
    yolo_head_saliency.py        head-only generator with the study's
                                 folder layout baked in

Keeping one copy here means the two scripts cannot drift apart. Nothing in
this file knows about argv or about output folders; that is the callers'
business.

Dependencies: opencv-python (YuNet) always; the vendored mediapipe in
_vendor/ only for mesh shapes; ultralytics only if a hand model is used.
"""

import os
import sys

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
FRAME_PREFIX = "frame_"

CORE_HERE = os.path.dirname(os.path.abspath(__file__))

# Assets (the ONNX/task models and the vendored mediapipe) are looked up in
# this order, so a script can live in a working folder without carrying 63 MB
# of dependencies along:
#   1. an explicit --face_model/--mesh_model/--hair_model argument
#   2. $FACE_HAND_HOME/models and $FACE_HAND_HOME/_vendor
#   3. models/ and _vendor/ next to this module
#   4. models/ and _vendor/ next to the script that imported this module
ASSET_HOME_ENV = "FACE_HAND_HOME"
ASSET_HOME = os.environ.get(ASSET_HOME_ENV, "").strip()

# set by a caller that lives outside this folder (see note in asset_path)
CALLER_HERE = CORE_HERE


def set_caller_dir(path):
    """Tell the core where the importing script lives, for asset lookup."""
    global CALLER_HERE
    if path:
        CALLER_HERE = os.path.abspath(path)


def _search_roots():
    roots = []
    if ASSET_HOME:
        roots.append(ASSET_HOME)
    roots.append(CORE_HERE)
    if CALLER_HERE not in roots:
        roots.append(CALLER_HERE)
    return roots


def asset_path(filename):
    """Locate one asset file, honouring $FACE_HAND_HOME then the folder layout.

    A model passed explicitly on the command line never goes through here,
    so an explicit argument always wins.
    """
    for root in _search_roots():
        candidate = os.path.join(root, "models", filename)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(CORE_HERE, "models", filename)


def vendor_dir():
    """The _vendor folder to add to sys.path, or None if there is none."""
    for root in _search_roots():
        candidate = os.path.join(root, "_vendor")
        if os.path.isdir(candidate):
            return candidate
    return None


DEFAULT_FACE_MODEL = asset_path("face_detection_yunet_2023mar.onnx")
DEFAULT_MESH_MODEL = asset_path("face_landmarker.task")
DEFAULT_HAIR_MODEL = asset_path("hair_segmenter.tflite")

FACE_LANDMARK_NAMES = ("right eye", "left eye", "nose", "right mouth", "left mouth")

MASK_SHAPES = ("box", "mesh")


# =====================================================================
# Face regions: one painted area each, box-shaped or face-shaped
# =====================================================================

class FaceRegion:
    """One detected face, able to paint itself into a map.

    Box regions come from YuNet (a rectangle). Mesh regions come from
    MediaPipe FaceLandmarker (the convex hull of 478 landmarks, i.e. the
    actual facial outline). Both expose paint() so the frame loop does not
    care which one it is holding.
    """

    __slots__ = ("box", "score", "kind", "hull", "hair_mask")

    def __init__(self, box, score, kind, hull=None, hair_mask=None):
        self.box = box            # (x1, y1, x2, y2)
        self.score = float(score)
        self.kind = kind          # "box" | "mesh"
        self.hull = hull          # (N, 2) int32 landmark hull, mesh only
        self.hair_mask = hair_mask  # uint8 0/1 full-frame mask, optional

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def paint(self, map_, value):
        """Set the strongest value into every pixel covered by this region."""
        height, width = map_.shape[:2]
        if self.hull is not None:
            shape_mask = np.zeros((height, width), np.uint8)
            cv2.drawContours(shape_mask, [self.hull], -1, 1, cv2.FILLED)
            if self.hair_mask is not None:
                shape_mask = np.maximum(shape_mask, self.hair_mask)
            covered = shape_mask > 0
            if covered.any():
                painted = map_[covered]
                np.maximum(painted, np.float32(value), out=painted)
                map_[covered] = painted
        else:
            paint(map_, (self.box[0], self.box[1], self.box[2], self.box[3], value), value)



# =====================================================================
# Detectors
# =====================================================================

class FaceDetector:
    """Thin wrapper around cv2.FaceDetectorYN."""

    def __init__(self, model_path, det_size=640, score_threshold=0.5, nms_threshold=0.3, top_k=5000):
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"ERROR: YuNet face model not found: {model_path}\n"
                "Download face_detection_yunet_2023mar.onnx (232 KB) into the models/ folder:\n"
                "    https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet\n"
                "(the raw URL serves a git-lfs pointer -- use the 'Download raw file' button,\n"
                " or any mirror that serves the real 232,589-byte file).\n"
                f"Or set {ASSET_HOME_ENV} to an existing asset folder (the one holding\n"
                "models/ and _vendor/), or pass --face_model explicitly.")

        if not hasattr(cv2, "FaceDetectorYN"):
            raise SystemExit(
                "ERROR: this OpenCV build has no cv2.FaceDetectorYN.\n"
                "Install opencv-python >= 4.5.4 (or opencv-contrib-python).")

        self.det_size = (int(det_size), int(det_size))
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", self.det_size, float(score_threshold), float(nms_threshold), int(top_k))
        self.calls = 0

    def detect(self, frame):
        """Return an (N, 15) float array, or an empty (0, 15) array.

        Columns: x, y, w, h, 5x (x, y) landmarks, score.
        """
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        # YuNet was trained on BGR, which is what cv2 gives us -- no swap.
        _, faces = self.detector.detect(frame)
        self.calls += 1
        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)
        return np.asarray(faces, dtype=np.float32)


class HandDetector:
    """Pluggable hand detector: a YOLO detection model OR a YOLO pose model.

    Both are optional. is_available() is False when no --hand_model was
    given or ultralytics is missing, and the caller then produces
    face-only maps.
    """

    def __init__(self, model_path, conf=0.25, box_scale=1.0, hand_class_filter="hand"):
        self.available = False
        self.reason = ""
        self.model = None
        self.hand_classes = set()
        self.is_pose = False
        self.conf = float(conf)
        self.box_scale = float(box_scale)
        self.filter = hand_class_filter.lower()

        if not model_path:
            self.reason = "no --hand_model given"
            return

        if not os.path.isfile(model_path):
            self.reason = f"hand model file not found: {model_path}"
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            self.reason = "the 'ultralytics' package is not installed (pip install ultralytics)"
            return

        try:
            self.model = YOLO(model_path)
        except Exception as exc:                                  # noqa: BLE001
            self.reason = f"could not load the hand model: {exc}"
            return

        names = getattr(self.model, "names", {}) or {}
        self.is_pose = "pose" in str(getattr(self.model, "task", "")).lower()

        for index, name in names.items():
            if self.filter in str(name).lower():
                self.hand_classes.add(int(index))

        if not self.hand_classes:
            self.reason = (f"the model has no class whose name contains '{self.filter}' "
                           f"(classes: {list(names.values())})")
            return

        self.available = True
        kind = "pose" if self.is_pose else "detection"
        print(f"Hand detector: {os.path.basename(model_path)} ({kind}, "
              f"classes {sorted(names[i] for i in self.hand_classes)})")

    def detect(self, frame):
        """Return a list of (x1, y1, x2, y2, score) hand boxes."""
        if not self.available:
            return []

        result = self.model(frame, verbose=False, conf=self.conf)[0]
        boxes = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            keypoints = None
            if self.is_pose and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()

            for index in range(len(xyxy)):
                if classes[index] not in self.hand_classes:
                    continue

                if keypoints is not None and index < len(keypoints):
                    box = self._box_from_keypoints(keypoints[index], frame.shape)
                    if box is None:
                        continue
                    x1, y1, x2, y2, kpt_conf = box
                    boxes.append((x1, y1, x2, y2, float(confs[index]) * kpt_conf))
                else:
                    x1, y1, x2, y2 = xyxy[index]
                    boxes.append((float(x1), float(y1), float(x2), float(y2), float(confs[index])))

        return [self._scale_box(box, frame.shape) for box in boxes]

    def _box_from_keypoints(self, keypoints, shape, min_keypoints=3, keypoint_conf=0.3):
        """Bounding box of the confident keypoints, padded to a wrist-sized box.

        min_keypoints is deliberately low: a hand half outside the frame or
        partly behind an object still deserves a box, and a missing hand is
        worse for the analysis than a slightly loose box.
        """
        height, width = shape[:2]
        usable = keypoints[keypoints[:, 2] > keypoint_conf]
        if len(usable) < min_keypoints:
            return None

        xs, ys = usable[:, 0], usable[:, 1]
        # a bare keypoint hull is too tight; hands need margin around the fingers
        pad = 0.35 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))

        x1 = max(0.0, float(xs.min()) - pad)
        y1 = max(0.0, float(ys.min()) - pad)
        x2 = min(float(width - 1), float(xs.max()) + pad)
        y2 = min(float(height - 1), float(ys.max()) + pad)
        return x1, y1, x2, y2, float(usable[:, 2].mean())

    def _scale_box(self, box, shape):
        if self.box_scale == 1.0:
            return box

        height, width = shape[:2]
        x1, y1, x2, y2, score = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_w = (x2 - x1) / 2.0 * self.box_scale
        half_h = (y2 - y1) / 2.0 * self.box_scale
        return (max(0.0, cx - half_w), max(0.0, cy - half_h),
                min(float(width - 1), cx + half_w), min(float(height - 1), cy + half_h), score)


class FaceMeshSegmenter:
    """Face-shaped masks from MediaPipe FaceLandmarker (478 landmarks).

    The mask is the convex hull of the landmarks: the facial oval, which
    follows the real outline instead of a rectangle's four corners. The
    hull does NOT include hair -- MediaPipe has no hair landmarks. Pass
    include_hair to union in the hair segmenter output for the whole head
    (about 15x slower, and the hair model is unreliable on some subjects).

    Detection is a HYBRID on purpose. MediaPipe's own face detector only
    finds large, near-frontal faces: on our stimulus set it scored 100% on
    the close-up clips and 0% on every srt clip, even at confidence 0.15,
    where YuNet finds 65-92%. So YuNet supplies the hints and MediaPipe
    only measures the shape inside each hinted crop. Frames where YuNet
    sees a face but MediaPipe cannot fit landmarks fall back to the box
    (self.box_fallbacks), so recall never drops below box mode.

    mediapipe is imported lazily so the box path keeps working with no
    vendored dependencies at all.
    """

    def __init__(self, model_path, face_detector, include_hair=False, hair_model_path=None,
                 num_faces=5, min_confidence=0.5, dilate=0.0, crop_pad=0.25, box_scale=1.0):
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"ERROR: MediaPipe face landmarker model not found: {model_path}\n"
                "Either download face_landmarker.task (3.8 MB) into models/:\n"
                "    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task\n"
                f"or set {ASSET_HOME_ENV} to an existing asset folder (the one holding\n"
                "models/ and _vendor/), or pass --mesh_model explicitly.\n"
                "Box mode, by contrast, needs only the 227 KB YuNet ONNX.")

        found_vendor = vendor_dir()
        if found_vendor and found_vendor not in sys.path:
            sys.path.insert(0, found_vendor)

        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise SystemExit(
                "ERROR: --mask_shape mesh needs mediapipe, which is not importable:\n"
                f"    {exc}\n"
                "This environment blocks pip, so either:\n"
                f"  1. set {ASSET_HOME_ENV} to the folder that holds _vendor and models, e.g.\n"
                f"       set {ASSET_HOME_ENV}=D:\\path\\to\\Yolo\n"
                "  2. or copy _vendor next to this script\n"
                "  3. or use --mask_shape box, which needs nothing but OpenCV.")

        self.vision = vision
        self.face_detector = face_detector
        self.dilate = float(dilate)
        self.crop_pad = float(crop_pad)
        self.box_scale = float(box_scale)
        self.include_hair = bool(include_hair)
        self.num_faces = int(num_faces)

        # counting why a frame fell back, so a silently box-shaped result
        # cannot be mistaken for a mesh
        self.crop_hits = 0
        self.fullframe_hits = 0
        self.box_fallbacks = 0

        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.num_faces,
                min_face_detection_confidence=float(min_confidence),
                min_face_presence_confidence=float(min_confidence),
                min_tracking_confidence=float(min_confidence)))

        self.segmenter = None
        if self.include_hair:
            if not hair_model_path or not os.path.isfile(hair_model_path):
                raise SystemExit(
                    f"ERROR: --head_hair needs the hair segmenter model: {hair_model_path}\n"
                    "Either download hair_segmenter.tflite (780 KB) into models/:\n"
                    "    https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                    "hair_segmenter/float32/1/hair_segmenter.tflite\n"
                    f"or set {ASSET_HOME_ENV} to an existing asset folder, "
                    "or pass --hair_model explicitly.")
            self.segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=hair_model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    output_category_mask=True))

    def _as_mp_image(self, rgb):
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        return Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def detect(self, frame):
        """Return a list of mesh FaceRegions for this frame.

        YuNet proposes the faces; MediaPipe fits the outline inside each
        proposal, falling back to the rectangle when it cannot.
        """
        faces = self.face_detector.detect(frame)
        if not len(faces):
            return [], faces

        hair_mask = None
        if self.segmenter is not None:
            hair_mask = self._hair_mask(frame)

        height, width = frame.shape[:2]
        regions = []
        for row, box in zip(faces, face_boxes(faces, self.box_scale)):
            hint = (box[0], box[1], box[2], box[3])
            hull = self._hull_for_box(frame, hint)
            if hull is None:
                self.box_fallbacks += 1
                regions.append(FaceRegion(hint, float(row[14]), "box"))
                continue

            if self.dilate:
                hull = self._dilate_hull(hull, self.dilate, width, height)
            hull = np.clip(hull, [0, 0], [width - 1, height - 1]).astype(np.int32)
            regions.append(FaceRegion(hint, float(row[14]), "mesh", hull=hull,
                                      hair_mask=hair_mask))

        return regions, faces

    def _hull_for_box(self, frame, box):
        """Fit landmarks inside one YuNet box; None when no face is found."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * self.crop_pad
        pad_y = (y2 - y1) * self.crop_pad
        left = int(max(0, np.floor(x1 - pad_x)))
        top = int(max(0, np.floor(y1 - pad_y)))
        right = int(min(width, np.ceil(x2 + pad_x)))
        bottom = int(min(height, np.ceil(y2 + pad_y)))
        if right - left < 8 or bottom - top < 8:
            return None

        crop = frame[top:bottom, left:right]
        found = self._fit(crop)
        if found is not None:
            self.crop_hits += 1
            return found + np.array([left, top], dtype=np.float32)

        # second chance: the full frame sometimes succeeds where the crop
        # does not (MediaPipe scales its own detector input)
        found = self._fit(frame)
        if found is not None:
            self.fullframe_hits += 1
            return found
        return None

    def _fit(self, image_bgr):
        """Convex hull of the largest face in this image, or None."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(self._as_mp_image(rgb))
        if not result.face_landmarks:
            return None

        height, width = image_bgr.shape[:2]
        hulls = []
        for keypoints in result.face_landmarks:
            points = np.array([[kp.x * width, kp.y * height] for kp in keypoints], dtype=np.float32)
            hulls.append(cv2.convexHull(points.astype(np.int32)).reshape(-1, 2).astype(np.float32))
        return max(hulls, key=cv2.contourArea)

    def _hair_mask(self, frame):
        """Hair pixels, as a full-frame 0/1 uint8 mask."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(self._as_mp_image(rgb))
        raw = np.squeeze(result.category_mask.numpy_view())
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        return (np.asarray(raw) > 0.5).astype(np.uint8)

    @staticmethod
    def _dilate_hull(hull, factor, width, height):
        """Grow the hull about its centre (factor is a fraction of its size)."""
        center, _, _ = cv2.minAreaRect(hull.astype(np.float32))
        cx, cy = center
        grown = np.empty_like(hull, dtype=np.float32)
        grown[:, 0] = cx + (hull[:, 0] - cx) * (1.0 + factor)
        grown[:, 1] = cy + (hull[:, 1] - cy) * (1.0 + factor)
        grown[:, 0] = np.clip(grown[:, 0], 0, width - 1)
        grown[:, 1] = np.clip(grown[:, 1], 0, height - 1)
        return cv2.convexHull(grown.astype(np.int32)).reshape(-1, 2)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:                                      # noqa: BLE001
            pass
        if self.segmenter is not None:
            try:
                self.segmenter.close()
            except Exception:                                  # noqa: BLE001
                pass




# =====================================================================
# Map building
# =====================================================================

def empty_map(height, width):
    return np.zeros((height, width), dtype=np.float32)


def paint(map_, box, value):
    """Set every pixel inside the box to max(current, value)."""
    x1, y1, x2, y2, score = box
    if x2 <= x1 or y2 <= y1:
        return

    height, width = map_.shape[:2]
    left = int(max(0, np.floor(x1)))
    top = int(max(0, np.floor(y1)))
    right = int(min(width, np.ceil(x2) + 1))
    bottom = int(min(height, np.ceil(y2) + 1))
    if right <= left or bottom <= top:
        return

    np.maximum(map_[top:bottom, left:right], np.float32(value), out=map_[top:bottom, left:right])


def face_boxes(faces, box_scale=1.0):
    """YuNet rows -> (x1, y1, x2, y2, score) boxes."""
    boxes = []
    for row in faces:
        confidence = float(row[14])
        x1, y1, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        half_w, half_h = w / 2.0 * box_scale, h / 2.0 * box_scale
        boxes.append((max(0.0, cx - half_w), max(0.0, cy - half_h),
                      cx + half_w, cy + half_h, confidence))
    return boxes


def finish_map(map_, blur_sigma, normalize):
    """float map -> the 8-bit PNG the scoring scripts read."""
    if blur_sigma and blur_sigma > 0:
        # kernel is odd and ~6 sigma wide, matching the usual convention
        radius = max(1, int(round(3 * blur_sigma)))
        kernel = 2 * radius + 1
        map_ = cv2.GaussianBlur(map_, (kernel, kernel), float(blur_sigma), borderType=cv2.BORDER_CONSTANT)

    if normalize:
        peak = float(map_.max())
        if peak > 0:
            map_ = map_ / peak

    return (np.clip(map_, 0.0, 1.0) * 255.0).astype(np.uint8)


def faces_for(frame, face_detector, mesh_segmenter, mask_shape, box_scale=1.0):
    """Detect faces for one frame in whichever shape mode is selected.

    Returns (regions, yunet_rows). The YuNet rows are kept only so the
    debug video can draw the five landmarks in box mode and the hint boxes
    in mesh mode.
    """
    if mask_shape == "mesh":
        return mesh_segmenter.detect(frame)

    faces = face_detector.detect(frame)
    regions = [FaceRegion(box[:4], box[4], "box") for box in face_boxes(faces, box_scale)]
    return regions, faces


def draw_debug(frame, regions, hands, map_8u, weight_mode, mask_shape, yunet_rows=None):
    """Side-by-side panel: annotated frame | colored map. For eyeballing only.

    regions are the areas actually painted, so the debug video always shows
    the geometry that produced the map.
    """
    panel = frame.copy()

    for region in regions:
        x1, y1, x2, y2 = (int(round(v)) for v in region.box)
        if region.hull is not None:
            # the YuNet hint box, faint: shows what the mesh was fitted into
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.polylines(panel, [region.hull], True, (0, 255, 0), 2)
            if region.hair_mask is not None:
                overlay = panel.copy()
                overlay[region.hair_mask > 0] = (0, 200, 255)
                panel = cv2.addWeighted(panel, 0.6, overlay, 0.4, 0)
            label = "face mesh"
        else:
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"face {region.score:.2f}"
            if mask_shape == "mesh":
                label += " (box fallback)"
        cv2.putText(panel, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if yunet_rows is not None:
        for row in yunet_rows:
            for point in range(5):
                px, py = int(row[4 + point * 2]), int(row[5 + point * 2])
                cv2.circle(panel, (px, py), 2, (0, 255, 255), -1)

    for box in hands:
        x1, y1, x2, y2, score = (int(round(v)) for v in box)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(panel, f"hand {score:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)

    heat = cv2.applyColorMap(map_8u, cv2.COLORMAP_JET)
    label = f"face+hand ROI map  (shape={mask_shape}, weight={weight_mode})"
    cv2.putText(heat, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack((panel, heat))
