# Saliency maps for gaze analysis: objects, faces and hands

This document covers the **map-building half** of the pipeline: how a stimulus
video becomes a saliency map. The scoring half — NSS, AUC-Judd, the gaze model,
how to read the output CSVs — is documented in **[README.md](README.md)**.

Four kinds of map are supported, and all four share one output contract, so the
same scoring code compares them:

| Map | Region | Method | Script |
|---|---|---|---|
| **object** | every COCO object | YOLOv8 instance segmentation | `yolo_heatmap_video.py` |
| **face** | the facial oval | YuNet detection + MediaPipe 478 landmarks | `yolo_face_saliency.py` |
| **hand** | each hand, soft-edged | MediaPipe HandLandmarker (21 landmarks) | `yolo_social_saliency.py` |
| **social** | face ∪ hand | `max(face, hand)` | `yolo_social_saliency.py` |

```
stimulus video ──┬─> saliency_frames\<task>\       + output\<task>.mp4          object
                 ├─> saliency_frames_face\<task>\   + output\<task>_face.mp4     face
                 ├─> saliency_frames_hand\<task>\   + output\<task>_hand.mp4     hand
                 └─> saliency_frames_social\<task>\ + output\<task>_social.mp4   social
                                    │
                                    v
              gaze CSV ──> resample_gaze.py ──> nss_score_new.py / auc_score_new.py
```

---

## Contents

- [Quick start](#quick-start)
- [Which script do I need?](#which-script-do-i-need)
- [Installation](#installation)
  - [Vendored mediapipe](#vendored-mediapipe)
- [Usage](#usage)
- [Output format](#output-format)
- [How each map is built](#how-each-map-is-built)
- [Measured behaviour](#measured-behaviour)
  - [Why the face maps use YuNet, not MediaPipe's FaceLandmarker](#why-the-face-maps-use-yunet-not-mediapipes-facelandmarker)
  - [Hand coverage per stimulus](#hand-coverage-per-stimulus)
  - [Tuning the hand edge](#tuning-the-hand-edge)
  - [Speed and thresholds](#speed-and-thresholds)
- [Model files and licences](#model-files-and-licences)
- [Building a distribution folder for someone else](#building-a-distribution-folder-for-someone-else)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Quick start

For a colleague setting this up on their own machine — the complete face + hand
pipeline, assuming Python is installed:

```bash
# 1. install the only three packages needed
pip install numpy pandas opencv-python

# 2. verify the output contract without touching any video
python yolo_face_saliency.py --self_test
python yolo_social_saliency.py --self_test

# 3. put your stimulus videos in video\ then build the maps
python yolo_face_saliency.py --video_dir video     # step A: face maps
python yolo_social_saliency.py --video_dir video   # step B: hand + social maps
```

No `mediapipe` install and **no model download** is required: both are contained
in the repository (see [Vendored mediapipe](#vendored-mediapipe)). Step B **reads**
`saliency_frames_face\`, so run step A first — otherwise step B skips every video
with a clear message rather than writing half-finished output.

---

## Which script do I need?

| I want | Run this | Notes |
|---|---|---|
| face maps only | `yolo_face_saliency.py --video_dir video` | needs `models/face_*` |
| hand + social maps | `yolo_social_saliency.py --video_dir video` | the face maps must exist first |
| object maps | `yolo_heatmap_video.py --video_dir video` | needs `ultralytics` + torch |
| a configurable one-off | `yolo_face_hand_saliency.py` | box **or** mesh faces, optional custom YOLO hand model |

`face_hand_core.py` is the shared engine (detectors, mask painting, debug
renderer) imported by the three face/hand scripts. It is not run directly.

---

## Installation

- **Python 3.9+** (tested on 3.14, Windows)
- **`numpy`**, **`pandas`** (scoring), **`opencv-python >= 4.5.4`**

```bash
pip install numpy pandas opencv-python
```

That is the complete list for face/hand maps. `mediapipe` is vendored (below);
`torch` is only needed for the *object* maps, which additionally need
`ultralytics` (see [README.md](README.md#requirements)).

> **Do not install `opencv-contrib-python` alongside `opencv-python`.** It
> shadows `cv2` and removes `cv2.FaceDetectorYN`, which the face maps depend on.
> `opencv-python >= 4.5.4` is the requirement — that is when `FaceDetectorYN`
> was added.

### Vendored mediapipe

`_vendor\` contains **mediapipe 1.1.0** plus three small pure-Python dependencies
(`absl-py`, `certifi`, `flatbuffers`), unpacked from their PyPI wheels. At startup
the scripts add that folder to `sys.path`, so `import mediapipe` resolves from the
repository with no installation step.

Two consequences worth knowing:

- The pipeline runs on machines where **pip is blocked** or downloads are not
  possible — which is why it is packaged this way.
- `opencv-contrib-python` is deliberately **not** vendored, even though mediapipe
  declares it as a dependency: it would shadow the `opencv-python` build that
  provides `cv2.FaceDetectorYN`.

`_face_hand_trial/vendor_mediapipe.py` rebuilds `_vendor\` from scratch if it ever
needs refreshing. To use a normal install instead:

```bash
pip install mediapipe
# then delete _vendor\ -- the scripts fall back to the installed copy
```

---

## Usage

Run every command **from the folder that should receive the outputs**: the output
folders are created relative to the current working directory, not to the script.

```bash
# --- face maps (step A) ---
python yolo_face_saliency.py --video_dir video
python yolo_face_saliency.py video\look4CupH.mp4              # one video
python yolo_face_saliency.py --video_dir video --dry_run      # plan only, writes nothing

# --- hand + social maps (step B; reads the face maps) ---
python yolo_social_saliency.py --video_dir video
python yolo_social_saliency.py video\look4CupH.mp4
python yolo_social_saliency.py --video_dir video --no_social  # hand maps only

# --- object maps (alternative) ---
python yolo_heatmap_video.py --video_dir video
```

Both map scripts skip `*_flipped*` and `*_with_ROIs*` by default, because those are
re-exports of the same content and would double-count a stimulus. Override with
`--no_default_excludes`. Work that is already complete is skipped, so an
interrupted batch can simply be restarted; `--overwrite` forces a recompute.

Then score the maps exactly as described in [README.md](README.md), pointing
`--saliency_root` at whichever family you want to test:

```bash
python nss_score_new.py --gaze_dir Resample_Output --recursive --saliency_root saliency_frames_face
python auc_score_new.py --gaze_dir Resample_Output --recursive --saliency_root saliency_frames_face
```

### Options worth knowing

**`yolo_face_saliency.py`**

| Option | Default | Purpose |
|---|---|---|
| `--mask_shape {mesh,box}` | `mesh` | face-shaped outline, or a rectangle |
| `--include_hair` | off | extend the mask to the hair (≈15× slower, less reliable) |
| `--mesh_confidence` | `0.5` | keep at 0.5 — see [measured behaviour](#speed-and-thresholds) |
| `--face_threshold` | `0.5` | YuNet score threshold |
| `--blur_sigma` | `0.0` | extra Gaussian softening of the mask edge |

**`yolo_social_saliency.py`**

| Option | Default | Purpose |
|---|---|---|
| `--hand_soft_edge` | `12.0` | px the hull is grown outward and faded over; `0` = hard hull |
| `--hand_dilate` | `0.0` | grow the hull itself before the soft edge (reaches held objects) |
| `--hand_confidence` | `0.5` | hand detection threshold |
| `--num_hands` | `4` | maximum hands per frame |
| `--no_social` | off | hand maps only |

---

## Output format

For a stimulus `NAME.mp4`:

```
saliency_frames_face\NAME\frame_0001.png … frame_NNNN.png
output\NAME_face.mp4
```

- one PNG per video frame, **no gaps**, numbered from `0001`
- single channel, 8-bit, **exactly the video's pixel size** — the coordinate space
  the gaze data refers to
- `0` = background, `255` = strongest evidence for the region
- the `.mp4` is a visualisation only: original frame | colourised map

A frame with no detection produces an **all-zero map**. That is legitimate data,
not an error; how the scoring scripts treat such frames is an analysis decision.

---

## How each map is built

Every map here is a **region-of-interest** map: the value at a pixel answers *"how
confident is the detector that this pixel belongs to the region?"*. None of them is
a gradient/CAM attribution map.

**object** — YOLOv8-seg masks weighted by detection confidence, combined per pixel
with a maximum.

**face** — YuNet proposes face boxes, then MediaPipe's 478-landmark model is fitted
*inside each box* and the convex hull of those landmarks becomes the mask. The hull
is the **facial oval**: it does **not** include hair, because MediaPipe has no hair
landmarks — the name `_face` reflects what is actually in the pixels. Where
landmarks cannot be fitted the region falls back to a rectangle, and the fit rate
is printed so a rectangle-shaped result is never mistaken for a shaped one.

**hand** — MediaPipe HandLandmarker (21 landmarks per hand) runs on the full frame.
HandLandmarker contains its own palm detector, so no additional model is needed.
The convex hull of the landmarks is filled, then **grown outward and faded to zero
over `--hand_soft_edge` pixels**. Measured on real frames the bare hull is accurate
but slightly tight, leaving a thin uncovered rim; the soft outward edge closes that
rim and gives the gradient edge of an ordinary heatmap rather than a hard cut. A
hand holding an object usually ends up with the object inside the hull — that is a
property of the geometry, not a bug.

**social** — `max(face_map, hand_map)`, pixel by pixel. The face side is *read* from
`saliency_frames_face\` rather than recomputed, so the face numbers stay identical
to the ones already produced.

---

## Measured behaviour

All figures below were measured on this project's stimulus set: 1920×1080, 60 fps,
CPU only, Windows. Re-measure on your own data before relying on them.

### Why the face maps use YuNet, not MediaPipe's FaceLandmarker

The face pipeline is a hybrid: **YuNet detects, MediaPipe measures the shape.** That
was a deliberate choice, made after measuring MediaPipe's own face detector:

| Video | YuNet (in use) | MediaPipe FaceLandmarker @0.5 | @0.3 |
|---|---|---|---|
| close-up presenter | 100 % | 100 % | 100 % |
| memorySimpleH | 100 % | **32 %** | 50 % |
| mixingColorsH | 100 % | **0 %** | 0 % |
| buildQueuH | 100 % | **0 %** | 37 % |
| squareBlockR | 100 % | **10 %** | 100 % |

MediaPipe's face detector misses large parts of these clips even at a permissive
threshold, so YuNet supplies the candidates and MediaPipe only fits landmarks inside
each one.

**Mesh fitting depends on face size**, and magnifying a small crop does *not* help —
upscaling by 2×–6× left the fit rate at 0/48 — so it is a limitation of the model,
not of the input resolution:

| Face width | Regions that got a real outline |
|---|---|
| ~210 px | 100 % |
| ~60–130 px | ~79 % |
| ~24 px | **0 % — all rectangles** |

The scripts print the fit rate per video and warn below 50 %.

### Hand coverage per stimulus

MediaPipe HandLandmarker at confidence 0.5, 60 frames sampled evenly across each
whole clip, on the 30 non-srt stimuli:

| Hands present in | Videos |
|---|---|
| ≥ 80 % of frames | 18 |
| 50–80 % | 10 |
| 20–50 % | 1 (`look4TomatoH`) |
| < 20 % | 1 (`look4CupH`, 3 %) |

Across that sample: **4254 hand detections**, median hand width 105–205 px, **zero
detections smaller than 20 px**, 16–32 ms per frame for the landmarker alone. Every
clip contained hands somewhere; none was empty.

> A cautionary note from developing this: sampling only the **first 40 frames**
> produced a "13 of 30 clips have no hands" result that was completely wrong — those
> clips simply start with no hands in shot. Sample across the whole clip.

### Tuning the hand edge

`--hand_soft_edge` trades specificity for recall. One frame of `memorySimpleH`,
all maps 1920×1080:

| Map | Pixels covered | Distinct grey levels |
|---|---|---|
| hand, `--hand_soft_edge 0` | hull only | 2 (hard binary) |
| hand, `--hand_soft_edge 12` | hull + 12 px graded | 64 (real gradient) |
| face (YuNet) | facial oval | 2 |
| social | face ∪ hand | 65 |

The right value is an empirical question for your scoring: run one clip at
`--hand_soft_edge 0`, `12` and `30`, score each, and compare NSS/AUC. A wider
falloff admits more background.

### Speed and thresholds

| Stage | Per frame |
|---|---|
| YuNet face detection (box mode) | ~30 ms |
| mesh shape fitting | ~9 ms per face |
| HandLandmarker, landmarker time alone | 16–32 ms |
| `yolo_social_saliency.py` end to end | ~136 ms |

The end-to-end hand figure is much larger than the raw landmarker time because each
frame also builds masks, writes a PNG and encodes video. Budget roughly **1–2 hours
per 30,000 frames** for the social maps on CPU.

On a clip containing exactly one person, `--mesh_confidence` behaves as follows
(120 frames):

| Threshold | 1 face reported | 2 faces reported |
|---|---|---|
| **0.5** | **120** | **0** |
| 0.3 | 79 | 41 |
| 0.1 | 40 | 80 |

Keep it at the default. For hands, lowering `--hand_confidence` from 0.5 to 0.15
changed almost nothing, so 0.5 is a safe default there too.

---

## Model files and licences

Weights are bundled so a fresh clone runs without downloading anything. **Verify
these licences against your own distribution requirements.**

| File | Size | Source | Licence |
|---|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | 227 KB | [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Apache-2.0 |
| `models/face_landmarker.task` | 3.6 MB | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Apache-2.0 |
| `models/hand_landmarker.task` | 7.5 MB | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Apache-2.0 |
| `models/hair_segmenter.tflite` | 763 KB | [MediaPipe Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) | Apache-2.0 |
| `_vendor/mediapipe/` | 22 MB | [mediapipe 1.1.0](https://pypi.org/project/mediapipe/1.1.0/) wheel, unpacked | Apache-2.0 |
| `_vendor/absl-py`, `certifi`, `flatbuffers` | ~40 MB | PyPI wheels, unpacked | Apache-2.0 / MPL-2.0 / Apache-2.0 |

Recorded checksums, so users can confirm the weights are unmodified:

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  face_landmarker.task
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  hand_landmarker.task
2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84  hair_segmenter.tflite
```

`yolov8n-seg.pt` (the object pipeline's weights) is downloaded automatically by
`ultralytics` on first use and is **not** bundled.

**Licence note.** The code here is MIT, but `ultralytics` is **AGPL-3.0**. It is
used only by `yolo_heatmap_video.py`; the face, hand and social pipelines do not
import it. If you redistribute anything that depends on Ultralytics, read their
licence first — AGPL-3.0 is materially stricter than MIT.

---

## Building a distribution folder for someone else

Everything except the object pipeline needs only these five entries:

```
yolo_face_saliency.py
yolo_social_saliency.py
face_hand_core.py
models/
_vendor/
```

Give the recipient these instructions:

1. `pip install numpy pandas opencv-python` (and **not** `opencv-contrib-python`)
2. run `python yolo_face_saliency.py --self_test` to confirm the setup
3. put the stimulus videos in `video\`, then run `yolo_face_saliency.py --video_dir video`
   followed by `yolo_social_saliency.py --video_dir video`

Stimulus videos are not part of the package. `_vendor\` is 63 MB; it may be omitted
only if the recipient installs `mediapipe` themselves.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'face_hand_core'`**
`face_hand_core.py` must sit next to the script you are running.

**`ERROR: YuNet face model not found` / `hand landmarker model not found`**
The `models/` folder is missing or incomplete. Copy it next to the scripts, or pass
`--face_model` / `--hand_model` explicitly.

**`ERROR: this script needs mediapipe, which is not importable`**
`_vendor/` is missing. Restore it (`_face_hand_trial/vendor_mediapipe.py` rebuilds
it) or `pip install mediapipe`.

**A video is skipped with `no face maps in saliency_frames_face\...`**
Run `yolo_face_saliency.py` for that video first — the social maps read the face
maps instead of recomputing them.

**Output folders appear in an unexpected place**
They are created relative to the *current working directory*, not to the script.
`cd` to the folder you want them in.

**`could not open the output video for writing`**
The output folder is not writable, or the file is held open by another program.

**Everything is rectangles, not face shapes**
The faces are too small for the landmark model — see
[mesh fitting](#why-the-face-maps-use-yunet-not-mediapipes-facelandmarker).

**I interrupted a batch and a video will not play**
Video files are finalised when the writer closes. A hard-killed process can leave a
`.mp4` with no index (`moov atom not found`). Delete that file and that video's
frame folder, then re-run — the scripts skip only work that looks complete.

---

## References

### Models and libraries

- **Ultralytics YOLO** — the object maps use YOLOv8 instance segmentation via
  [Ultralytics](https://github.com/ultralytics/ultralytics).
  Jocher, Chaurasia & Qiu, *Ultralytics YOLOv8* (2023). Licence: AGPL-3.0.
- **YuNet** — Wei, Yang, Wang et al., *YuNet: A Tiny Millisecond-level Face
  Detector*, Machine Intelligence Research (2024). Weights via
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- **MediaPipe Face Landmarker / Face Mesh** — Kartynnik, Ablavatski, Grishchenko &
  Grundmann, *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs*
  (2019), [arXiv:1907.06724](https://arxiv.org/abs/1907.06724); Grishchenko &
  Ablavatski, *Attention Mesh: High-fidelity Face Mesh Prediction in Real-time*
  (CVPRW 2020).
  [Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
- **MediaPipe Hand Landmarker** — Zhang, Bazarevsky, Vakunov et al., *MediaPipe
  Hands: On-device Real-time Hand Tracking* (2020),
  [arXiv:2006.10214](https://arxiv.org/abs/2006.10214).
  [Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
- **MediaPipe** framework — Lugaresi et al., *MediaPipe: A Framework for Building
  Perception Pipelines* (2019),
  [arXiv:1906.08172](https://arxiv.org/abs/1906.08172).
- **OpenCV** — Bradski, *The OpenCV Library*, Dr. Dobb's Journal of Software Tools
  (2000). <https://opencv.org>
- **NumPy** — Harris et al., *Array programming with NumPy*, Nature 585 (2020).

Metric references (NSS, AUC-Judd) are listed in [README.md](README.md#references).
