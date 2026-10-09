# Saliency vs. gaze: NSS and AUC-Judd pipeline

This is a small Python pipeline that asks one question: **how well does a
saliency map built from object detection predict where people actually looked?**

Inputs are (a) eye-tracking recordings made while participants watched short
videos and (b) the videos themselves. For every video frame the pipeline
computes two standard saliency metrics:

| Metric | Meaning | Chance level |
|---|---|---|
| **NSS** (Normalized Scanpath Saliency) | how much more salient than average the looked-at region is, in units of the map's standard deviation | 0 |
| **AUC-Judd** (AUC-J) | area under the ROC of "is this pixel fixated?", swept over all saliency thresholds | 0.5 |

Outputs are plain CSV files: per-frame scores, plus one row per participant and
a pooled summary per task.

---

## Contents

- [Requirements](#requirements)
- [Files](#files)
- [Folder layout](#folder-layout)
- [Run it: step by step](#run-it-step-by-step)
- [The gaze model (read before interpreting numbers)](#the-gaze-model-read-before-interpreting-numbers)
- [Reading the output files](#reading-the-output-files)
- [Option quick reference](#option-quick-reference)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [References](#references)

---

## Requirements

- **Python 3.8+**
- **Step 3 and 4 (scoring):** `pandas`, `numpy`, `opencv-python`
- **Step 2 (saliency maps):** `ultralytics` (pulls in `torch`), plus
  `opencv-python`. A GPU is *not* required; on CPU the model takes roughly
  0.2 s per 1920x1080 frame, i.e. about **5 minutes of 60 fps video per hour**
  of processing (writing the PNG maps adds a little on top).
- Step 1 needs `pandas`, `numpy` and `opencv-python` as well.

```bash
pip install pandas numpy opencv-python
pip install ultralytics          # only needed for step 2
```

Check the install before touching real data (both scripts use made-up data, no
files of yours are read):

```bash
python test_gaze_kernel.py       # -> ALL CHECKS PASSED
python test_auc.py               # -> ALL CHECKS PASSED
```

---

## Files

| File | Pipeline step | What it does |
|---|---|---|
| `resample_gaze.py` | **1** | raw tracker export -> one row per video frame |
| `yolo_heatmap_video.py` | **2** | stimulus video -> per-frame grayscale saliency maps (+ a heatmap video for looking at) |
| `nss_score_new.py` | **3** | saliency maps x gaze -> per-frame NSS |
| `auc_score_new.py` | **4** | saliency maps x gaze -> per-frame AUC-Judd (per participant **and** pooled) |
| `gaze_kernel.py` | shared | the "what does one gaze point stand for" model used by steps 3 and 4; **must sit next to those two scripts** |
| `test_gaze_kernel.py`, `test_auc.py` | self-test | synthetic correctness checks; run them to confirm the install |

---

## Folder layout

Names are conventions, not requirements - every path is a command-line
argument. `<task>` is the video/task name (e.g. `blockTowerR`), `<name>` is a
recording (e.g. `1268_bl_2025-04-28_blockTowerR.mp4_gaze.csv`).

```
gazedata/<name>.csv                       raw export, one row per ~120 Hz sample
video/<task>.mp4                          the stimulus videos
Resample_Output/<task>/<name>.csv         step 1 output
saliency_frames/<task>/frame_0001.png     step 2 output (grayscale maps)
output/<task>.mp4                         step 2 output (side-by-side heatmap video)
nss_Output/<task>/...                     step 3 output
auc_Output/<task>/...                     step 4 output
```

The task name is read from the gaze file name, so
`1268_bl_2025-04-28_blockTowerR.mp4_gaze.csv` belongs to `blockTowerR`.

---

## Run it: step by step

### Step 1 - resample the raw gaze data

The sampling rate of the eye tracker is 120Hz, but the videos are normally 60, 30 or 25 fps. 
Therefore, resampling is needed. 
The code will first automatically read the FPS from video/<task>.mp4, and then resample the gaze data based on the FPS of the video.

```bash
# normal case: read the corresponding FPS of the video for each gaze file automatically, resample the data, and sort it into its own subfolder. 
python resample_gaze.py <inputfolder> <outputfolder> --by_task

# one task only, sorted into its own subfolder
python resample_gaze.py <inputfolder> <outputfolder> --only "*videoname*" --by_task

```

Output:

```
without --by_task : <output_folder>/<same file name>.csv        one flat folder
with    --by_task : <output_folder>/<task>/<same file name>.csv
```

with the columns `videoFrame, videoTime, gaze_x, gaze_y, calib_winner,
gaze_missing`. The task (video) name is read from the gaze file name, so
`--by_task` is how you get one folder per video. `--dry_run` prints the task and
the exact destination path of every file - plus a per-task count - without
writing anything, and a normal run ends with the same per-task count.

**Check afterwards:** the row count should be close to the number of saliency
maps for that task (a few frames fewer is normal if the recording is shorter
than the video). The script also warns by itself when the frames it wrote
disagree with the frame count of the stimulus video.

### Step 2 - build the saliency maps

Independent of the gaze data: run it once per video and reuse it for every
participant. YOLOv8-seg detects objects, each frame's masks are combined into
one grayscale map, and the map is saved as PNG (plus a colourised
side-by-side video for a quick visual sanity check).

```bash
# every stimulus video of a folder
python yolo_heatmap_video.py --video_dir video

# skip mirrored / ROI-annotated exports of the same content
python yolo_heatmap_video.py --video_dir video --exclude "*_flipped*" --exclude "*_with_ROIs*"

# a single video
python yolo_heatmap_video.py video/blockTowerR.mp4

# see the plan without loading the model, write nothing
python yolo_heatmap_video.py --video_dir video --dry_run
```

Output: `saliency_frames/<name>/frame_0001.png ...` and `output/<name>.mp4`.
Single-video mode uses `--frames_dir DIR` for the maps and the second
positional argument for the video (default `yolo_heatmap_output.mp4`); note
that `--frames_dir` is the single-video option and `--frames_root` the batch
one.

Videos that already have results are skipped, so an interrupted batch can
simply be started again (`--overwrite` recomputes them). On CPU this is the
slow step - use `--only` and `--max_frames 50` first if you want a smoke test.

### Step 3 - NSS

```bash
# every participant of one task
python nss_score_new.py --gaze_dir Resample_Output/<taskname> --saliency_dir saliency_frames/<taskname>

# many tasks at once: the map folder is picked per file from its own name
python nss_score_new.py --gaze_dir Resample_Output --recursive --saliency_root saliency_frames

# single gaze file
python nss_score_new.py --gaze_file x.csv --saliency_dir saliency_frames/blockTowerR --output result.csv
```

Output, per task:

```
nss_Output/<task>/<gaze file name>          frame,x,y,nss   (one row per frame)
nss_Output/<task>/<task>_nss_summary.csv    one row per participant
                                            (mean_nss, std_nss, min/max,
                                             rows_scored, skipped_*)
```

### Step 4 - AUC-Judd

Same inputs and options as step 3, plus `--fp_scope`.

```bash
python auc_score_new.py --gaze_dir Resample_Output/blockTowerR --saliency_dir saliency_frames/blockTowerR
python auc_score_new.py --gaze_dir Resample_Output --recursive --saliency_root saliency_frames
```

Output, per task (four files, see the next section for what to use when):

```
auc_Output/<task>/<gaze file name>                   frame,x,y,auc_j
auc_Output/<task>/<task>_auc_summary.csv             one row per participant
auc_Output/<task>/<task>_auc_pooled_by_frame.csv     frame,n_fixations,auc_j
auc_Output/<task>/<task>_auc_pooled_summary.csv      one-row pooled summary
```

---

## The gaze model (read before interpreting numbers)

An eye tracker is not exact: this setup is accurate to roughly **2 degrees of
visual angle**, and each recording carries its own measured calibration error.
A recorded gaze point therefore stands for a small **region of belief**, not a
single pixel, and the metrics are computed against that region:

```
NSS  = weighted mean of the z-scored map over the kernel, sum(w*z)/sum(w)
AUC  = ROC weighted by the same kernel: TP(t) = belief mass on pixels with
       value >= t, FP(t) = non-fixated pixels with value >= t
```

A region that half covers an object therefore contributes **half** its mass,
instead of an all-or-nothing "more than half the circle counts as a hit" rule.

`--kernel` chooses the shape:

| `--kernel` | Region | Notes |
|---|---|---|
| `gaussian` (default) | anisotropic Gaussian, `--sigma_deg` degrees | 86.9% of the mass lies within 2 degrees at the default sigma of 1 |
| `disk` | uniform disk, `--radius` **pixels** | for a 2 degree disk pass the radius in pixels, e.g. `--radius 82` on a 1920x1080 screen |
| `point` | one pixel | the classic single-pixel behaviour, useful as a baseline |

Degrees are converted to pixels with the screen geometry
(`--screen_width_cm`, `--screen_height_cm`, `--viewing_distance_cm`, or a
single `--px_per_deg`). The shipped defaults describe a 1920x1080 screen of
52.0 x 29.0 cm viewed from 60 cm: the screen subtends 46.86 x 27.17 degrees, so
one degree is **41.0 px horizontally and 39.7 px vertically** in a 1920x1080
video (23.0 and 26.5 in a 1080x720 one). Every run prints the kernel it
actually used, so the log records the geometry behind the numbers.

Two consequences worth knowing:

- NSS is unbounded and centred on 0 (negative = looked at a less salient than
  average region); AUC-J is bounded by [0, 1] with 0.5 = chance. Do not compare
  the two magnitudes with each other.
- With a large kernel even a perfectly centred gaze on a **small** object earns
  only partial credit (a 100 px object under the default kernel covers about
  60% of the belief mass). Object size therefore matters, and `--kernel point`
  is the version that gives binary credit.

---

## Reading the output files

| File | Granularity | Use it for |
|---|---|---|
| `*_nss_summary.csv` / `*_auc_summary.csv` | one row per participant | statistics across participants (t-test, ANOVA, LMM): one number per person |
| `<gaze file name>` | one row per frame | per-frame time courses, frame-level modelling, checking outliers |
| `*_auc_pooled_summary.csv` | one row per task | the "one number for the whole video" figure, using all participants' gaze at once |
| `*_auc_pooled_by_frame.csv` | one row per frame (all participants) | how well the model predicted the *group* over time |

Common columns: `rows_in_file` (rows found), `rows_scored` (rows that produced
a score), `skipped_no_map` (no saliency map for that frame), `skipped_*`
(missing/NaN or out-of-frame gaze), `status` (`ok` or an error description).

The pooled numbers are only meaningful when the batch really contains **all**
participants of that task; `--limit` makes them partial by definition.

All outputs are CSV, which Excel, R, SPSS and pandas read directly (in Excel
use *Data -> From Text/CSV*, do not just rename the file to `.xlsx`).

---

## Option quick reference

**Step 1 - `resample_gaze.py "input_folder" "output_folder"`**

| Option | Default | Meaning |
|---|---|---|
| `--fps auto\|N` | `auto` | read the rate from `video/<task>.mp4` (29.97 -> 30, 23.976 -> 24), or force one rate for the whole run |
| `--video_dir DIR` | `video` | where the stimulus videos are |
| `--only PATTERN` | - | only files matching (repeatable) |
| `--by_task` | off | write into `<output>/<task>/` instead of one flat folder |
| `--dry_run` | off | print what would happen, write nothing |

**Step 2 - `yolo_heatmap_video.py [input_video] [output_video]`**

| Option | Default | Meaning |
|---|---|---|
| `--video_dir DIR` | - | batch mode: process every video in this folder |
| `--frames_root DIR` | `saliency_frames` | batch: root of the per-video map folders |
| `--output_dir DIR` | `output` | batch: the heatmap videos |
| `--frames_dir DIR` | - | single-video mode: where the maps go |
| `--only` / `--exclude PATTERN` | - | filter video names (repeatable) |
| `--overwrite` | off | recompute videos that already have results |
| `--dry_run` | off | list what would be processed, load no model |
| `--max_frames N` | 0 (all) | stop after N frames per video (smoke test) |
| `--model PATH` | `yolov8n-seg.pt` next to the script | weights to use |

**Steps 3 and 4** (identical option sets; step 4 adds `--fp_scope`)

| Option | Default | Meaning |
|---|---|---|
| `--gaze_file FILE` \| `--gaze_dir DIR` | - | one file, or a whole folder |
| `--saliency_dir DIR` \| `--saliency_root DIR` | - | one map folder, or one subfolder per task |
| `--recursive` | off | also search subfolders of `--gaze_dir` |
| `--limit N` | 0 (all) | only the first N gaze files |
| `--output_dir DIR` | `nss_Output` / `auc_Output` | result root |
| `--group NAME` | task name | result subfolder |
| `--output FILE` | - | single-file mode: exact output path |
| `--frame_col / --x_col / --y_col` | auto | override the auto-detected columns |
| `--pattern "frame_{frame:04d}.png"` | as shown | saliency map file naming |
| `--coords normalized\|pixel` | `normalized` | how to read x and y |
| `--kernel gaussian\|disk\|point` | `gaussian` | gaze model (see above) |
| `--sigma_deg DEG` | 1.0 | Gaussian sigma in degrees of visual angle |
| `--radius PX` | 20 | disk kernel radius in pixels |
| `--truncate_sigma N` | 3.0 | cut the Gaussian at N sigmas |
| `--screen_width_cm / --screen_height_cm / --viewing_distance_cm` | 52.0 / 29.0 / 60.0 | screen geometry |
| `--px_per_deg X` | - | override the geometry with one number |
| `--fp_scope auto\|all\|nonfixation` | `auto` | AUC only: which pixels form the false-positive pool |

Both scripts accept the same gaze-column layouts automatically:
`frame, x, y` and `videoFrame, gaze_x, gaze_y` (the latter is what step 1
writes).

---

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `ERROR: this step needs the 'ultralytics' package` | `pip install ultralytics` (step 2 only) |
| `could not read the frame rate of the stimulus video` | the stimulus video was not found for that gaze file: pass `--video_dir`, or the rate with `--fps N` |
| `no frame/x/y columns ... columns found: [...]` | the gaze table uses other column names: pass `--frame_col`, `--x_col`, `--y_col` |
| `no saliency folder '<task>' under ...` | the task name inside the gaze file name has no matching subfolder: check the naming, or use `--saliency_dir` for a single task |
| `rows had no saliency map file` or a large `skipped_no_map` | the frame numbering of the gaze file does not match the maps. Usually step 1 was run at the wrong frame rate - re-run it (step 1 now reads the video's own rate) |
| `No NSS scores could be computed` | nothing overlapped at all: check both folders and the `--pattern` |
| where is my per-participant table? | `nss_Output/<task>/<task>_nss_summary.csv` and `auc_Output/<task>/<task>_auc_summary.csv`; they are CSV files, not Excel workbooks |
| a long batch was interrupted | just start it again: step 2 skips videos that already have results, and steps 3-4 simply recompute (use `--limit` for a quick check first) |

---

## Known limitations

1. **Videos that are not the screen's aspect ratio.** The gaze -> pixel mapping
   assumes the video filled the screen. If a video was shown with black bars
   (e.g. a 3:2 video on a 16:9 screen), the mapping is off by up to ~9% of the
   frame width near the edges and the sigma in pixels is off by ~16%
   horizontally. This is **not handled yet**; if your stimuli include such
   videos, resample/exclude them or handle the display rectangle explicitly.
2. **Frame rate.** Step 1 must use the stimulus video's own rate. If a gaze file
   is resampled at the wrong rate, frame indices silently stop matching the
   maps (the scores are then meaningless rather than obviously broken).
3. **Pooled metrics need the whole group.** `--limit` or a subset of
   participants makes the pooled numbers partial.
4. **Pixel level, not object level.** The metrics score pixels; they do not
   report hit/miss per object. Per-object hit rates would be a separate
   analysis built on the same maps.
5. **`--radius` is in pixels.** For a disk of a given visual angle, read the
   pixels-per-degree the run prints (e.g. 82 px = 2 degrees at 1920x1080 on the
   default screen).

---

## References

The metric implementations follow the reference code behind the MIT saliency
benchmark:

- Judd, Ehinger, Durand & Torralba (2009), *Learning to predict where humans
  look*, ICCV - AUC-Judd.
- Judd, Durand & Torralba (2012), *A Benchmark of Computational Models of
  Saliency to Predict Human Fixations*, MIT-CSAIL-TR-2012-001.
- Bylinskii, Judd, Oliva, Torralba & Durand (2019), *What do different
  evaluation metrics tell us about saliency models?*, IEEE TPAMI 41(3),
  [arXiv:1604.03605](https://arxiv.org/abs/1604.03605).
- Reference implementation:
  [cvzoya/saliency](https://github.com/cvzoya/saliency) (`code_forMetrics/`,
  in particular `NSS.m` and `AUC_Judd.m`).
- NSS was introduced in Peters, Iyer, Itti & Koch (2005), *Components of
  bottom-up gaze allocation in natural images*, Vision Research 45(18).

Deliberate differences from that reference code, all documented in the script
docstrings: ties in an 8-bit saliency map are counted exactly instead of being
broken with `rand/1e7` noise; the AUC threshold is swept over every 8-bit level
(the exact staircase ROC, which equals the rank-based AUC) instead of only over
the fixation values; and the gaze point is modelled as a kernel (default
Gaussian, sigma = 1 degree) rather than a single pixel.
