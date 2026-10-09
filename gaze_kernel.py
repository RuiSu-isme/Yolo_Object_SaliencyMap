"""
Gaze belief kernels, shared by nss_score_new.py and auc_score_new.py.

=====================================================================
SHARED HELPER - DO NOT RUN THIS FILE DIRECTLY
=====================================================================
Step 3 (nss_score_new.py) and step 4 (auc_score_new.py) both import this
module, so it must sit in the same folder as those two scripts:

    folder\\
        gaze_kernel.py       <- this file
        nss_score_new.py
        auc_score_new.py

It holds everything about "what does one recorded gaze point stand for":

    - converting degrees of visual angle into pixels for a given frame size,
      from the screen width/height and the viewing distance
    - building the kernel itself (Gaussian, disk or single pixel)
    - reading the saliency values and weights under the kernel, and the
      weighted histograms the AUC is computed from

It needs numpy only, and it is imported automatically - there is no command to
run. The defaults describe a 1920x1080 screen of 52.0 x 29.0 cm viewed from
60 cm, which is about 41.0 px per degree horizontally and 39.7 px vertically;
every scoring run prints the kernel it actually used, so the log always records
the geometry behind the numbers.
=====================================================================

WHY THIS EXISTS
---------------
An eye tracker does not report the exact fixation location. This setup has
roughly 2 degrees of visual angle of deviation, and every recording carries its
own measured calibration error (calib_acc_raw / calib_acc_nudged in the raw
gaze files - 0.77 to 4.40 degrees across the participants of this study). A
recorded gaze point is therefore better modelled as a small region of belief
around that point:

    belief(pixel) = Gaussian(pixel - gaze; sigma_x, sigma_y)      sums to 1

sigma is given in DEGREES OF VISUAL ANGLE and converted to pixels with the
screen geometry (screen width, viewing distance) and each saliency map's own
size, so "1 degree" means the same physical region in a 1920x1080 video and in
a 1080x720 one. With the default sigma of 1 degree, 86.5% of the belief mass
falls inside 2 degrees, i.e. inside the tracker deviation being modelled.

Three kernels are available, all used through the same code path:

    gaussian   anisotropic Gaussian, --sigma_deg (default 1.0)
    disk       uniform over a disk, --radius in pixels
    point      a single pixel: the original scripts' behaviour, --sigma_deg 0

WHAT THE TWO METRICS DO WITH THE WEIGHTS
----------------------------------------
NSS  the weighted mean of the z-scored saliency map over the kernel,
     sum(w * z) / sum(w). Weights are renormalised over the part of the kernel
     that falls inside the frame, so edge fixations stay well defined.
AUC  the ROC is swept down over every 8-bit level, with

         TP(t) = (kernel mass on pixels with value >= t) / (total mass)
         FP(t) = (pixels with value >= t outside the kernel) / (their count)

     so a kernel that half-covers a salient region contributes half its mass
     instead of an all-or-nothing "more than half the circle" vote, and ties in
     an 8-bit saliency map are counted exactly.
"""

import math

import numpy as np


# Measured setup of the experiment: a 1920x1080 panel of 52.0 x 29.0 cm viewed
# from 60 cm. The screen therefore subtends
#
#     2 * atan(26.0 / 60) = 46.86 degrees horizontally
#     2 * atan(14.5 / 60) = 27.17 degrees vertically
#
# which is about 41.0 px per degree horizontally and 39.7 px per degree
# vertically for a 1920x1080 video (23.0 and 26.5 for a 1080x720 one).
DEFAULT_SCREEN_WIDTH_CM = 52.0
DEFAULT_SCREEN_HEIGHT_CM = 29.0
DEFAULT_VIEWING_DISTANCE_CM = 60.0

KERNELS = ("gaussian", "disk", "point")


def screen_angles(screen_width_cm, screen_height_cm, viewing_distance_cm):
    """Horizontal and vertical angle (degrees) subtended by the screen."""
    return (
        2.0 * math.degrees(math.atan((screen_width_cm / 2.0) / viewing_distance_cm)),
        2.0 * math.degrees(math.atan((screen_height_cm / 2.0) / viewing_distance_cm)),
    )


def pixels_per_degree(frame_width, frame_height,
                      screen_width_cm=DEFAULT_SCREEN_WIDTH_CM,
                      screen_height_cm=DEFAULT_SCREEN_HEIGHT_CM,
                      viewing_distance_cm=DEFAULT_VIEWING_DISTANCE_CM):
    """Pixels per degree of visual angle for one frame size.

    Assumes the video fills the screen, which is true for the 1920x1080 videos
    of this study. A video with another aspect ratio (the 1080x720 ones) would
    normally be letterboxed or stretched, which this cannot know about.
    """
    angle_x, angle_y = screen_angles(screen_width_cm, screen_height_cm, viewing_distance_cm)
    return frame_width / angle_x, frame_height / angle_y


class GazeKernel:
    """The belief kernel for one saliency map size."""

    def __init__(self, frame_width, frame_height, kind="gaussian", sigma_deg=1.0,
                 radius_px=0, screen_width_cm=DEFAULT_SCREEN_WIDTH_CM,
                 screen_height_cm=DEFAULT_SCREEN_HEIGHT_CM,
                 viewing_distance_cm=DEFAULT_VIEWING_DISTANCE_CM,
                 truncate_sigma=3.0, px_per_deg=None):

        if kind not in KERNELS:
            raise ValueError(f"unknown kernel '{kind}', expected one of {KERNELS}")

        # A zero-width Gaussian means "one pixel", so let --sigma_deg 0 keep
        # working instead of dividing by zero.
        if kind == "gaussian" and float(sigma_deg) <= 0:
            kind = "point"
        if kind == "disk" and int(round(radius_px)) < 0:
            raise ValueError("the disk kernel radius cannot be negative")

        self.kind = kind
        self.sigma_deg = float(sigma_deg)
        self.screen_width_cm = float(screen_width_cm)
        self.screen_height_cm = float(screen_height_cm)
        self.viewing_distance_cm = float(viewing_distance_cm)

        if px_per_deg is not None:
            self.px_per_deg_x = self.px_per_deg_y = float(px_per_deg)
        else:
            self.px_per_deg_x, self.px_per_deg_y = pixels_per_degree(
                frame_width, frame_height, self.screen_width_cm,
                self.screen_height_cm, self.viewing_distance_cm)

        if kind == "point":
            self.sigma_x = self.sigma_y = 0.0
            self.radius = 0
            self.weights = np.ones((1, 1), dtype=np.float64)

        elif kind == "disk":
            self.sigma_x = self.sigma_y = 0.0
            self.radius = int(round(radius_px))
            size = 2 * self.radius + 1
            ys, xs = np.ogrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
            self.weights = ((xs * xs + ys * ys) <= self.radius * self.radius).astype(np.float64)

        else:  # gaussian
            self.sigma_x = self.sigma_deg * self.px_per_deg_x
            self.sigma_y = self.sigma_deg * self.px_per_deg_y
            self.radius = int(math.ceil(truncate_sigma * max(self.sigma_x, self.sigma_y)))
            xs = np.arange(-self.radius, self.radius + 1, dtype=np.float64)
            gx = np.exp(-0.5 * (xs / self.sigma_x) ** 2)
            gy = np.exp(-0.5 * (xs / self.sigma_y) ** 2)
            self.weights = np.outer(gy, gx)

        self.weights = self.weights / self.weights.sum()
        self.weights_flat = self.weights.ravel()
        self.size = int(self.weights.size)

    # ----------------------------------------------------------------

    def block(self, saliency_map, ix, iy):
        """Kernel values and weights around (ix, iy), clipped to the frame.

        Returns (values_flat, weights_flat) where weights are the raw kernel
        weights of the pixels that actually exist.
        """
        height, width = saliency_map.shape
        y0 = max(0, iy - self.radius)
        y1 = min(height, iy + self.radius + 1)
        x0 = max(0, ix - self.radius)
        x1 = min(width, ix + self.radius + 1)

        values = saliency_map[y0:y1, x0:x1].ravel()
        ky = y0 - (iy - self.radius)
        kx = x0 - (ix - self.radius)
        weights = self.weights[ky:ky + (y1 - y0), kx:kx + (x1 - x0)].ravel()

        return values, weights

    def weighted_mean(self, value_map, ix, iy):
        """Weighted mean of a (float) map over the kernel, renormalised."""
        values, weights = self.block(value_map, ix, iy)
        total = weights.sum()
        if total <= 0:
            return None
        return float(np.dot(weights, values.astype(np.float64)) / total)

    def weighted_histograms(self, saliency_map, ix, iy, with_counts=False):
        """Belief mass (and optionally pixel counts) per 8-bit saliency value.

        mass[v]  = kernel weight on pixels whose saliency value is v
        count[v] = number of *fixated* pixels with that value, i.e. those with
                   a non-zero weight. A disk kernel is stored in its bounding
                   box, whose corners carry zero weight and must not be counted
                   as fixated.
        """
        values, weights = self.block(saliency_map, ix, iy)
        mass = np.bincount(values, weights=weights, minlength=256)
        if not with_counts:
            return mass, None
        counts = np.bincount(values[weights > 0], minlength=256)
        return mass, counts

    # ----------------------------------------------------------------

    def containment(self, degrees):
        """Fraction of the kernel mass within the given radius in degrees."""
        ys, xs = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        distance = np.sqrt((xs / self.px_per_deg_x) ** 2 + (ys / self.px_per_deg_y) ** 2)
        return float(self.weights[distance <= degrees].sum())

    def describe(self):
        if self.kind == "point":
            return "single pixel (the original point model)"
        if self.kind == "disk":
            return (f"uniform disk of radius {self.radius} px "
                    f"(kernel {self.weights.shape[0]}x{self.weights.shape[1]}, "
                    f"{self.size} pixels)")
        return (f"Gaussian sigma = {self.sigma_deg:.2f} deg "
                f"({self.sigma_x:.1f} px horizontally, {self.sigma_y:.1f} px vertically), "
                f"cut at {self.radius} px ({self.weights.shape[0]}x{self.weights.shape[1]} "
                f"= {self.size} pixels), {100 * self.containment(2.0):.1f}% of the mass "
                f"within 2.0 deg")


def make_kernel(frame_width, frame_height, **kwargs):
    return GazeKernel(frame_width, frame_height, **kwargs)
