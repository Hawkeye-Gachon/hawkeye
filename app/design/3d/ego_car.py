"""
DriverGuard ego car: "my car" seen from behind, driving toward a blue-hour horizon.
Procedural, generic EV fastback sedan: no real make/model, no logos, badges or text.

The app draws the 3D road itself (Compose, true perspective) with the camera given in
camera.json, then layers two images rendered here with exactly that camera:
  ego_car_body.webp   car + soft contact shadow, transparent background, light bar OFF
  ego_car_light.webp  rear light bar emission only: WHITE rgb, alpha = intensity
                      (bar + bloom + spill on the bumper); the app tints it per state
                      (ColorFilter SrcIn) and draws it additively.

Run (headless; renders + post-processes + writes WebP into the app):
  /opt/homebrew/bin/blender -b --factory-startup -P design/3d/ego_car.py -- \
      [--samples 384] [--scale 1.0] [--preview-dir design/3d/renders/ego_car_blue] \
      [--view side|top|rear34  (debug views)] [--cpu]

The same file doubles as the post-processing step (system python3 + numpy + Pillow);
Blender calls it automatically:
  python3 design/3d/ego_car.py --post --work <dir> --out <dir> [--preview-dir <dir>]

Camera (exact; the app reproduces it): on the centre line (x = 0) behind the car,
pure downward pitch about X (no yaw/roll), 30 mm lens on a 36 mm sensor (horizontal
fit), 1600x1000 px, shift_x = 0. Distance and shift_y are solved so the car silhouette
(mirrors included) spans FRAME_W of the width and the contact shadow ends SHADOW_BOTTOM
above the bottom edge. World: z up, ground z = 0, car centred at the origin, front +Y.

Pipeline
  1. Build the car from lofted cross-sections (Catmull-Clark subdivided), wheel
     arches by boolean, revolved tyres/aero covers, ray-cast light bar. Glass
     outline = per-vertex field thresholded in the shader (smooth curved edges).
  2. Solve the camera, export camera.json.
  3. Body pass: blue-hour sky (brighter horizon ahead of the car = broad rim light),
     dim fill from the camera side; road invisible to the camera but present for
     bounce / reflections; film transparent -> car RGBA.
  4. Shadow pass: car invisible to the camera, road = shadow catcher, soft overhead
     light + sky -> contact-shadow alpha.
  5. Light pass: world and lights off, light bar emitting white, small spill light;
     linear EXR -> intensity.
  6. Post: layer shadow + car (sRGB), map light intensity + bloom to alpha, WebP.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

try:
    import bpy  # noqa
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# ---- camera (exported to camera.json; the app reproduces it) -------------------
CAM_PITCH_DEG = 10.0    # downward pitch about X; no yaw, no roll
CAM_LENS = 30.0         # mm
CAM_SENSOR = 36.0       # mm, sensor_fit HORIZONTAL
CAM_HEIGHT = 1.80       # m above the road
FRAME_W = 0.46          # car silhouette (incl. mirrors) as a fraction of image width
SHADOW_BOTTOM = 0.05    # contact shadow ends this fraction of the height above the bottom
SHADOW_REACH = 0.25     # m: visible shadow ends this far behind the rear-most car point
W_PX, H_PX = 1600, 1000

# ---- look ---------------------------------------------------------------------
SKY_ZENITH, SKY_HORIZON, SKY_GROUND = "#13223D", "#3A5A8E", "#0F1C33"
ROAD = "#16233C"
WORLD_STRENGTH = 1.25
FWD_GLOW = 14.0         # extra horizon brightness ahead of the car (broad soft rim light)
KEY_W, FILL_W, TOP_W = 2200.0, 330.0, 160.0
KICK_W = 220.0
EXPOSURE = 0.85
BAR_EMIT = 8.0          # light pass: bar radiance (linear)
SPILL_W = 2.4           # light pass: spill light onto the bumper (W)
SHADOW_GAIN = 0.90
SHADOW_FOOT = (0.90, 2.30, 0.45)   # footprint half width, half length, corner radius (m)
SHADOW_FALLOFF = 0.16   # m: shadow decay outside the footprint
SHADOW_AO_BOOST = 1.6   # catcher alpha (dome occlusion) gain before the curve
SHADOW_GAMMA = 1.3
LIGHT_CORE_K = 1.5      # alpha = 1 - exp(-L / k) for the rendered light
BLOOM = ((3.0, 0.45), (10.0, 0.25), (34.0, 0.12))   # (sigma px @1600, alpha at the bar)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(REPO, "app", "src", "main", "res", "drawable-nodpi")
DEFAULT_PREVIEW = os.path.join(HERE, "renders", "ego_car_blue")

# preview mock (not shipped)
MOCK_SKY_TOP, MOCK_SKY_HZ = "#0C1730", "#3A5A8E"
MOCK_ROAD_HZ, MOCK_ROAD_BOTTOM = "#1B2A45", "#121D33"
PREVIEW_TINTS = {"teal": "#5BC8B1", "red": "#EC5A5F"}


def hex_srgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def hex_lin(h):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in hex_srgb(h)]


def horizon_row(cam):
    """Pixel row (from the top, continuous) of the horizon for a pure-pitch camera."""
    v = math.tan(math.radians(cam["pitch_deg"])) * cam["lens_mm"] / cam["sensor_width_mm"]
    return cam["height_px"] / 2 - (v - cam["shift_y"]) * cam["width_px"]


def project(cam, X, Y, Z):
    """World point -> (col, row) pixel, pinhole with pure pitch (same maths as the app)."""
    p = math.radians(cam["pitch_deg"])
    cx, cy, cz = cam["loc"]
    dx, dy, dz = X - cx, Y - cy, Z - cz
    # camera axes: right = +X, forward = (0, cos p, -sin p), up = (0, sin p, cos p)
    fwd = dy * math.cos(p) - dz * math.sin(p)
    up = dy * math.sin(p) + dz * math.cos(p)
    k = cam["lens_mm"] / cam["sensor_width_mm"]
    u = dx / fwd * k + cam["shift_x"]
    v = up / fwd * k - cam["shift_y"]
    w, h = cam["width_px"], cam["height_px"]
    return w / 2 + u * w, h / 2 - v * w


# =========================================================================
# Post-processing (runs in system python3 with numpy + Pillow)
# =========================================================================
def post_main(argv):
    import argparse
    import numpy as np
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-dir", default="")
    a = ap.parse_args(argv)

    def box_blur(img, r):
        if r < 1:
            return img
        k = 2 * r + 1
        for axis in (0, 1):
            pad = [(0, 0)] * img.ndim
            pad[axis] = (r + 1, r)
            p = np.pad(img, pad, mode="edge")
            c = np.cumsum(p, axis=axis)
            n = img.shape[axis]
            hi = np.take(c, np.arange(k, k + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            img = (hi - lo) / k
        return img

    def gauss(img, sigma):
        # 3 box passes ~ gaussian; box radius r gives variance (r^2 + r) / 3 per pass
        r = max(1, int(round((math.sqrt(1 + 4 * sigma * sigma) - 1) / 2)))
        for _ in range(3):
            img = box_blur(img, r)
        return img

    cam = json.load(open(os.path.join(a.work, "camera.json")))
    body = np.load(os.path.join(a.work, "body.npy")).astype(np.float64)       # display RGBA
    shadow = np.load(os.path.join(a.work, "shadow.npy")).astype(np.float64)   # catcher RGBA
    light = np.load(os.path.join(a.work, "light.npy")).astype(np.float64)     # linear RGB(A)
    h, w = body.shape[:2]
    sc = w / float(cam["width_px"])
    cam = dict(cam, width_px=w, height_px=h)      # drafts render at --scale < 1
    rng = np.random.default_rng(7)
    dither = (rng.random((h, w)) - 0.5) / 255.0

    # ---- body: car over contact shadow
    car_a = np.clip(body[..., 3], 0, 1)
    car_p = np.clip(body[..., :3], 0, 1) * car_a[..., None]     # premultiplied
    yy, xx = np.mgrid[0:h, 0:w]
    # ground point of every pixel: ray through the pixel (pure pitch pinhole, same maths as the app)
    p = math.radians(cam["pitch_deg"])
    k = cam["lens_mm"] / cam["sensor_width_mm"]
    u = ((xx + 0.5) - w / 2) / w - cam["shift_x"]
    v = (h / 2 - (yy + 0.5)) / w + cam["shift_y"]
    dxc, dyc = u / k, v / k                      # camera-space ray (right, up, forward = 1)
    dz_w = dyc * math.cos(p) - math.sin(p)
    dy_w = math.cos(p) + dyc * math.sin(p)
    t = np.where(dz_w < -1e-6, -cam["loc"][2] / np.minimum(dz_w, -1e-6), np.inf)
    gx = cam["loc"][0] + t * dxc
    gy = cam["loc"][1] + t * dy_w
    pix_m = np.where(np.isfinite(t), t * (1.0 / k) / w, 1.0)    # metres per pixel at that depth

    # contact shadow: AO-like catcher pass (uniform dome), tightened to the car footprint
    sh = np.clip(shadow[..., 3], 0, 1)
    hx, hy, rc = SHADOW_FOOT
    qx = np.abs(gx) - (hx - rc)
    qy = np.abs(gy) - (hy - rc)
    sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rc
    sd = np.where(np.isfinite(t), sd, 1e3)
    foot = np.exp(-(np.maximum(sd, 0) / SHADOW_FALLOFF) ** 1.6)
    sh = np.clip(sh * SHADOW_AO_BOOST, 0, 1) ** SHADOW_GAMMA * foot
    sh = gauss(sh, 1.2 * sc)
    sh = np.clip((sh - 0.01) / 0.99, 0, 1) * SHADOW_GAIN
    ex = np.clip(np.minimum(xx, w - 1 - xx) / (0.06 * w), 0, 1)
    ey = np.clip(np.minimum(yy, h - 1 - yy) / (0.03 * h), 0, 1)
    sh = sh * (ex * ey)
    sh = np.clip(sh + dither * (sh > 0.004), 0, 1)
    A = car_a + sh * (1 - car_a)
    C = car_p
    straight = np.where(A[..., None] > 1e-5, C / np.maximum(A[..., None], 1e-5), 0)
    body_rgba = np.dstack([np.clip(straight, 0, 1), np.clip(A, 0, 1)])

    # ---- light: intensity -> alpha (+ bloom), white rgb
    L = light[..., :3].max(-1)
    core = 1 - np.exp(-L / LIGHT_CORE_K)
    bar_src = np.clip(L / BAR_EMIT, 0, 1)            # the bar surface itself
    bloom = np.zeros_like(L)
    for sigma, wgt in BLOOM:
        s = sigma * sc
        g = gauss(bar_src, s)
        bloom += wgt * g / max(g.max(), 1e-6)       # weight = bloom alpha right at the bar
    la = 1 - (1 - core) * (1 - np.clip(bloom, 0, 1))
    la = np.clip(la + dither * (la > 0.004), 0, 1)
    la = np.where(la < 1.5 / 255, 0, la)
    light_rgba = np.dstack([np.ones((h, w, 3)), la])

    os.makedirs(a.out, exist_ok=True)

    def to8(x):
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)

    p_body = os.path.join(a.out, "ego_car_body.webp")
    Image.fromarray(to8(body_rgba), "RGBA").save(p_body, "WEBP", quality=90, alpha_quality=100, method=6)
    p_light = os.path.join(a.out, "ego_car_light.webp")
    Image.fromarray(to8(light_rgba), "RGBA").save(p_light, "WEBP", quality=90, alpha_quality=90, method=6)
    for p in (p_body, p_light):
        print("wrote", p, os.path.getsize(p), "bytes")

    # ---- stats: car pixel bounds (no shadow) + horizon
    m = car_a > 0.5
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    shadow_rows = np.where((sh > 8 / 255).any(1))[0]   # visible end of the shadow
    hz = horizon_row(cam)
    stats = {"car_bounds_ltrb": [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])],
             "car_width_frac": round((cols[-1] - cols[0] + 1) / w, 4),
             "car_centre_x": round((cols[0] + cols[-1] + 1) / 2, 1),
             "shadow_bottom_row": int(shadow_rows[-1]) if len(shadow_rows) else None,
             "horizon_row": round(hz, 2)}
    print("stats", json.dumps(stats))
    with open(os.path.join(a.work, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)

    if not a.preview_dir:
        return
    os.makedirs(a.preview_dir, exist_ok=True)
    # mock: navy sky above the true horizon, asphalt below, lane lines at x = +-1.8 m
    rowf = (yy + 0.5)
    t_sky = np.clip(rowf / max(hz, 1), 0, 1)[..., None] ** 2.2
    sky = np.array(hex_srgb(MOCK_SKY_TOP)) * (1 - t_sky) + np.array(hex_srgb(MOCK_SKY_HZ)) * t_sky
    t_rd = np.clip((rowf - hz) / max(h - hz, 1), 0, 1)[..., None] ** 0.6
    road = np.array(hex_srgb(MOCK_ROAD_HZ)) * (1 - t_rd) + np.array(hex_srgb(MOCK_ROAD_BOTTOM)) * t_rd
    bg = np.where((rowf < hz)[..., None], sky, road)
    # lane lines (ground coordinates from above)
    for lx in (-1.8, 1.8):
        d = np.abs(gx - lx) / np.maximum(pix_m, 1e-6)
        line = np.clip(1 - (d - 0.06 / np.maximum(pix_m, 1e-6)), 0, 1) * np.isfinite(t)
        dash = ((np.floor(gy / 3.0) % 2) == 0)
        bg = bg * (1 - 0.35 * (line * dash)[..., None]) + 0.35 * (line * dash)[..., None] * 0.62
    for name, tint in PREVIEW_TINTS.items():
        comp = C + bg * (1 - A[..., None])
        comp = comp + np.array(hex_srgb(tint)) * la[..., None]
        Image.fromarray(to8(comp), "RGB").save(os.path.join(a.preview_dir, f"preview_{name}.png"))
    # plain body (no light) + light alpha for inspection
    Image.fromarray(to8(C + bg * (1 - A[..., None])), "RGB").save(
        os.path.join(a.preview_dir, "preview_body_only.png"))
    print("previews ->", a.preview_dir)

# =========================================================================
# Blender scene
# =========================================================================
def smooth_interp(keys, t):
    """Monotone cubic (Fritsch-Carlson) interpolation through (x, v) keys."""
    xs = [k[0] for k in keys]
    vs = [k[1] for k in keys]
    if t <= xs[0]:
        return vs[0]
    if t >= xs[-1]:
        return vs[-1]
    n = len(xs)
    d = [(vs[i + 1] - vs[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            h0, h1 = xs[i] - xs[i - 1], xs[i + 1] - xs[i]
            w1, w2 = 2 * h1 + h0, h1 + 2 * h0
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    i = max(j for j in range(n - 1) if xs[j] <= t)
    hh = xs[i + 1] - xs[i]
    u = (t - xs[i]) / hh
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * vs[i] + h10 * hh * m[i] + h01 * vs[i + 1] + h11 * hh * m[i + 1]


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


# --- body dimensions (metres). +Y forward, +X right, Z up, ground z=0 ------
Y_REAR, Y_FRONT = -2.36, 2.36
AXLE_R, AXLE_F = -1.37, 1.50
WHEEL_R = 0.335
TRACK_HALF = 0.805

K_ZTOP = [(-2.36, 1.00), (-2.30, 1.045), (-2.18, 1.055), (-2.05, 1.070), (-1.85, 1.120),
          (-1.55, 1.215), (-1.20, 1.315), (-0.80, 1.395), (-0.40, 1.435), (0.00, 1.445),
          (0.30, 1.425), (0.50, 1.375), (0.80, 1.235), (1.10, 1.085), (1.32, 0.995),
          (1.70, 0.925), (2.05, 0.860), (2.25, 0.775), (2.36, 0.640)]
K_ZBELT = [(-2.36, 0.94), (-2.10, 0.975), (-1.60, 0.990), (-0.60, 0.975), (0.60, 0.955),
           (1.30, 0.925), (1.80, 0.860), (2.20, 0.745), (2.36, 0.600)]
K_ZB = [(-2.36, 0.44), (-2.20, 0.41), (-2.05, 0.36), (-1.92, 0.27), (-1.80, 0.18), (-1.70, 0.165), (1.70, 0.160),
        (2.00, 0.19), (2.25, 0.25), (2.36, 0.30)]
K_W = [(-2.36, 0.800), (-2.25, 0.870), (-2.05, 0.905), (-1.70, 0.925), (-1.37, 0.932),
       (-0.60, 0.918), (0.40, 0.915), (1.30, 0.922), (1.75, 0.910), (2.05, 0.875),
       (2.25, 0.800), (2.36, 0.690)]
K_WG = [(-2.36, 0.68), (-2.05, 0.75), (-1.70, 0.79), (-1.00, 0.805), (0.40, 0.80),
        (1.20, 0.785), (1.80, 0.75), (2.36, 0.57)]
K_WR = [(-2.36, 0.50), (-1.80, 0.555), (-1.20, 0.600), (-0.40, 0.620), (0.40, 0.605),
        (1.20, 0.575), (2.00, 0.52), (2.36, 0.43)]
# body-colour sail panels (C-pillars): side-glass segments stay paint behind these y values,
# lower segment ends further back so the window's rear edge slants like a real DLO
C_PILLAR_END = {8: -1.32, 10: -0.95}
A_PILLAR_START = {8: 1.14, 10: 0.74}   # side glass ends here at the front (slanted A-pillar)
STATION_STEP = 0.045
GLASS_REAR, GLASS_FRONT = -2.02, 1.27   # rear-glass base / windshield base (roof is all glass)
SHUT_LINE_Z = 0.715       # trunk-lid / bumper panel gap on the rear face
VALANCE_Z = 0.50          # top of the dark rear valance/diffuser at the centre


def section(y):
    w = smooth_interp(K_W, y)
    zb = smooth_interp(K_ZB, y)
    zbelt = smooth_interp(K_ZBELT, y)
    zt = smooth_interp(K_ZTOP, y)
    wg = smooth_interp(K_WG, y)
    wr = smooth_interp(K_WR, y)
    g = clamp01((zt - zbelt - 0.06) / 0.25)
    z8 = zbelt + 0.012
    z10 = zt - (0.012 + 0.045 * g)
    pts = [
        (0.0, zb),
        (0.60 * w, zb),
        (0.90 * w, zb + 0.012),
        (0.975 * w, zb + 0.065),
        (w, zb + 0.30 * (zbelt - zb)),
        (0.995 * w, zb + 0.64 * (zbelt - zb)),
        (0.972 * w, zbelt - 0.055),
        (0.935 * w, zbelt),
        (wg, z8),
        (lerp(wg, wr, 0.5) + 0.022 * g, lerp(z8, z10, 0.52)),
        (wr, z10),
        (0.5 * wr, zt - 0.006),
        (0.0, zt),
    ]
    # densify the greenhouse with Catmull-Rom mid points (p = 8.5 ... 11.5) so the glass
    # outline, which is interpolated from vertex values, stays smooth across the roof
    ext = pts + [(-pts[11][0], pts[11][1])]
    dense = pts[:9]
    for i in range(8, 12):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        mid = tuple((-p0[j] + 9 * p1[j] + 9 * p2[j] - p3[j]) / 16 for j in (0, 1))
        dense += [mid, pts[i + 1]]
    return dense, g


P_INDEX = [float(i) for i in range(9)] + [8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]


def smin(a, b, k):
    """Polynomial smooth minimum (rounds the corners of the glass outline)."""
    h = clamp01(0.5 + 0.5 * (b - a) / k)
    return lerp(b, a, h) - k * h * (1 - h)


def glass_field(y, p):
    """Signed glass field at station y, half-profile index p (8 = belt, 10 = roof edge, 12 = top)."""
    sp = STATION_STEP  # station spacing -> field unit ~ one face in either direction
    pc = (12 - p) / 2.0
    yr = GLASS_REAR + 0.13 * pc * pc          # rear-glass base curves forward at its corners
    yf = GLASS_FRONT - 0.10 * pc * pc
    f_top = smin(smin(2 * (p - 9.55), (y - yr) / sp, 2.2), (yf - y) / sp, 2.2)
    t = clamp01((p - 8) / 2.0)
    ce = lerp(C_PILLAR_END[8], C_PILLAR_END[10], t)   # slanted rear edge of the side window
    af = lerp(A_PILLAR_START[8], A_PILLAR_START[10], t)
    f_side = smin(smin(2 * (p - 8.2), (y - ce) / sp, 1.5), (af - y) / sp, 1.5)
    return max(f_top, f_side)


def ring_from_section(pts):
    right = pts
    left = [(-x, z) for (x, z) in reversed(pts[1:-1])]
    return right + left  # 24 points, idx0 bottom centre, idx12 top centre


def bend(y, x):
    br = smoothstep(-1.90, -2.31, y)
    bf = smoothstep(1.85, 2.31, y)
    return 0.15 * br * x * x - 0.24 * bf * x * x


def build_body(mats):
    stations = []
    y = -2.27
    while y < 2.27 - 1e-6:
        stations.append(y)
        y += STATION_STEP
    stations.append(2.27)
    stations = [-2.31] + stations + [2.31]

    rings = []    # list of (list of (x,y,z), kind, ymid-ish)
    for ys in stations:
        pts, g = section(ys)
        ring = [(x, ys + bend(ys, x), z) for (x, z) in ring_from_section(pts)]
        rings.append((ring, "st", ys, g))

    def caps(y_end, direction):
        pts, _ = section(y_end)
        ring = ring_from_section(pts)
        zc = 0.5 * (pts[0][1] + pts[-1][1])
        out = []
        for dy, sc in ((0.022, 0.955), (0.040, 0.875), (0.050, 0.72), (0.055, 0.46), (0.057, 0.18)):
            yy = y_end + direction * dy
            r = []
            for (x, z) in ring:
                xx = x * sc
                zz = zc + (z - zc) * sc
                r.append((xx, yy + bend(y_end, xx), zz))
            out.append((r, "cap", yy, 0.0))
        return out

    rear_caps = caps(-2.31, -1)
    front_caps = caps(2.31, +1)
    all_rings = list(reversed(rear_caps)) + rings + front_caps

    bm = bmesh.new()
    gl = bm.verts.layers.float.new("glass")
    vrings = [[bm.verts.new(p) for p in r[0]] for r in all_rings]
    N = len(vrings[0])
    NH = N // 2
    n_rear = len(rear_caps)
    n_st = len(rings)
    for i in range(len(vrings) - 1):
        ra, rb = vrings[i], vrings[i + 1]
        ia = all_rings[i]
        ib = all_rings[i + 1]
        is_cap = ia[1] == "cap" or ib[1] == "cap"
        ymid = 0.5 * (ia[2] + ib[2])
        gmid = min(ia[3], ib[3]) if not is_cap else 0.0
        for k in range(N):
            k2 = (k + 1) % N
            f = bm.faces.new((ra[k], ra[k2], rb[k2], rb[k]))
            seg = k if k < NH else N - 1 - k
            mat = 0
            if not is_cap and seg in (0, 1):
                mat = 3
            elif is_cap and ymid > 0:
                cz = sum(v.co.z for v in f.verts) / 4
                if cz < smooth_interp(K_ZB, ymid) + 0.06:
                    mat = 2
            f.material_index = mat
            f.smooth = True
    # close the tiny centre holes of the end caps
    for vr in (vrings[0], vrings[-1]):
        f = bm.faces.new(vr)
        f.smooth = True
        f.material_index = 0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Glass field per vertex (>0 = glass). Subdivision interpolates it smoothly, the shader
    # thresholds it -> curved, anti-aliased glass edges independent of the station spacing.
    for ri, (ring, kind, ys, _g) in enumerate(all_rings):
        for k, v in enumerate(vrings[ri]):
            v[gl] = -1.0 if kind == "cap" else glass_field(ys, P_INDEX[k if k <= NH else N - k])

    me = bpy.data.meshes.new("body")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Body", me)
    bpy.context.collection.objects.link(ob)
    for m in (mats["body"], mats["glass"], mats["trim"], mats["under"]):
        me.materials.append(m)
    sub = ob.modifiers.new("subd", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 3
    # wheel arches
    cutters = []
    for ay in (AXLE_R, AXLE_F):
        for sx in (-1, 1):
            bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=WHEEL_R + 0.055, depth=0.55,
                                                location=(sx * 0.93, ay, WHEEL_R + 0.005),
                                                rotation=(0, math.pi / 2, 0))
            c = bpy.context.active_object
            c.name = f"arch_{ay}_{sx}"
            c.data.materials.append(mats["well"])
            c.hide_render = True
            c.hide_viewport = True
            cutters.append(c)
    for c in cutters:
        bo = ob.modifiers.new("arch", "BOOLEAN")
        bo.operation = "DIFFERENCE"
        bo.solver = "EXACT"
        bo.object = c
        try:
            bo.material_mode = "TRANSFER"
        except Exception:
            pass
    es = ob.modifiers.new("split", "EDGE_SPLIT")
    es.split_angle = math.radians(50)
    return ob


def revolve(name, profile, steps=72):
    """profile: list of (axial x, radius) -> surface of revolution about X axis."""
    bm = bmesh.new()
    rings = []
    for i in range(steps):
        a = 2 * math.pi * i / steps
        rings.append([bm.verts.new((x, r * math.cos(a), r * math.sin(a))) for (x, r) in profile])
    for i in range(steps):
        ra, rb = rings[i], rings[(i + 1) % steps]
        for k in range(len(profile) - 1):
            f = bm.faces.new((ra[k], ra[k + 1], rb[k + 1], rb[k]))
            f.smooth = True
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def build_wheel(mats, x, y, side):
    hw = 0.124
    tyre = revolve("tyre", [
        (-0.100, 0.236), (-hw, 0.262), (-hw, 0.300), (-0.108, 0.325), (-0.075, 0.3355),
        (0.0, WHEEL_R + 0.002), (0.075, 0.3355), (0.108, 0.325), (hw, 0.300), (hw, 0.262),
        (0.100, 0.236)])
    tyre.data.materials.append(mats["tyre"])
    # aero cover: shallow dish with a raised outer lip and small centre cap
    cover = revolve("cover", [
        (0.090, 0.0), (0.094, 0.035), (0.093, 0.050), (0.086, 0.060), (0.084, 0.150),
        (0.090, 0.205), (0.100, 0.228), (0.098, 0.240), (0.060, 0.242)])
    cover.data.materials.append(mats["wheel"])
    for o in (tyre, cover):
        o.location = (x, y, WHEEL_R)
        if side < 0:
            o.rotation_euler = (0, 0, math.pi)
        m = o.modifiers.new("subd", "SUBSURF")
        m.levels = 1
        m.render_levels = 2
    return [tyre, cover]


def build_mirror(mats, side):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(side * 1.005, 0.98, 1.0))
    o = bpy.context.active_object
    o.name = f"mirror_{side}"
    o.scale = (0.21, 0.115, 0.11)
    o.rotation_euler = (math.radians(-4), 0, math.radians(side * -8))
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    m = o.modifiers.new("subd", "SUBSURF")
    m.levels = 3
    m.render_levels = 3
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mats["mirror"])
    # stalk
    bpy.ops.mesh.primitive_cube_add(size=1, location=(side * 0.89, 1.0, 0.985))
    s = bpy.context.active_object
    s.name = f"mirror_stalk_{side}"
    s.scale = (0.11, 0.05, 0.035)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    m = s.modifiers.new("subd", "SUBSURF")
    m.levels = 2
    m.render_levels = 2
    bpy.ops.object.shade_smooth()
    s.data.materials.append(mats["trim"])
    return [o, s]


def ribbon_along_rear(body, name, z, height, offset, mat, phi0=197, phi1=343, n=120, origin_y=-1.95):
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body, dg)
    pts = []
    for i in range(n + 1):
        phi = math.radians(lerp(phi0, phi1, i / n))
        d = Vector((math.cos(phi), math.sin(phi), 0.0))
        hit = tree.ray_cast(Vector((0.0, origin_y, z)), d, 5.0)
        if hit[0] is None:
            continue
        loc, nor = hit[0], hit[1]
        pts.append((loc, nor))
    bm = bmesh.new()
    cols = []
    for loc, nor in pts:
        base = loc + nor * offset
        cols.append((bm.verts.new(base + Vector((0, 0, -height / 2))),
                     bm.verts.new(base + Vector((0, 0, height / 2)))))
    for i in range(len(cols) - 1):
        a0, a1 = cols[i]
        b0, b1 = cols[i + 1]
        f = bm.faces.new((a0, b0, b1, a1))
        f.smooth = True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    me.materials.append(mat)
    return ob


def body_material(mats):
    """Paint / tinted glass / dark rear valance in one material, masked by the 'glass'
    vertex field and by object-space position (smooth curved boundaries)."""
    m = bpy.data.materials.new("body")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    def copy_bsdf(src):
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        sb = src.node_tree.nodes["Principled BSDF"]
        for name in ("Base Color", "Roughness", "Metallic", "Specular IOR Level",
                     "Coat Weight", "Coat Roughness"):
            b.inputs[name].default_value = sb.inputs[name].default_value
        return b

    paint, trim = copy_bsdf(mats["paint"]), copy_bsdf(mats["trim"])
    # tinted glass: near-black base + a capped reflection, so grazing panes (windshield)
    # stay smoky instead of mirroring the bright sky
    gbase = nt.nodes.new("ShaderNodeBsdfPrincipled")
    gbase.inputs["Base Color"].default_value = (*hex_lin("#0A0C0F"), 1)
    gbase.inputs["Roughness"].default_value = 0.3
    gbase.inputs["Specular IOR Level"].default_value = 0.0
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Color"].default_value = (0.50, 0.53, 0.58, 1)
    gloss.inputs["Roughness"].default_value = 0.04
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    gmix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Fresnel"], gmix.inputs[0])
    nt.links.new(gbase.outputs[0], gmix.inputs[1])
    nt.links.new(gloss.outputs[0], gmix.inputs[2])
    gadd = nt.nodes.new("ShaderNodeAddShader")
    gl2 = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl2.inputs["Color"].default_value = (0.075, 0.08, 0.09, 1)
    gl2.inputs["Roughness"].default_value = 0.03
    nt.links.new(gmix.outputs[0], gadd.inputs[0])
    nt.links.new(gl2.outputs[0], gadd.inputs[1])
    glass = gadd.outputs[0].node
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "glass"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = -0.03
    mr.inputs["From Max"].default_value = 0.03
    nt.links.new(attr.outputs["Fac"], mr.inputs["Value"])
    mix1 = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix1.inputs[0])
    nt.links.new(paint.outputs[0], mix1.inputs[1])
    nt.links.new(glass.outputs[0], mix1.inputs[2])

    # rear lower valance: y < -2.05 and z below a gently curved line
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])

    def math(op, a, b=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, val in enumerate((a, b)):
            if val is None:
                continue
            if isinstance(val, (int, float)):
                n.inputs[i].default_value = val
            else:
                nt.links.new(val, n.inputs[i])
        return n.outputs[0]

    ax = math("ABSOLUTE", sep.outputs["X"])
    xn = math("DIVIDE", ax, 0.86)
    x4 = math("POWER", xn, 4.0)
    zline = math("SUBTRACT", VALANCE_Z, math("MULTIPLY", x4, 0.11))
    below = math("SUBTRACT", zline, sep.outputs["Z"])               # >0 below the line
    low = math("MULTIPLY", math("ADD", below, 0.003), 1 / 0.006, clamp=True)
    rear = math("MULTIPLY", math("SUBTRACT", -2.02, sep.outputs["Y"]), 1 / 0.10, clamp=True)
    val = math("MULTIPLY", low, rear)
    mix2 = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(val, mix2.inputs[0])
    nt.links.new(mix1.outputs[0], mix2.inputs[1])
    nt.links.new(trim.outputs[0], mix2.inputs[2])

    # trunk / bumper shut line: thin dark groove on rear-facing surfaces only
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nsep.inputs[0])
    facing = math("MULTIPLY", math("SUBTRACT", math("MULTIPLY", nsep.outputs["Y"], -1.0), 0.55),
                  1 / 0.25, clamp=True)
    dz = math("ABSOLUTE", math("SUBTRACT", sep.outputs["Z"], SHUT_LINE_Z))
    line = math("SUBTRACT", 1.0, math("MULTIPLY", math("SUBTRACT", dz, 0.0018), 1 / 0.0015, clamp=True))
    gap = math("MULTIPLY", math("MULTIPLY", line, facing), 0.85)
    groove = nt.nodes.new("ShaderNodeBsdfDiffuse")
    groove.inputs["Color"].default_value = (0.02, 0.022, 0.025, 1)
    mix3 = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(gap, mix3.inputs[0])
    nt.links.new(mix2.outputs[0], mix3.inputs[1])
    nt.links.new(groove.outputs[0], mix3.inputs[2])
    nt.links.new(mix3.outputs[0], out.inputs["Surface"])
    return m


def mirror_material(mats):
    """Body-colour housing; the rear-facing side carries the (dark, reflective) mirror glass."""
    m = bpy.data.materials.new("mirror")
    m.use_nodes = True
    nt = m.node_tree
    paint = nt.nodes["Principled BSDF"]
    sb = mats["paint"].node_tree.nodes["Principled BSDF"]
    for name in ("Base Color", "Roughness", "Coat Weight", "Coat Roughness"):
        paint.inputs[name].default_value = sb.inputs[name].default_value
    glass = nt.nodes.new("ShaderNodeBsdfGlossy")
    glass.inputs["Color"].default_value = (0.30, 0.32, 0.35, 1)
    glass.inputs["Roughness"].default_value = 0.02
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.55
    mr.inputs["From Max"].default_value = -0.62
    nt.links.new(sep.outputs["Y"], mr.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix.inputs[0])
    nt.links.new(paint.outputs[0], mix.inputs[1])
    nt.links.new(glass.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def make_mat(name, base_lin, rough, metal=0.0, coat=0.0, coat_rough=0.05, spec=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base_lin, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    if coat > 0:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = coat_rough
    return m


def build_world(scene):
    """Blue-hour sky: ground / horizon band / zenith by elevation, plus a broad brighter
    glow on the horizon ahead of the car (+Y) that acts as the soft back/rim light."""
    world = bpy.data.worlds.new("blue_hour")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])   # = view direction (unit vector)
    mr = nt.nodes.new("ShaderNodeMapRange")                 # sin(elevation) -1..1 -> 0..1
    mr.inputs["From Min"].default_value = -1.0
    mr.inputs["From Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    cr = ramp.color_ramp
    cr.interpolation = "B_SPLINE"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (*hex_lin(SKY_GROUND), 1)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (*hex_lin(SKY_ZENITH), 1)
    for pos, col in ((0.47, SKY_GROUND), (0.50, SKY_HORIZON), (0.53, SKY_HORIZON), (0.62, "#22375E"),
                     (0.78, SKY_ZENITH)):
        e = cr.elements.new(pos)
        e.color = (*hex_lin(col), 1)

    def m(op, a, b=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, val in enumerate((a, b)):
            if val is None:
                continue
            if isinstance(val, (int, float)):
                n.inputs[i].default_value = val
            else:
                nt.links.new(val, n.inputs[i])
        return n.outputs[0]

    # forward lobe: max(0, dir.y)^3 * exp(-(elev/0.16)^2), elevation >= ~ -0.02
    fwd = m("POWER", m("MAXIMUM", sep.outputs["Y"], 0.0), 3.0)
    band = m("EXPONENT", m("MULTIPLY", m("POWER", m("DIVIDE", m("ABSOLUTE", m("SUBTRACT", sep.outputs["Z"], 0.04)),
                                                                  0.16), 2.0), -1.0))
    above = m("MULTIPLY", m("ADD", sep.outputs["Z"], 0.03), 1 / 0.04, clamp=True)
    lobe = m("MULTIPLY", m("MULTIPLY", fwd, band), above)
    gain = m("ADD", 1.0, m("MULTIPLY", lobe, FWD_GLOW))
    warm = nt.nodes.new("ShaderNodeMix")
    warm.data_type = "RGBA"
    warm.blend_type = "MULTIPLY"
    warm.inputs["Factor"].default_value = 1.0
    nt.links.new(ramp.outputs["Color"], warm.inputs["A"])
    gv = nt.nodes.new("ShaderNodeCombineXYZ")
    for ch in ("X", "Y", "Z"):
        nt.links.new(gain, gv.inputs[ch])
    nt.links.new(gv.outputs[0], warm.inputs["B"])
    nt.links.new(warm.outputs["Result"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = WORLD_STRENGTH
    return bg


def blender_main(argv):
    import argparse
    from bpy_extras.object_utils import world_to_camera_view
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--work", default="")
    ap.add_argument("--preview-dir", default=DEFAULT_PREVIEW)
    ap.add_argument("--samples", type=int, default=384)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--view", default="ego", help="ego | side | top | rear34 (debug views)")
    a = ap.parse_args(argv)
    work = a.work or tempfile.mkdtemp(prefix="ego_car_")
    os.makedirs(work, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # ---------------- materials
    mats = {
        "paint": make_mat("pearl_silver", hex_lin("#B8C1CF"), 0.28, metal=0.45, coat=1.0, coat_rough=0.03),
        "glass": make_mat("tinted_glass", hex_lin("#0B0D10"), 0.06, spec=0.22),
        "trim": make_mat("satin_black", hex_lin("#121418"), 0.40),
        "under": make_mat("underbody", hex_lin("#0A0A0B"), 0.8),
        "well": make_mat("wheel_well", hex_lin("#08090A"), 0.85),
        "tyre": make_mat("tyre", hex_lin("#1A1C21"), 0.62, spec=0.35),
        "wheel": make_mat("aero_cover", hex_lin("#1A1D22"), 0.32, coat=0.6, coat_rough=0.10),
        "lens": make_mat("smoked_lens", hex_lin("#07080A"), 0.06, spec=0.9),
    }
    mats["body"] = body_material(mats)
    mats["mirror"] = mirror_material(mats)
    bar = bpy.data.materials.new("light_bar")
    bar.use_nodes = True
    bb = bar.node_tree.nodes["Principled BSDF"]
    bb.inputs["Base Color"].default_value = (*hex_lin("#0A0B0D"), 1)
    bb.inputs["Roughness"].default_value = 0.15
    bb.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    bb.inputs["Emission Strength"].default_value = 0.0
    mats["bar"] = bar

    # ---------------- car
    body = build_body(mats)
    car = [body]
    for ay in (AXLE_R, AXLE_F):
        for sx in (-1, 1):
            car += build_wheel(mats, sx * TRACK_HALF, ay, sx)
    for sx in (-1, 1):
        car += build_mirror(mats, sx)
    bpy.context.view_layer.update()
    car.append(ribbon_along_rear(body, "lens_band", 0.875, 0.050, 0.0015, mats["lens"]))
    bar_ob = ribbon_along_rear(body, "light_bar", 0.877, 0.0095, 0.0032, mats["bar"], phi0=199, phi1=341)
    car.append(bar_ob)

    # ---------------- road (bounce + reflections; camera-visible only in the shadow pass)
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "road"
    ground.data.materials.append(make_mat("asphalt", hex_lin(ROAD), 0.75, spec=0.3))

    bg = build_world(scene)

    # ---------------- lights
    def area(name, loc, target, size, size_y, energy, color="#FFFFFF", glossy=False):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "RECTANGLE"
        ld.size = size
        ld.size_y = size_y
        ld.energy = energy
        ld.color = hex_lin(color)
        o = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(o)
        o.location = loc
        d = Vector(target) - Vector(loc)
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        o.visible_camera = False
        o.visible_glossy = glossy
        return o

    # broad, low, soft back light from the horizon ahead; dim cool fill from behind the camera;
    # faint cool top light (diffuse only: no softbox shapes in the clearcoat)
    key = area("key_horizon", (0, 9.0, 2.2), (0, 0, 0.9), 12.0, 3.0, KEY_W, "#B9CCF0")
    fill = area("fill_cam", (0, -9.0, 2.6), (0, -2.0, 0.7), 6.0, 3.0, FILL_W, "#C9D2E2")
    top = area("top", (0, -0.5, 6.0), (0, -0.5, 0), 6.0, 8.0, TOP_W, "#9FB6E2")
    # low kicker from behind, linked to the wheels only: tyre backs read under the bumper
    kick = area("kick_wheels", (0, -6.5, 0.35), (0, -1.4, 0.30), 4.0, 0.4, KICK_W, "#B4C4E4")
    wheels = bpy.data.collections.new("wheels")
    scene.collection.children.link(wheels)
    for o in car:
        if o.name.startswith(("tyre", "cover")):
            wheels.objects.link(o)
    try:
        kick.light_linking.receiver_collection = wheels
    except Exception as e:   # very old Blender: unlinked kicker (slightly brighter bumper)
        print("light linking unavailable:", e)
        kick.data.energy *= 0.3
    shade_lights = [key, fill, top, kick]
    # light pass only: thin strip just behind the bar, grazing down onto the bumper
    dg = bpy.context.evaluated_depsgraph_get()
    bar_pts = [bar_ob.matrix_world @ v.co for v in bar_ob.data.vertices]
    bar_y = min(p.y for p in bar_pts if abs(p.x) < 0.1)
    spill = area("bar_spill", (0, bar_y - 0.035, 0.872), (0, bar_y - 0.6, 0.15), 1.55, 0.012, SPILL_W,
                 glossy=True)

    # ---------------- camera (pure pitch, solved distance + shift_y)
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = CAM_LENS
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = CAM_SENSOR
    cam_d.shift_x = 0.0
    cam_d.shift_y = 0.0
    cam_d.clip_start = 0.05
    cam_d.clip_end = 500
    cam = bpy.data.objects.new("cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = (math.radians(90.0 - CAM_PITCH_DEG), 0.0, 0.0)
    r = scene.render
    r.resolution_x = int(W_PX * a.scale)
    r.resolution_y = int(H_PX * a.scale)
    r.resolution_percentage = 100

    pts = []
    y_min = 1e9
    for o in car:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = o.matrix_world
        for i, v in enumerate(me.vertices):
            p = mw @ v.co
            y_min = min(y_min, p.y)
            if i % 2 == 0:
                pts.append(p)
        ev.to_mesh_clear()

    def frame_x(yc):
        cam.location = (0.0, yc, CAM_HEIGHT)
        bpy.context.view_layer.update()
        xs = [world_to_camera_view(scene, cam, p).x for p in pts]
        return max(xs) - min(xs)

    lo, hi = -40.0, y_min - 0.6
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if frame_x(mid) > FRAME_W:
            hi = mid
        else:
            lo = mid
    yc = round(0.5 * (lo + hi), 3)
    cam.location = (0.0, yc, CAM_HEIGHT)
    bpy.context.view_layer.update()
    g = world_to_camera_view(scene, cam, Vector((0.0, y_min - SHADOW_REACH, 0.0))).y
    cam_d.shift_y = round((g - SHADOW_BOTTOM) * H_PX / W_PX, 5)
    bpy.context.view_layer.update()
    camj = {
        "lens_mm": CAM_LENS, "sensor_width_mm": CAM_SENSOR, "sensor_fit": "HORIZONTAL",
        "width_px": W_PX, "height_px": H_PX,
        "shift_x": 0.0, "shift_y": cam_d.shift_y,
        "shift_units": "fraction of the larger image dimension (width), Blender convention; "
                       "+shift_y moves the frame up (content down)",
        "loc": [0.0, yc, CAM_HEIGHT],
        "pitch_deg": CAM_PITCH_DEG,
        "rotation": "pure downward pitch about +X, no yaw/roll; Blender rotation_euler XYZ = "
                    f"({90.0 - CAM_PITCH_DEG:g} deg, 0, 0); view direction (0, cos p, -sin p)",
        "world": "metres, z up, ground plane z = 0, car centred at the origin (x = 0 centre line), "
                 "car front +Y, camera behind the car looking toward +Y",
        "car_extent_m": {"y_rear": round(y_min, 3), "y_front": round(-y_min, 3)},
        "projection": "col = W/2 + W*(k*X_c/F + shift_x); row = H/2 - W*(k*Y_c/F - shift_y); "
                      "k = lens_mm/sensor_width_mm, X_c right, Y_c up, F forward (camera space)",
    }
    with open(os.path.join(work, "camera.json"), "w") as fh:
        json.dump(camj, fh, indent=2)
    print(f"camera y={yc} m (rear gap {y_min - yc:.2f} m), shift_y={cam_d.shift_y}, "
          f"width frac={frame_x(yc):.4f}, rear y_min={y_min:.3f}")
    print("horizon row", horizon_row(camj), "shadow-end row check",
          project(camj, 0, y_min - SHADOW_REACH, 0))

    if a.view != "ego":
        cam_d.type = "ORTHO" if a.view in ("side", "top") else "PERSP"
        cam_d.ortho_scale = 5.6
        cam_d.shift_y = 0
        poses = {"side": ((9, 0, 0.7), (0, 0, 0.7)), "top": ((0, 0, 9), (0, 0.001, 0)),
                 "rear34": ((-5.5, -7.5, 3.0), (0, -0.3, 0.6))}
        loc, tgt = poses[a.view]
        cam.location = loc
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        if a.view == "rear34":
            cam_d.lens = 60

    # ---------------- render settings
    r.engine = "CYCLES"
    r.film_transparent = True
    r.filter_size = 1.2
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if not a.cpu:
        try:
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.glossy_bounces = 4
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.sample_clamp_indirect = 3.0
    vs = scene.view_settings

    def agx():
        vs.view_transform = "AgX"
        for look in ("AgX - Medium High Contrast", "Medium High Contrast", "None"):
            try:
                vs.look = look
                break
            except Exception:
                pass
        vs.exposure = EXPOSURE

    def save_npy(name, fmt):
        """Save the render result and dump its pixels (as stored) to <name>.npy."""
        path = os.path.join(work, name + (".exr" if fmt == "EXR" else ".png"))
        s = r.image_settings
        if fmt == "EXR":
            s.file_format = "OPEN_EXR"
            s.color_mode = "RGBA"
            s.color_depth = "32"
        else:
            s.file_format = "PNG"
            s.color_mode = "RGBA"
            s.color_depth = "16"
        bpy.data.images["Render Result"].save_render(filepath=path, scene=scene)
        im = bpy.data.images.load(path, check_existing=False)
        im.alpha_mode = "CHANNEL_PACKED"                 # no (un)premultiplication on read
        if fmt != "EXR":
            im.colorspace_settings.name = "Non-Color"    # raw display-referred values
        import numpy as np
        buf = np.empty(im.size[0] * im.size[1] * 4, dtype=np.float32)
        im.pixels.foreach_get(buf)
        arr = buf.reshape(im.size[1], im.size[0], 4)[::-1]   # top row first
        np.save(os.path.join(work, name + ".npy"), arr)
        bpy.data.images.remove(im)
        return path

    agx()
    if a.view != "ego":
        r.film_transparent = False
        ground.visible_camera = True
        scene.cycles.samples = a.samples
        bpy.ops.render.render(write_still=False)
        s = r.image_settings
        s.file_format, s.color_mode, s.color_depth = "PNG", "RGB", "8"
        out = os.path.join(a.preview_dir or work, f"view_{a.view}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bpy.data.images["Render Result"].save_render(filepath=out, scene=scene)
        return

    # 1) body: car only (road invisible to the camera), bar off
    spill.hide_render = True
    ground.visible_camera = False
    scene.cycles.samples = a.samples
    bpy.ops.render.render(write_still=False)
    save_npy("body", "PNG")
    print("rendered body")

    # 2) shadow: car invisible to the camera, road = shadow catcher under a uniform dome (AO-like)
    for o in shade_lights:
        o.hide_render = True
    dome = bpy.data.worlds.new("dome")
    dome.use_nodes = True
    dome.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    dome.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    sky_world = scene.world
    scene.world = dome
    ground.visible_camera = True
    ground.is_shadow_catcher = True
    for o in car:
        o.visible_camera = False
    scene.cycles.samples = max(64, a.samples // 3)
    bpy.ops.render.render(write_still=False)
    save_npy("shadow", "PNG")
    print("rendered shadow")

    # 3) light: only the bar emits (white) + spill strip; world off; linear output
    scene.world = sky_world
    spill.hide_render = False
    ground.is_shadow_catcher = False
    ground.hide_render = True
    for o in car:
        o.visible_camera = True
    bg.inputs["Strength"].default_value = 0.0
    bb.inputs["Emission Strength"].default_value = BAR_EMIT
    r.film_transparent = False
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    scene.cycles.samples = max(64, a.samples // 2)
    bpy.ops.render.render(write_still=False)
    save_npy("light", "EXR")
    print("rendered light")

    os.makedirs(a.preview_dir, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(work, "camera.json"), os.path.join(a.preview_dir, "camera.json"))
    if not a.no_post:
        py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else "python3"
        cmd = [py, os.path.abspath(__file__), "--post", "--work", work, "--out", a.out_dir,
               "--preview-dir", a.preview_dir]
        subprocess.run(cmd, check=True)
    print("work dir:", work)


if __name__ == "__main__":
    if IN_BLENDER:
        blender_main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    else:
        post_main(sys.argv[1:])
