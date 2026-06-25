#!/usr/bin/env python3
"""Render the first failed closed-loop run (SD-VAE latent objective) to an MP4.

Reads results/mpc_nearfan.rrd (goal=nearfan, the run where dist_to_goal hovered ~45
for the whole run while the robot wandered and yaw flip-flopped) and composites, per
logged step: the live camera frame, the goal frame, and a readout bar with a growing
dist-to-goal trace + the vx / yaw commands. Output → docs/assets/first_closedloop_vae.mp4.

Self-contained: rerun dataframe API for reading, PIL for compositing, imageio(+bundled
ffmpeg) for H.264. No system ffmpeg needed.
"""
import glob
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np
import rerun.dataframe as rrd
from PIL import Image, ImageDraw, ImageFont

RRD = "/workspace/results/mpc_nearfan.rrd"
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets", "first_closedloop_vae.mp4")
FPS = 3
END_HOLD = 6  # extra duplicate frames on the last step

# palette tuned to the site (solarized-ish warm)
BG = (253, 246, 227)
INK = (40, 40, 40)
MUTED = (110, 110, 110)
ACCENT = (179, 85, 44)
FAINT = (210, 200, 175)
GRID = (224, 214, 190)

IMG_W, IMG_H = 360, 270      # display size per camera panel
GAP = 18
MARGIN = 22
LABEL_H = 24
PANEL_H = 132                # readout bar height
CANVAS_W = MARGIN * 2 + IMG_W * 2 + GAP
CANVAS_H = MARGIN + LABEL_H + IMG_H + 18 + PANEL_H + MARGIN

_FONTDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
def _font(name, size):
    return ImageFont.truetype(os.path.join(_FONTDIR, name), size)
F_LABEL = _font("DejaVuSans-Bold.ttf", 15)
F_BIG = _font("DejaVuSans-Bold.ttf", 26)
F_SMALL = _font("DejaVuSans.ttf", 13)
F_TINY = _font("DejaVuSans.ttf", 11)


def decode(buf, fmt):
    """rerun ImageBuffer (raw row) + ImageFormat dict -> HxWx3 uint8."""
    f = fmt[0] if isinstance(fmt, list) else fmt
    w, h = int(f["width"]), int(f["height"])
    arr = np.asarray(buf).reshape(-1).astype(np.uint8)
    return arr[: h * w * 3].reshape(h, w, 3)


def load_rows():
    rec = rrd.load_recording(RRD)
    view = rec.view(index="step", contents={
        "/live": ["ImageBuffer", "ImageFormat"],
        "/goal": ["ImageBuffer", "ImageFormat"],
        "/dist_to_goal": ["Scalar"], "/cmd/vx": ["Scalar"], "/cmd/theta_deg": ["Scalar"],
    })
    d = view.select().read_all().to_pydict()
    scal = lambda col, i: float(d[col][i][0])
    rows = []
    for i in range(len(d["step"])):
        rows.append(dict(
            step=int(d["step"][i]),
            live=decode(d["/live:ImageBuffer"][i], d["/live:ImageFormat"][i]),
            goal=decode(d["/goal:ImageBuffer"][i], d["/goal:ImageFormat"][i]),
            dist=scal("/dist_to_goal:Scalar", i),
            vx=scal("/cmd/vx:Scalar", i),
            theta=scal("/cmd/theta_deg:Scalar", i),
        ))
    return rows


def fit(arr):
    return Image.fromarray(arr).resize((IMG_W, IMG_H), Image.LANCZOS)


def draw_trace(dr, x, y, w, h, dists, upto, dmin, dmax):
    """Growing dist-to-goal line; flatness is the whole point."""
    dr.rectangle([x, y, x + w, y + h], outline=FAINT, width=1)
    # reach-thresh reference would sit far below; annotate the band instead
    def px(i):
        return x + (w * i / (len(dists) - 1))
    def py(v):
        return y + h - h * (v - dmin) / (dmax - dmin)
    # faint full path
    pts_all = [(px(i), py(v)) for i, v in enumerate(dists)]
    dr.line(pts_all, fill=GRID, width=2)
    # solid up to current
    pts = [(px(i), py(dists[i])) for i in range(upto + 1)]
    if len(pts) >= 2:
        dr.line(pts, fill=ACCENT, width=3)
    dr.ellipse([px(upto) - 4, py(dists[upto]) - 4, px(upto) + 4, py(dists[upto]) + 4],
               fill=ACCENT)
    dr.text((x, y + h + 4), "distance-to-goal (SD-VAE latent L2)", font=F_TINY, fill=MUTED)
    dr.text((x + w - 2, y - 2), f"{dmax:.0f}", font=F_TINY, fill=MUTED, anchor="ra")
    dr.text((x + w - 2, y + h - 12), f"{dmin:.0f}", font=F_TINY, fill=MUTED, anchor="ra")


def render(rows):
    dists = [r["dist"] for r in rows]
    dmin, dmax = min(dists) - 2, max(dists) + 2
    ix, iy = MARGIN, MARGIN + LABEL_H
    frames = []
    for k, r in enumerate(rows):
        cv = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(cv)
        # labels
        dr.text((ix, MARGIN), "LIVE  ·  what the robot sees", font=F_LABEL, fill=INK)
        dr.text((ix + IMG_W + GAP, MARGIN), "GOAL  ·  where it's trying to go",
                font=F_LABEL, fill=INK)
        # images
        cv.paste(fit(r["live"]), (ix, iy))
        cv.paste(fit(r["goal"]), (ix + IMG_W + GAP, iy))
        dr.rectangle([ix, iy, ix + IMG_W, iy + IMG_H], outline=FAINT, width=1)
        dr.rectangle([ix + IMG_W + GAP, iy, ix + 2 * IMG_W + GAP, iy + IMG_H],
                     outline=FAINT, width=1)
        # readout panel
        py0 = iy + IMG_H + 18
        # left: big step + dist readout
        dr.text((ix, py0), f"step {r['step']:>2d}", font=F_SMALL, fill=MUTED)
        dr.text((ix, py0 + 16), f"dist {r['dist']:.1f}", font=F_BIG, fill=ACCENT)
        sign = "+" if r["theta"] >= 0 else "−"
        dr.text((ix, py0 + 52),
                f"vx {r['vx']:.3f} m/s     ω {sign}{abs(r['theta']):.1f}°/s",
                font=F_SMALL, fill=INK)
        dr.text((ix, py0 + 72),
                "forward speed nonzero, yaw sign flips — yet the metric doesn't move",
                font=F_TINY, fill=MUTED)
        # right: the trace
        tw, th = IMG_W, PANEL_H - 34
        draw_trace(dr, ix + IMG_W + GAP, py0, tw, th, dists, k, dmin, dmax)
        frames.append(np.asarray(cv))
    frames.extend([frames[-1]] * END_HOLD)
    return frames


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} steps; dist {rows[0]['dist']:.1f} -> {rows[-1]['dist']:.1f} "
          f"(min {min(r['dist'] for r in rows):.1f}, max {max(r['dist'] for r in rows):.1f})")
    frames = render(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=FPS, codec="libx264", quality=8,
                    macro_block_size=2, output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {os.path.normpath(OUT)} ({os.path.getsize(OUT) // 1024} KB, "
          f"{len(frames)} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
