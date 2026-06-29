"""
Diffusion Forcing vs Standard Diffusion — modern animated SVG (warm theme).

Grounded in Chen et al., "Diffusion Forcing: Next-token Prediction Meets
Full-Sequence Diffusion" (NeurIPS 2024, arXiv:2407.01392):

  Standard full-sequence diffusion  — "the noise level is identical across all
    tokens" (sec 1). The whole window is denoised together from pure noise
    (k = K) to clean (k = 0). Top panel: every frame shares one level and
    denoises in lockstep.

  Diffusion Forcing  — "each token is associated with a random, independent
    noise level" (sec 3). This lets you "stably roll out long sequences ... by
    updating the latents using the previous latent associated with slightly
    noisy tokens" and keep the future uncertain (sec 3.1). Bottom panel: a
    denoising wavefront sweeps forward, so past frames are clean context
    (k = 0), the next frame is denoising, and the future stays noisy (k = K) —
    which is exactly independent per-frame noise.

Noise is drawn as feTurbulence grain (opacity proportional to level k) plus a
per-frame level bar. SMIL <animate> drives everything; runs inside an <img>.

Set DF_DEBUG=1 to emit a static representative still (for layout checking).
"""
import os

NOISE = "#c0613c"
INK = "#2c2722"
MUTE = "#7c7565"
TEAL_EC = "#5d8a8a"
TERRA_EC = "#b3552c"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

DEBUG = os.environ.get("DF_DEBUG") == "1"

W, H = 1160, 518
N = 6
FW, FH, GAP = 138, 92, 16
STEP = FW + GAP
X0 = (W - (N * FW + (N - 1) * GAP)) // 2      # centered row
MAXBAR = 30
D = 7.0                                         # loop seconds

SCENE = ("#eee9de", "#e3ddcc", "#b3aa95")       # neutral stone frame
GRAIN_MAX = 0.82


def frame(fx, fy, label, op0, op_anim):
    sky, floor, obj = SCENE
    gid = f"g{fx}{fy}"
    s = [f'<g transform="translate({fx},{fy})">',
         f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="{sky}"/><stop offset="1" stop-color="{floor}"/>'
         f'</linearGradient>',
         '<g clip-path="url(#fclip)">',
         f'<rect width="{FW}" height="{FH}" fill="url(#{gid})"/>',
         f'<line x1="0" y1="{FH*0.62:.0f}" x2="{FW}" y2="{FH*0.62:.0f}" '
         f'stroke="{obj}" stroke-width="1.4" opacity="0.55"/>',
         f'<circle cx="{FW*0.66:.0f}" cy="{FH*0.40:.0f}" r="10" fill="{obj}" opacity="0.9"/>',
         f'<rect width="{FW}" height="{FH}" filter="url(#grain)" opacity="{op0:.3f}">{op_anim}</rect>',
         '</g>',
         f'<rect width="{FW}" height="{FH}" rx="12" fill="none" stroke="#cdbfa0" stroke-width="1.8"/>',
         f'<text x="{FW/2}" y="{FH*0.40+6:.0f}" text-anchor="middle" font-size="16" '
         f'font-style="italic" font-weight="600" fill="#5c5446" opacity="0.85">{label}</text>',
         '</g>']
    return "\n".join(s)


def bar(cx, base, h0, anim):
    return (f'<rect x="{cx-20}" y="{base-h0:.1f}" width="40" height="{h0:.1f}" rx="2" '
            f'fill="{NOISE}" opacity="0.82">{anim}</rect>')


def anim_op(keytimes, values):
    if DEBUG:
        return ''
    return (f'<animate attributeName="opacity" values="{values}" keyTimes="{keytimes}" '
            f'dur="{D}s" repeatCount="indefinite" calcMode="spline" '
            f'keySplines="{";".join(["0.4 0 0.6 1"]*(len(keytimes.split(";"))-1))}"/>')


def anim_bar(base, keytimes, heights):
    if DEBUG:
        return ''
    ys = ";".join(f"{base-h:.1f}" for h in heights)
    hs = ";".join(f"{h:.1f}" for h in heights)
    ks = ";".join(["0.4 0 0.6 1"] * (len(keytimes.split(";")) - 1))
    return (f'<animate attributeName="height" values="{hs}" keyTimes="{keytimes}" dur="{D}s" '
            f'repeatCount="indefinite" calcMode="spline" keySplines="{ks}"/>'
            f'<animate attributeName="y" values="{ys}" keyTimes="{keytimes}" dur="{D}s" '
            f'repeatCount="indefinite" calcMode="spline" keySplines="{ks}"/>')


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}" role="img" aria-label="Diffusion forcing vs standard diffusion">']

# defs
parts.append('<defs>')
parts.append('<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0" stop-color="#fdf7e9"/><stop offset="1" stop-color="#f6ecd2"/>'
             '</linearGradient>')
parts.append(f'<clipPath id="fclip"><rect width="{FW}" height="{FH}" rx="12"/></clipPath>')
parts.append('<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.9" '
             'numOctaves="3" seed="7" stitchTiles="stitch"/>'
             '<feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  '
             '0.42 0.42 0.42 0 0"/></filter>')
for col in (NOISE, MUTE, TERRA_EC):
    parts.append(f'<marker id="ah-{col[1:]}" markerWidth="9" markerHeight="9" refX="7" refY="3" '
                 f'orient="auto" markerUnits="userSpaceOnUse">'
                 f'<path d="M0,0 L7.5,3 L0,6 Z" fill="{col}"/></marker>')
parts.append('</defs>')

parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e7dcc0" stroke-width="1.5"/>')
parts.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="19" font-weight="700" '
             f'fill="{INK}">Diffusion Forcing vs Standard Diffusion</text>')

# ════════════════════ PANEL A — standard ════════════════════
ay = 84
parts.append(f'<text x="{X0}" y="68" font-size="14" font-weight="700" fill="{INK}">'
             f'Standard Diffusion <tspan font-weight="400" fill="{MUTE}" font-style="italic">'
             f'· one shared noise level for the whole sequence</tspan></text>')
# unison denoise: high -> clean -> hold -> re-noise
a_kt = "0;0.10;0.52;0.84;1"
a_op = (f"{GRAIN_MAX:.2f};{GRAIN_MAX:.2f};0.04;0.04;{GRAIN_MAX:.2f}")
a_heights = [MAXBAR, MAXBAR, 2, 2, MAXBAR]
abase = ay + FH + 42
for i in range(N):
    fx = X0 + i * STEP
    n0 = 0.5 if DEBUG else GRAIN_MAX
    parts.append(frame(fx, ay, f"f{i+1}", n0 * GRAIN_MAX if DEBUG else GRAIN_MAX,
                       anim_op(a_kt, a_op)))
    parts.append(bar(fx + FW/2, abase, (0.5*MAXBAR if DEBUG else MAXBAR), anim_bar(abase, a_kt, a_heights)))
# shared denoise direction label
parts.append(f'<text x="{X0 + N*STEP - GAP - 4}" y="68" text-anchor="end" font-size="11" '
             f'font-style="italic" fill="{NOISE}">denoise k = K &#8594; k = 0, all frames together</text>')
parts.append(f'<text x="{W/2}" y="{abase+22}" text-anchor="middle" font-size="12" fill="{INK}">'
             f'The whole window denoises in lockstep, so a new frame cannot be added without '
             f'denoising the entire sequence again.</text>')

# divider
parts.append(f'<line x1="40" y1="{abase+40}" x2="{W-40}" y2="{abase+40}" stroke="#e7dcc0" stroke-width="1.3"/>')

# ════════════════════ PANEL B — diffusion forcing ════════════════════
by = abase + 104
parts.append(f'<text x="{X0}" y="{by-38}" font-size="14" font-weight="700" fill="{INK}">'
             f'Diffusion Forcing <tspan font-weight="400" fill="{MUTE}" font-style="italic">'
             f'· an independent noise level per frame</tspan></text>')
# forward denoising wavefront: frame i cleans at staggered time -> pyramid gradient
ST = 0.105       # stagger fraction
DW = 0.17        # denoise window fraction
bbase = by + FH + 42
# debug gradient (clean left -> noisy right)
dbg = [0.05, 0.22, 0.40, 0.56, 0.70, 0.82]
for i in range(N):
    fx = X0 + i * STEP
    ts = 0.06 + i * ST
    te = ts + DW
    kt = f"0;{ts:.3f};{te:.3f};0.9;1"
    op = f"{GRAIN_MAX:.2f};{GRAIN_MAX:.2f};0.04;0.04;{GRAIN_MAX:.2f}"
    heights = [MAXBAR, MAXBAR, 2, 2, MAXBAR]
    n0 = dbg[i] if DEBUG else GRAIN_MAX
    h0 = dbg[i]*MAXBAR if DEBUG else MAXBAR
    parts.append(frame(fx, by, f"f{i+1}", n0, anim_op(kt, op)))
    parts.append(bar(fx + FW/2, bbase, h0, anim_bar(bbase, kt, heights)))

# denoise-sweep arrow above the row
ax1, ax2 = X0 + 6, X0 + N*STEP - GAP - 6
parts.append(f'<line x1="{ax1}" y1="{by-14}" x2="{ax2}" y2="{by-14}" stroke="{NOISE}" '
             f'stroke-width="1.6" marker-end="url(#ah-{NOISE[1:]})" opacity="0.8"/>')
parts.append(f'<text x="{(ax1+ax2)/2}" y="{by-18}" text-anchor="middle" font-size="10.5" '
             f'font-style="italic" fill="{NOISE}">denoising wavefront sweeps forward</text>')

# context / future annotations under the bars
parts.append(f'<text x="{X0 + FW}" y="{bbase+22}" text-anchor="middle" font-size="11" '
             f'font-style="italic" fill="{TEAL_EC}">clean context · k = 0</text>')
parts.append(f'<text x="{X0 + 4*STEP + FW/2}" y="{bbase+22}" text-anchor="middle" font-size="11" '
             f'font-style="italic" fill="{NOISE}">uncertain future · k = K</text>')
parts.append(f'<text x="{W/2}" y="{bbase+44}" text-anchor="middle" font-size="12" fill="{INK}">'
             f'Past frames stay clean as context while the next frame is denoised and the future '
             f'stays noisy, so the model rolls out one frame at a time and stays stable.</text>')

parts.append('</svg>')

out_name = 'diffusion_forcing_debug.svg' if DEBUG else 'diffusion_forcing.svg'
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', out_name)
if DEBUG:
    out = os.path.join('/private/tmp/claude-501/-Users-kaushikprakash-Documents-NanoNAV/'
                       'a1fc6232-f5f9-4b31-a5a2-bd5b367841d6/scratchpad', out_name)
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
