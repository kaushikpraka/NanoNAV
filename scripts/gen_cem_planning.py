"""
CEM planning diagram — modern animated SVG (warm cream/terracotta theme).

Left  — the Cross-Entropy Method search: a sampling distribution N(mu, sigma)
        over candidate action plans. Across iterations it narrows and shifts
        toward the best plans (closest to the goal in latent space). Four
        iteration snapshots cross-fade in a loop; elite samples are highlighted.
Right — the five-step MPC + CEM loop, with the inner "repeat" loop over
        iterations and the outer "replan" loop after executing a chunk.

Self-contained: SMIL opacity animations cross-fade the iteration snapshots; runs
inside an <img> with no JS, crisp at any zoom.
"""
import os

TEAL_FC, TEAL_EC, TEAL_TX = "#d3e3e3", "#5d8a8a", "#2d4a4a"
TERRA_FC, TERRA_EC, TERRA_TX = "#f4d9c1", "#b3552c", "#7a3a1a"
PLUM_FC, PLUM_EC, PLUM_TX = "#e2d2e6", "#845a8e", "#523460"
STONE = "#b3aa95"
INK = "#2c2722"
MUTE = "#7c7565"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

W, H = 1160, 540
GX, GY = 525, 158          # goal location in the plot

# a fixed unit cloud reused across snapshots so it reads as one contracting cloud
CLOUD = [(-0.62, -0.28), (0.12, -0.58), (0.52, -0.12), (-0.32, 0.42),
         (0.36, 0.46), (-0.06, 0.06), (0.66, 0.22), (-0.52, 0.16),
         (0.22, -0.22), (-0.22, -0.5), (0.46, -0.42), (-0.36, -0.06)]
ELITES = {1, 2, 10}        # samples on the goal side, kept each iteration


def opacity_window(i):
    wins = {
        0: ('1;1;0;0;1', '0;0.22;0.25;0.98;1'),
        1: ('0;0;1;1;0;0', '0;0.25;0.28;0.47;0.50;1'),
        2: ('0;0;1;1;0;0', '0;0.50;0.53;0.72;0.75;1'),
        3: ('0;0;1;1;0', '0;0.75;0.78;0.97;1'),
    }
    v, kt = wins[i]
    return (f'<animate attributeName="opacity" values="{v}" keyTimes="{kt}" '
            f'dur="8s" repeatCount="indefinite"/>')


def snapshot(i, cx, cy, rx, ry, label, winner=False):
    s = [f'<g opacity="0">{opacity_window(i)}']
    s.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{TEAL_FC}" '
             f'fill-opacity="0.4" stroke="{TEAL_EC}" stroke-width="1.8"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{TEAL_EC}"/>')
    for k, (ux, uy) in enumerate(CLOUD):
        x, y = cx + ux * rx, cy + uy * ry
        if k in ELITES:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{TERRA_EC}" '
                     f'stroke="#fffdf6" stroke-width="1.3"/>')
        else:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{STONE}" '
                     f'fill-opacity="0.9"/>')
    if winner:
        s.append(f'<circle cx="{GX-9}" cy="{GY+9}" r="6" fill="{TERRA_EC}" '
                 f'stroke="#fffdf6" stroke-width="1.6"/>')
        s.append(f'<text x="{GX-9}" y="{GY+62}" text-anchor="middle" font-size="11" '
                 f'font-style="italic" fill="{TERRA_TX}">winning plan</text>')
    s.append(f'<text x="330" y="128" text-anchor="middle" font-size="12.5" '
             f'fill="{INK}">{label}</text>')
    s.append('</g>')
    return "\n".join(s)


def badge(n, cx, cy):
    return (f'<circle cx="{cx}" cy="{cy}" r="15" fill="{TERRA_FC}" stroke="{TERRA_EC}" '
            f'stroke-width="1.7"/>'
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="14" '
            f'font-weight="700" fill="{TERRA_TX}">{n}</text>')


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}" role="img" aria-label="CEM planning loop">'
]

# ── defs ───────────────────────────────────────────────────────────────────
parts.append('<defs>')
parts.append('<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0" stop-color="#fdf7e9"/><stop offset="1" stop-color="#f6ecd2"/>'
             '</linearGradient>')
parts.append(f'<radialGradient id="score" gradientUnits="userSpaceOnUse" '
             f'cx="{GX}" cy="{GY}" r="380">'
             f'<stop offset="0" stop-color="{TERRA_EC}" stop-opacity="0.32"/>'
             f'<stop offset="1" stop-color="{TERRA_EC}" stop-opacity="0"/></radialGradient>')
parts.append(f'<clipPath id="plot"><rect x="70" y="100" width="530" height="360" rx="10"/></clipPath>')
for col in (TERRA_EC, PLUM_EC, MUTE):
    parts.append(f'<marker id="ah-{col[1:]}" markerWidth="9" markerHeight="9" refX="7" '
                 f'refY="3" orient="auto" markerUnits="userSpaceOnUse">'
                 f'<path d="M0,0 L7.5,3 L0,6 Z" fill="{col}"/></marker>')
parts.append('</defs>')

# ── card + title ───────────────────────────────────────────────────────────
parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e7dcc0" stroke-width="1.5"/>')
parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="19" '
             f'font-weight="700" fill="{INK}">CEM planning · sample, score, refit, repeat</text>')
parts.append(f'<line x1="636" y1="74" x2="636" y2="500" stroke="#e7dcc0" stroke-width="1.3"/>')

# ── plot: score field + goal + contracting search ──────────────────────────
parts.append('<g clip-path="url(#plot)">')
parts.append(f'<rect x="70" y="100" width="530" height="360" fill="#fbf4e4"/>')
parts.append(f'<rect x="70" y="100" width="530" height="360" fill="url(#score)"/>')
parts.append('</g>')
parts.append(f'<rect x="70" y="100" width="530" height="360" rx="10" fill="none" '
             f'stroke="#e0d4b6" stroke-width="1.4"/>')
parts.append(f'<text x="335" y="480" text-anchor="middle" font-size="11.5" '
             f'font-style="italic" fill="{MUTE}">space of candidate action plans · warmer = closer to goal in latent space</text>')

# the four iteration snapshots (contracting toward the goal)
parts.append(snapshot(0, 205, 372, 120, 84, 'iteration 1 · wide search'))
parts.append(snapshot(1, 322, 300, 88, 64, 'iteration 2 · refit to elites'))
parts.append(snapshot(2, 430, 226, 56, 42, 'iteration 3 · narrowing'))
parts.append(snapshot(3, 500, 178, 28, 22, 'iteration 4 · converged', winner=True))

# goal marker (always on top)
for r, op in ((16, 0.25), (10, 0.45), (4.5, 1)):
    parts.append(f'<circle cx="{GX}" cy="{GY}" r="{r}" fill="{TERRA_EC}" '
                 f'fill-opacity="{op}"/>')
parts.append(f'<text x="{GX}" y="{GY-26}" text-anchor="middle" font-size="12" '
             f'font-style="italic" fill="{TERRA_TX}">goal latent '
             f'z<tspan baseline-shift="sub" font-size="9">g</tspan></text>')

# ── right: the five-step loop ──────────────────────────────────────────────
steps = [
    ("1", "Encode the current frame and goal once"),
    ("2", "Sample candidate plans, roll each through the world model"),
    ("3", "Score each imagined endpoint by latent distance to goal"),
    ("4", "Keep the top fraction, resample around them, repeat"),
    ("5", "Execute the first chunk of the winner, then replan"),
]
bx, tx = 722, 748
ys = [150, 222, 294, 366, 430]
parts.append(f'<text x="{bx-15}" y="112" font-size="13.5" font-weight="700" '
             f'fill="{INK}">The planning loop</text>')
for (n, txt), y in zip(steps, ys):
    parts.append(badge(n, bx, y))
    parts.append(f'<text x="{tx}" y="{y+5}" font-size="13" fill="{INK}">{txt}</text>')

# inner CEM-iteration loop (step 4 -> step 2), drawn in the left gutter
parts.append(f'<path d="M{bx-18},{ys[3]} C{bx-54},{ys[3]} {bx-54},{ys[1]} {bx-18},{ys[1]}" '
             f'fill="none" stroke="{PLUM_EC}" stroke-width="1.8" stroke-dasharray="5 4" '
             f'marker-end="url(#ah-{PLUM_EC[1:]})"/>')
parts.append(f'<text transform="rotate(-90 {bx-62} {(ys[1]+ys[3])/2})" '
             f'x="{bx-62}" y="{(ys[1]+ys[3])/2}" text-anchor="middle" font-size="10.5" '
             f'font-style="italic" fill="{PLUM_TX}">repeat · CEM iterations</text>')

# outer MPC replan note under step 5
parts.append(f'<text x="{tx}" y="{ys[4]+24}" font-size="11.5" font-style="italic" '
             f'fill="{TERRA_TX}">&#8635; then replan from the new observation</text>')

parts.append('</svg>')

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'cem_planning.svg')
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
