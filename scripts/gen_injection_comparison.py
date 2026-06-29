"""
Action-conditioning diagram — Additive injection vs AdaLN injection.

Modern animated SVG, warm cream/terracotta palette to match the site.

Left  — Additive: the action enters as a residual W*a added to the stream. The
        model can drive W -> 0, closing the gradient path; the action-embedding
        RMS bar collapses to ~0.0028 (atrophy).
Right — AdaLN: the action predicts gamma(a), beta(a) that scale/shift every
        LayerNorm, multiplicatively gating the whole feature map. It cannot be
        zeroed without collapsing activations; the RMS bar holds at ~0.2.

SMIL <animate> drives the W->0 fade, the RMS bars and the gamma-gated feature
map; runs inside an <img> with no JS and stays crisp at any zoom.
"""
import os

TEAL_FC, TEAL_EC, TEAL_TX = "#d3e3e3", "#5d8a8a", "#2d4a4a"
TERRA_FC, TERRA_EC, TERRA_TX = "#f4d9c1", "#b3552c", "#7a3a1a"
PLUM_FC, PLUM_EC, PLUM_TX = "#e2d2e6", "#845a8e", "#523460"
NOISE = "#c0613c"
DEAD = "#a79f8d"
INK = "#2c2722"
MUTE = "#7c7565"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

W, H = 1160, 520
RMS_BASE, RMS_MAX = 404, 132   # meter baseline y and full-scale height (RMS≈0.25)


def arrow(d, color, lw=2.2, cls="", dash=""):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    c = f' class="{cls}"' if cls else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{lw}" '
            f'stroke-linecap="round" marker-end="url(#ah-{color[1:]})"{dd}{c}/>')


def opcirc(cx, cy, sym, ec):
    return (f'<circle cx="{cx}" cy="{cy}" r="15" fill="#fffdf6" stroke="{ec}" '
            f'stroke-width="2" filter="url(#soft)"/>'
            f'<text x="{cx}" y="{cy+5.5}" text-anchor="middle" font-size="18" '
            f'font-weight="700" fill="{ec}">{sym}</text>')


def pill(cx, y, w, text, fc, ec, tx, fs=15):
    x = cx - w / 2
    return (f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="9" fill="{fc}" '
            f'stroke="{ec}" stroke-width="1.6" filter="url(#soft)"/>'
            f'<text x="{cx}" y="{y+22}" text-anchor="middle" font-size="{fs}" '
            f'fill="{tx}">{text}</text>')


def labelbox(x, y, w, h, text, fc, ec, tx, fs=13, bold=True):
    fw = 'font-weight="700"' if bold else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fc}" '
            f'stroke="{ec}" stroke-width="1.7" filter="url(#soft)"/>'
            f'<text x="{x+w/2}" y="{y+h/2+5}" text-anchor="middle" font-size="{fs}" '
            f'{fw} fill="{tx}">{text}</text>')


def meter(cx, fill, hold_h, anim, value, ok):
    """Vertical RMS meter with a track, an animated fill and a value label."""
    s = []
    # track
    s.append(f'<rect x="{cx-16}" y="{RMS_BASE-RMS_MAX}" width="32" height="{RMS_MAX}" '
             f'rx="4" fill="#efe7d4" stroke="#e0d4b6" stroke-width="1"/>')
    # animated fill (grows up from baseline)
    s.append(f'<rect x="{cx-16}" width="32" rx="4" fill="{fill}">{anim}</rect>')
    s.append(f'<text x="{cx}" y="{RMS_BASE-RMS_MAX-10}" text-anchor="middle" '
             f'font-size="11.5" font-weight="700" fill="{INK}">action RMS</text>')
    mark = "✓" if ok else "✗"
    mc = TERRA_EC if ok else NOISE
    s.append(f'<text x="{cx}" y="{RMS_BASE+20}" text-anchor="middle" font-size="12.5" '
             f'font-weight="700" fill="{mc}">{mark} {value}</text>')
    return "\n".join(s)


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}" role="img" aria-label="Additive vs AdaLN action injection">'
]

# ── defs ───────────────────────────────────────────────────────────────────
parts.append('<defs>')
parts.append('<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
             '<stop offset="0" stop-color="#fdf7e9"/><stop offset="1" stop-color="#f6ecd2"/>'
             '</linearGradient>')
parts.append('<filter id="soft" x="-25%" y="-25%" width="150%" height="150%">'
             '<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#2c2722" '
             'flood-opacity="0.12"/></filter>')
for col in (TEAL_EC, TERRA_EC, PLUM_EC, DEAD, MUTE):
    parts.append(f'<marker id="ah-{col[1:]}" markerWidth="9" markerHeight="9" refX="7" '
                 f'refY="3" orient="auto" markerUnits="userSpaceOnUse">'
                 f'<path d="M0,0 L7.5,3 L0,6 Z" fill="{col}"/></marker>')
parts.append('</defs>')

# ── card + title ───────────────────────────────────────────────────────────
parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e7dcc0" stroke-width="1.5"/>')
parts.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="19" '
             f'font-weight="700" fill="{INK}">Action conditioning · Additive vs AdaLN injection</text>')
parts.append(f'<line x1="580" y1="64" x2="580" y2="446" stroke="#e7dcc0" stroke-width="1.3"/>')

# ════════════════════════════ LEFT — additive ═════════════════════════════
parts.append(f'<text x="56" y="78" font-size="15" font-weight="700" fill="{INK}">'
             f'Additive injection</text>')
parts.append(f'<text x="56" y="97" font-size="12" font-style="italic" fill="{MUTE}">'
             f'the action is a residual the model can zero out</text>')
parts.append(pill(300, 116, 168, 'x&#8242; = x + W&#183;a', '#fbf3e2', '#e7dcc0', INK))

# stream x  --(+)-->  x'
parts.append(f'<text x="74" y="196" font-size="12" font-style="italic" fill="{TEAL_TX}">hidden state x</text>')
parts.append(arrow("M70,212 L286,212", TEAL_EC, 3))
parts.append(arrow("M316,212 L486,212", TEAL_EC, 3))
parts.append(opcirc(300, 212, '+', NOISE))
parts.append(f'<text x="498" y="217" font-size="15" font-style="italic" fill="{TEAL_TX}">x&#8242;</text>')

# action a  --xW-->  (+)   [the W-gain path fades to ~0]
parts.append(labelbox(108, 300, 116, 46, 'action  a', TERRA_FC, TERRA_EC, TERRA_TX, fs=14))
fade = ('<animate attributeName="opacity" values="0.95;0.12;0.12;0.95" '
        'keyTimes="0;0.42;0.9;1" dur="4.5s" repeatCount="indefinite"/>')
parts.append(f'<g>{fade}'
             + arrow("M196,300 C220,262 250,235 292,224", TERRA_EC, 2.4)
             + f'<rect x="232" y="246" width="44" height="26" rx="7" fill="#fffdf6" '
               f'stroke="{TERRA_EC}" stroke-width="1.6"/>'
               f'<text x="254" y="263" text-anchor="middle" font-size="13" '
               f'font-weight="700" fill="{TERRA_EC}">&#215;W</text>'
             + '</g>')
parts.append(f'<text x="150" y="372" font-size="11" font-style="italic" fill="{MUTE}">'
             f'model learns W &#8594; 0</text>')

# RMS meter (collapses)
rms_l = ('<animate attributeName="height" values="106;2;2;106" keyTimes="0;0.42;0.9;1" '
         'dur="4.5s" repeatCount="indefinite"/>'
         '<animate attributeName="y" values="%d;%d;%d;%d" keyTimes="0;0.42;0.9;1" '
         'dur="4.5s" repeatCount="indefinite"/>' % (RMS_BASE-106, RMS_BASE-2, RMS_BASE-2, RMS_BASE-106))
parts.append(meter(516, NOISE, 106, rms_l, '0.0028', ok=False))
parts.append(f'<text x="300" y="476" text-anchor="middle" font-size="12.5" fill="{INK}">'
             f'gradient path through the action closes &#8212; it atrophies</text>')

# ════════════════════════════ RIGHT — adaln ═══════════════════════════════
parts.append(f'<text x="616" y="78" font-size="15" font-weight="700" fill="{INK}">'
             f'AdaLN injection</text>')
parts.append(f'<text x="616" y="97" font-size="12" font-style="italic" fill="{MUTE}">'
             f'the action scales every LayerNorm, so it cannot be zeroed</text>')
parts.append(pill(860, 116, 250, 'x&#8242; = &#947;(a)&#183;LN(x) + &#946;(a)', '#fbf3e2', '#e7dcc0', INK))

# stream:  x -> [LN] -> (x)gamma -> (+)beta -> x'
parts.append(f'<text x="634" y="196" font-size="12" font-style="italic" fill="{TEAL_TX}">x</text>')
parts.append(arrow("M634,212 L676,212", TEAL_EC, 3))
parts.append(labelbox(678, 196, 48, 32, 'LN', TEAL_FC, TEAL_EC, TEAL_TX, fs=13))
parts.append(arrow("M726,212 L786,212", TEAL_EC, 3))
parts.append(opcirc(802, 212, '&#215;', PLUM_EC))
parts.append(arrow("M818,212 L876,212", TEAL_EC, 3))
parts.append(opcirc(892, 212, '+', PLUM_EC))
parts.append(arrow("M908,212 L1042,212", TEAL_EC, 3))
parts.append(f'<text x="1050" y="217" font-size="15" font-style="italic" fill="{TEAL_TX}">x&#8242;</text>')

# action a -> gamma(a), beta(a)
parts.append(labelbox(648, 300, 116, 46, 'action  a', TERRA_FC, TERRA_EC, TERRA_TX, fs=14))
parts.append(arrow("M736,300 C760,266 778,244 800,230", TERRA_EC, 2.4))
parts.append(arrow("M752,300 C800,262 856,244 890,230", TERRA_EC, 2.4))
parts.append(f'<text x="766" y="268" font-size="13" font-style="italic" fill="{TERRA_EC}">&#947;(a)</text>')
parts.append(f'<text x="858" y="262" font-size="13" font-style="italic" fill="{TERRA_EC}">&#946;(a)</text>')

# feature map gated by gamma (bars pulse together, multiplicatively)
fm_x0, fm_n, fm_step, fm_base = 928, 10, 11.5, 360
parts.append(f'<text x="{fm_x0 + fm_n*fm_step/2}" y="378" text-anchor="middle" '
             f'font-size="11" font-style="italic" fill="{PLUM_TX}">feature map gated by &#947;(a)</text>')
for i in range(fm_n):
    bx = fm_x0 + i * fm_step
    base_h = 14 + (i % 4) * 7          # varied static profile
    lo, hi = base_h * 0.5, base_h * 1.25
    delay = -(i % 5) * 0.25
    anim = (f'<animate attributeName="height" values="{lo:.1f};{hi:.1f};{lo:.1f}" '
            f'dur="2.6s" begin="{delay}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="0;0.5;1" keySplines="0.4 0 0.6 1;0.4 0 0.6 1"/>'
            f'<animate attributeName="y" values="{fm_base-lo:.1f};{fm_base-hi:.1f};{fm_base-lo:.1f}" '
            f'dur="2.6s" begin="{delay}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="0;0.5;1" keySplines="0.4 0 0.6 1;0.4 0 0.6 1"/>')
    parts.append(f'<rect x="{bx}" width="7" rx="1.5" fill="{PLUM_EC}" opacity="0.85">{anim}</rect>')

# RMS meter (holds, gentle pulse)
rms_r = ('<animate attributeName="height" values="100;108;100" dur="2.6s" '
         'repeatCount="indefinite" calcMode="spline" keyTimes="0;0.5;1" '
         'keySplines="0.4 0 0.6 1;0.4 0 0.6 1"/>'
         '<animate attributeName="y" values="%d;%d;%d" dur="2.6s" repeatCount="indefinite" '
         'calcMode="spline" keyTimes="0;0.5;1" keySplines="0.4 0 0.6 1;0.4 0 0.6 1"/>'
         % (RMS_BASE-100, RMS_BASE-108, RMS_BASE-100))
parts.append(meter(1086, TERRA_EC, 104, rms_r, '0.20', ok=True))
parts.append(f'<text x="860" y="476" text-anchor="middle" font-size="12.5" fill="{INK}">'
             f'action gates the whole map &#8212; its influence persists</text>')

parts.append('</svg>')

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'injection_comparison.svg')
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
