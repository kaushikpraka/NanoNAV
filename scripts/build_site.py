#!/usr/bin/env python3
"""Build docs/index.html from writeup/website_draft.md.

The markdown is the SINGLE SOURCE OF TRUTH for the write-up's prose. Run this
after editing the markdown to regenerate the deployed page:

    python scripts/build_site.py

Conventions understood in the markdown:
  # Title …                       -> masthead <h1> (markdown link allowed)
  **Subtitle:** …                 -> masthead subtitle
  **Byline:** …                   -> masthead byline
  ## Hero video                   -> hero slot (uses the [FIGURE:] under it)
  ## TL;DR                        -> the TL;DR callout box
  ## Background — …               -> <h2 id="background">
  ## N · Title                    -> <h2 id="<slug>">  (slug from SLUGS below)
  [FIGURE: … assets/x.png …]      -> <figure> (img/video); next *italic* line is the caption
      + "wide"                    -> figure.wide (full-bleed)
      + two media paths / "side by side" -> .row2 two-up
      missing asset               -> dashed "pending" placeholder
  | a | b |  (markdown table)     -> <table>
  > note text                     -> <p class="note"> callout
  *Lesson: …*  (standalone)       -> <p class="note"> callout (italic)
  **"Question?"** answer …        -> <aside class="fieldnote">
  1. / - lists                    -> <ol>/<ul>
  [TODO: …]                       -> stripped from output

Figures/tables/asides are maintained in the markdown alongside the prose; only
the page scaffolding (head, hero shell, sticky-TOC script, footer) lives here.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "writeup", "website_draft.md")
OUT = os.path.join(ROOT, "docs", "index.html")
ASSETS_DIR = os.path.join(ROOT, "docs", "assets")

MEDIA_RE = re.compile(r"[\w./-]+\.(?:png|jpe?g|gif|svg|mp4|webm)", re.I)
_pair_counter = [0]


def slugify(s):
    """Stable #anchor id from a section title."""
    s = re.sub(r"<[^>]+>", "", s).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NanoNAV — real-robot navigation with Nano World Models</title>
<meta name="description" content="Latent-space planning with a Nano World Model drives a LeKiwi robot to goal images — learned from 25 minutes of driving, no maps, no depth, no pose.">
<link rel="stylesheet" href="style.css?v=%s">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
</head>
<body>
"""

SCRIPT = """<script>
// Sticky left-rail TOC with scroll-spy. Clones the inline Contents nav so there's
// a single source of truth; CSS shows the rail only on wide screens.
(function () {
  var inline = document.querySelector('nav.toc');
  if (!inline) return;

  var rail = inline.cloneNode(true);
  rail.className = 'toc-rail';
  rail.removeAttribute('id');
  document.body.appendChild(rail);

  var links = Array.prototype.slice.call(rail.querySelectorAll('a'));
  var targets = links.map(function (a) {
    var id = (a.getAttribute('href') || '').replace(/^#/, '');
    return { el: document.getElementById(id), link: a };
  }).filter(function (t) { return t.el; });
  if (!targets.length) return;

  var current = null;
  function update() {
    var threshold = 140; // px below the viewport top counts as "current"
    var active = targets[0];
    for (var i = 0; i < targets.length; i++) {
      if (targets[i].el.getBoundingClientRect().top <= threshold) active = targets[i];
      else break;
    }
    if (active === current) return;
    if (current) current.link.classList.remove('active');
    active.link.classList.add('active');
    current = active;
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () { update(); ticking = false; });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update();
})();

// Theme toggle — cycles cream → gray → dark, persists via localStorage
(function () {
  var themes = ['cream', 'gray', 'slate', 'dark'];
  var saved = localStorage.getItem('nanovnav-theme') || 'cream';

  function apply(theme) {
    document.body.classList.remove('theme-gray', 'theme-slate', 'theme-dark');
    if (theme === 'gray')  document.body.classList.add('theme-gray');
    if (theme === 'slate') document.body.classList.add('theme-slate');
    if (theme === 'dark')  document.body.classList.add('theme-dark');
    btn.textContent = theme;
    localStorage.setItem('nanovnav-theme', theme);
    saved = theme;
  }

  var btn = document.createElement('button');
  btn.id = 'theme-toggle';
  btn.setAttribute('aria-label', 'Toggle colour theme');
  btn.onclick = function () {
    apply(themes[(themes.indexOf(saved) + 1) % themes.length]);
  };
  document.body.appendChild(btn);
  apply(saved);
})();
</script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]})"></script>

</body>
</html>
"""


# ---------------------------------------------------------------- inline markup
def inline(s):
    """Convert markdown inline syntax to HTML. Order matters: protect code,
    then links, then bold/italic, then auto-link bare URLs."""
    s = re.sub(r"\[TODO:[^\]]*\]", "", s)            # strip author TODOs
    codes = []
    def stash(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)
    s = re.sub(r"`([^`]+)`", stash, s)               # protect `code`
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)  # [t](u)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    # auto-link bare URLs not already inside an href
    s = re.sub(r'(?<![">=])\bhttps?://[^\s<)]*[\w/]', lambda m: '<a href="%s">%s</a>' % (m.group(0), m.group(0)), s)
    for i, c in enumerate(codes):                    # restore code spans
        s = s.replace("\x00%d\x00" % i, "<code>%s</code>" % c)
    return s.strip()


def caption_html(cap):
    c = cap.strip()
    if c.startswith("*") and c.endswith("*"):
        c = c[1:-1].strip()
    c = re.sub(r"\[TODO:[^\]]*\]", "", c).strip()
    if not c:
        return ""
    h = inline(c)
    # bold the lead sentence, guarding common abbreviations
    guard = {"vs.": "vs\x01", "e.g.": "e\x01g\x01", "i.e.": "i\x01e\x01", "Dr.": "Dr\x01"}
    g = h
    for k, v in guard.items():
        g = g.replace(k, v)
    m = re.match(r"(.+?[.!?])(\s+)(.*)", g, re.S)
    if m:
        lead = m.group(1)
        rest = m.group(3)
        for k, v in guard.items():
            lead = lead.replace(v, k)
            rest = rest.replace(v, k)
        return "<figcaption><b>%s</b> %s</figcaption>" % (lead, rest)
    return "<figcaption>%s</figcaption>" % h


# -------------------------------------------------------------------- figures
def render_figure(marker, caption):
    paths = MEDIA_RE.findall(marker)
    # resolve each to assets/<basename> and check existence in docs/assets
    resolved = []
    for p in paths:
        base = os.path.basename(p)
        resolved.append((base, os.path.exists(os.path.join(ASSETS_DIR, base))))
    wide = "wide" in marker.lower()
    row2 = len(resolved) >= 2 or "side by side" in marker.lower()
    have = resolved and all(ok for _, ok in resolved)

    if not have:
        desc = ""
        if caption:
            c = re.sub(r"\[TODO:[^\]]*\]", "", caption.strip().strip("*")).strip()
            desc = inline(c) if c else ""
        if not desc:                              # caption empty/TODO-only → use the marker text
            d = re.sub(r"^\[FIGURE:\s*", "", marker).rstrip("]")
            d = re.sub(r"^[✅🆕⏳]\s*", "", d)
            d = re.sub(r"\[TODO:[^\]]*\]", "", d).strip()
            desc = inline(d)
        return '<div class="pending"><b>Pending figure.</b> %s</div>' % desc

    cls = ' class="wide"' if wide else ""
    cap = caption_html(caption) if caption else ""

    def media(base):
        src = "assets/" + base
        if base.lower().endswith((".mp4", ".webm")):
            attrs = "controls loop muted playsinline" if "controls" in marker.lower() \
                else "autoplay loop muted playsinline"
            return '<video src="%s" %s></video>' % (src, attrs)
        return '<img src="%s" alt="">' % src

    if row2:
        inner = '\n  <div class="row2">\n    %s\n    %s\n  </div>' % (
            media(resolved[0][0]), media(resolved[1][0]))
    else:
        inner = "\n  " + media(resolved[0][0])
    return "<figure%s>%s\n  %s\n</figure>" % (cls, inner, cap)


def render_figure_pair(marker, caption):
    """Render two synced side-by-side videos for [FIGURE_PAIR: a.mp4 | b.mp4 ...]."""
    paths = MEDIA_RE.findall(marker)
    if len(paths) < 2:
        return '<div class="pending"><b>Pending figure pair.</b> %s</div>' % inline(marker)
    bases = [os.path.basename(p) for p in paths[:2]]
    missing = [b for b in bases if not os.path.exists(os.path.join(ASSETS_DIR, b))]
    if missing:
        desc = re.sub(r"^\[FIGURE_PAIR:\s*", "", marker).rstrip("]")
        return '<div class="pending"><b>Pending figure pair.</b> %s</div>' % inline(desc)

    gid = _pair_counter[0]
    _pair_counter[0] += 1
    attrs = "controls loop muted playsinline"

    cap = caption_html(caption) if caption else ""
    videos = "\n    ".join(
        '<video id="vp-%d-%s" src="assets/%s" %s></video>' % (gid, side, b, attrs)
        for side, b in zip(("a", "b"), bases)
    )
    return (
        '<figure>\n'
        '  <div class="video-pair" data-sync-group="%d">\n'
        '    %s\n'
        '  </div>\n'
        '  %s\n'
        '</figure>' % (gid, videos, cap)
    )


# --------------------------------------------------------------------- tables
def split_cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render_table(block):
    header = split_cells(block[0])
    rows = [split_cells(r) for r in block[2:] if r.strip()]
    th = "".join("<th>%s</th>" % inline(c) for c in header)
    body = ""
    for row in rows:
        tds = "".join("<td>%s</td>" % inline(c) for c in row)
        body += "    <tr>%s</tr>\n" % tds
    return ("<table>\n  <thead><tr>%s</tr></thead>\n  <tbody>\n%s  </tbody>\n</table>"
            % (th, body))


def render_list(block, ordered):
    tag = "ol" if ordered else "ul"
    items = []
    for line in block:
        line = line.strip()
        line = re.sub(r"^(?:\d+\.|[-*])\s+", "", line)
        items.append("  <li>%s</li>" % inline(line))
    return "<%s>\n%s\n</%s>" % (tag, "\n".join(items), tag)


# -------------------------------------------------------------- body renderer
def render_body(text, is_last_section):
    text = re.sub(r"(?m)^\s*---+\s*$", "", text)   # drop separator rules
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        lines = b.split("\n")
        first = lines[0].strip()
        stripped = re.sub(r"\[TODO:[^\]]*\]", "", b).strip()
        last_block = is_last_section and i == len(blocks) - 1

        if not stripped and not first.startswith("[FIGURE"):
            i += 1
            continue
        if first.startswith("### "):
            out.append("<h3>%s</h3>" % inline(first[4:]))
        elif first.startswith("[FIGURE_PAIR"):
            caption = None
            if len(lines) > 1 and lines[1].strip().startswith("*"):
                caption = lines[1].strip()
            elif i + 1 < len(blocks):
                nb = blocks[i + 1].strip()
                if nb.startswith("*") and nb.endswith("*") and "\n" not in nb:
                    caption = nb
                    i += 1
            out.append(render_figure_pair(first, caption))
        elif first.startswith("[FIGURE"):
            caption = None
            if len(lines) > 1 and lines[1].strip().startswith("*"):
                caption = lines[1].strip()          # caption shares the figure block
            elif i + 1 < len(blocks):
                nb = blocks[i + 1].strip()
                if nb.startswith("*") and nb.endswith("*") and "\n" not in nb:
                    caption = nb
                    i += 1
            out.append(render_figure(first, caption))
        elif "|" in first and len(lines) >= 2 and set(lines[1].strip()) <= set("|-: "):
            out.append(render_table(lines))
        elif re.match(r"^\d+\.\s", first):
            out.append(render_list(lines, ordered=True))
        elif first.startswith("- "):
            out.append(render_list(lines, ordered=False))
        elif first.startswith(">"):
            content = " ".join(re.sub(r"^>\s?", "", l).strip() for l in lines)
            out.append('<p class="note">%s</p>' % inline(content))
        elif first.startswith('**"'):
            out.append('<aside class="fieldnote">\n  <p>%s</p>\n</aside>' % inline(b.replace("\n", " ")))
        elif first.startswith("*") and stripped.endswith("*"):
            if last_block:                       # closing acknowledgments line
                out.append("<hr>\n<p>%s</p>" % inline(b.replace("\n", " ")))
            else:
                out.append('<p class="note">%s</p>' % inline(b.replace("\n", " ")))
        elif first == "---":
            pass                                  # section separator
        else:
            out.append("<p>%s</p>" % inline(b.replace("\n", " ")))
        i += 1
    return "\n\n".join(out)


# --------------------------------------------------------------------- build
def build():
    raw = open(MD, encoding="utf-8").read()
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)   # drop HTML comments

    title = re.search(r"^#\s+(.+)$", raw, re.M).group(1).strip()
    subtitle = re.search(r"^\*\*Subtitle:\*\*\s*(.+)$", raw, re.M).group(1).strip()
    byline = re.search(r"^\*\*Byline:\*\*\s*(.+)$", raw, re.M).group(1).strip()

    # split into ## sections
    parts = re.split(r"^##\s+", raw, flags=re.M)[1:]
    sections = []
    for p in parts:
        head, _, body = p.partition("\n")
        sections.append((head.strip(), body.strip()))

    hero_src = "assets/plan-demo.mp4"
    tldr_html = ""
    numbered = []     # (n, title, slug, body)
    background = None  # (heading, body)

    for head, body in sections:
        if head.lower().startswith("hero"):
            m = MEDIA_RE.search(body)
            if m:
                hero_src = "assets/" + os.path.basename(m.group(0))
        elif head.lower().startswith("tl;dr") or head.lower().startswith("tldr"):
            body_c = re.sub(r"(?m)^\s*---+\s*$", "", body)
            paras = [inline(x.strip()) for x in re.split(r"\n\s*\n", body_c) if x.strip()]
            tldr_html = '<div class="tldr">\n  <span class="label">TL;DR</span>\n  ' + \
                "\n  <br><br>\n  ".join(paras) + "\n</div>"
        elif head.lower().startswith("background"):
            background = (head, body)
        else:
            # strip any literal "N · " — sections are auto-numbered by position,
            # so splitting/reordering never requires manual renumbering
            sec_title = re.sub(r"^\d+\s*·\s*", "", head).strip()
            numbered.append((sec_title, body))

    # masthead
    masthead = (
        '<header class="masthead">\n'
        '  <h1>%s</h1>\n'
        '  <p class="subtitle">%s</p>\n'
        '  <div class="byline">%s</div>\n'
        '</header>\n' % (inline(title), inline(subtitle), inline(byline))
    )
    hero = ('<div class="hero">\n  <div class="frame">\n'
            '    <video src="%s" autoplay loop muted playsinline controls></video>\n'
            '  </div>\n</div>\n' % hero_src)

    # table of contents
    toc = ['<nav class="toc">', '  <span class="label">Contents</span>']
    if background:
        toc.append('  <p class="toc-intro"><a href="#background">%s</a></p>' % background[0])
    toc.append("  <ol>")
    for ttl, _ in numbered:
        toc.append('    <li><a href="#%s">%s</a></li>' % (slugify(ttl), ttl))
    toc.append("  </ol>")
    toc.append("</nav>")
    toc_html = "\n".join(toc)

    # article
    art = ["<article>", "", tldr_html, "", toc_html, ""]
    if background:
        art.append('<h2 id="background">%s</h2>\n' % background[0])
        art.append(render_body(background[1], False))
        art.append("")
    for idx, (ttl, body) in enumerate(numbered):
        art.append('<h2 id="%s">%d · %s</h2>\n' % (slugify(ttl), idx + 1, ttl))
        art.append(render_body(body, is_last_section=(idx == len(numbered) - 1)))
        art.append("")
    art.append("</article>")
    article = "\n".join(art)

    footer = '<footer>\n  © 2026 Kaushik Prakash · written up June 2026\n</footer>\n'

    import time
    html = (HEAD % int(time.time())) + "\n" + masthead + "\n" + hero + "\n" + article + "\n\n" + footer + "\n" + SCRIPT
    open(OUT, "w", encoding="utf-8").write(html)
    print("wrote %s (%d sections)" % (os.path.relpath(OUT, ROOT), len(numbered)))


if __name__ == "__main__":
    build()
