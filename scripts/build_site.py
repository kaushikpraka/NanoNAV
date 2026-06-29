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
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "writeup", "website_draft.md")
OUT = os.path.join(ROOT, "docs", "index.html")
ASSETS_DIR = os.path.join(ROOT, "docs", "assets")

MEDIA_RE = re.compile(r"[\w./-]+\.(?:png|jpe?g|gif|svg|mp4|webm|glb|usdz)", re.I)
_pair_counter = [0]


def asset_src(base):
    """assets/<base> with a short content-hash ?v= so updated files never get
    served stale from a browser or GitHub Pages CDN cache."""
    path = os.path.join(ASSETS_DIR, base)
    try:
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()[:8]
        return "assets/%s?v=%s" % (base, h)
    except OSError:
        return "assets/" + base


def slugify(s):
    """Stable #anchor id from a section title."""
    s = re.sub(r"<[^>]+>", "", s).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def strip_md(s):
    """Strip basic markdown syntax to get plain text for slugification."""
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NanoNAV: real-robot navigation with Nano World Models</title>
<meta name="description" content="Latent-space planning with a Nano World Model drives a LeKiwi robot to goal images, learned from 25 minutes of driving, no maps, no depth, no pose.">
<link rel="canonical" href="https://kaushikpraka.github.io/NanoNAV/">
<meta property="og:type" content="article">
<meta property="og:url" content="https://kaushikpraka.github.io/NanoNAV/">
<meta property="og:title" content="NanoNAV: real-robot navigation with Nano World Models">
<meta property="og:description" content="Latent-space planning with a Nano World Model drives a LeKiwi robot to goal images, learned from 25 minutes of driving, no maps, no depth, no pose.">
<meta property="og:image" content="https://kaushikpraka.github.io/NanoNAV/assets/og_card.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="NanoNAV: real-robot navigation with Nano World Models">
<meta name="twitter:description" content="Latent-space planning with a Nano World Model drives a LeKiwi robot to goal images, learned from 25 minutes of driving, no maps, no depth, no pose.">
<meta name="twitter:image" content="https://kaushikpraka.github.io/NanoNAV/assets/og_card.jpg">
<link rel="stylesheet" href="style.css?v=%s">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
</head>
<body>
"""

SCRIPT = """<script>
// Sticky left-rail TOC with scroll-spy and subsection expand.
// Clones the inline Contents nav; CSS shows the rail only on wide screens.
// Active section gets .section-active (expands its subsection list);
// active subsection gets .active on its <a>.
(function () {
  var inlineNav = document.querySelector('nav.toc');
  if (!inlineNav) return;

  var rail = inlineNav.cloneNode(true);
  rail.className = 'toc-rail';
  rail.removeAttribute('id');
  document.body.appendChild(rail);

  // Build target lists from the cloned rail
  var sectionTargets = [], subTargets = [];
  Array.prototype.forEach.call(rail.querySelectorAll('ol > li'), function (li) {
    var a = li.querySelector(':scope > a');
    if (!a) return;
    var el = document.getElementById((a.getAttribute('href') || '').replace(/^#/, ''));
    if (!el) return;
    sectionTargets.push({ el: el, link: a, li: li });
    Array.prototype.forEach.call(li.querySelectorAll('.subsections a'), function (sa) {
      var sel = document.getElementById((sa.getAttribute('href') || '').replace(/^#/, ''));
      if (sel) subTargets.push({ el: sel, link: sa, parentLi: li });
    });
  });
  if (!sectionTargets.length) return;

  var curSection = null, curSub = null;

  function lastAbove(arr, th) {
    var active = arr[0];
    for (var i = 0; i < arr.length; i++) {
      if (arr[i].el.getBoundingClientRect().top <= th) active = arr[i];
      else break;
    }
    return active;
  }

  function update() {
    var th = 140;
    var activeSection = lastAbove(sectionTargets, th);

    // Active subsection: only within the current section, only if scrolled past
    var sectionSubs = subTargets.filter(function (s) { return s.parentLi === activeSection.li; });
    var activeSub = null;
    if (sectionSubs.length) {
      var candidate = lastAbove(sectionSubs, th);
      if (candidate.el.getBoundingClientRect().top <= th) activeSub = candidate;
    }

    if (activeSection !== curSection) {
      if (curSection) {
        curSection.link.classList.remove('active');
        curSection.li.classList.remove('section-active');
      }
      activeSection.link.classList.add('active');
      activeSection.li.classList.add('section-active');
      curSection = activeSection;
      if (curSub) { curSub.link.classList.remove('active'); curSub = null; }
    }

    if (activeSub !== curSub) {
      if (curSub) curSub.link.classList.remove('active');
      if (activeSub) activeSub.link.classList.add('active');
      curSub = activeSub;
    }
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
        src = asset_src(base)
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


def render_model(marker, caption):
    """Render [MODEL: assets/foo.glb assets/foo.usdz — desc] as a <model-viewer>."""
    paths = MEDIA_RE.findall(marker)
    glb  = next((p for p in paths if p.lower().endswith(".glb")),  None)
    usdz = next((p for p in paths if p.lower().endswith(".usdz")), None)
    if not glb:
        return '<div class="pending"><b>Pending 3-D model.</b> %s</div>' % inline(marker)
    glb_base  = os.path.basename(glb)
    glb_src   = "assets/" + glb_base
    usdz_attr = ('ios-src="assets/%s"' % os.path.basename(usdz)) if usdz else ""
    cap = caption_html(caption) if caption else ""
    return (
        '<figure>\n'
        '  <model-viewer src="{glb}" {usdz} alt="3-D model" '
        'camera-controls auto-rotate shadow-intensity="1" '
        'style="width:100%;height:480px;background:#f5f5f0;">'
        '</model-viewer>\n'
        '  {cap}\n'
        '</figure>'
    ).format(glb=glb_src, usdz=usdz_attr, cap=cap)


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
            h3_raw = first[4:].strip()
            out.append('<h3 id="%s">%s</h3>' % (slugify(strip_md(h3_raw)), inline(h3_raw)))
        elif first.startswith("[MODEL"):
            caption = None
            if len(lines) > 1 and lines[1].strip().startswith("*"):
                caption = lines[1].strip()
            elif i + 1 < len(blocks):
                nb = blocks[i + 1].strip()
                if nb.startswith("*") and nb.endswith("*") and "\n" not in nb:
                    caption = nb
                    i += 1
            out.append(render_model(first, caption))
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
    numbered = []     # (title, body)

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
    toc = ['<nav class="toc">', '  <span class="label">Contents</span>', '  <ol>']
    for ttl, body in numbered:
        subsecs = re.findall(r'^### (.+)$', body, re.M)
        li_open = '    <li><a href="#%s">%s</a>' % (slugify(ttl), ttl)
        if subsecs:
            sub_items = "\n".join(
                '          <li><a href="#%s">%s</a></li>' % (slugify(strip_md(s).strip()), inline(s))
                for s in subsecs
            )
            toc.append(
                li_open +
                '\n      <ul class="subsections">\n' +
                sub_items +
                '\n      </ul>\n    </li>'
            )
        else:
            toc.append(li_open + '</li>')
    toc.append("  </ol>")
    toc.append("</nav>")
    toc_html = "\n".join(toc)

    # article
    art = ["<article>", "", tldr_html, "", toc_html, ""]
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
