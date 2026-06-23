# Draft editing strategy

## Source of truth

`writeup/website_draft.md` is the only editing surface. `docs/index.html` is generated from it by `python3 scripts/build_site.py` and is never edited directly. Always rebuild and commit both files after any markdown change.

## Section review order

Sections are reviewed and cleaned one at a time, in document order:

1. Background — done
2. Problem Statement — done
3. Robot Hardware — done
4. Data — done
5. The World Model — done
6. Road to a working planner — done
   - Planning: MPC + CEM in latent space — done
   - Inference setup — done
   - Run 001 — done
   - Does translation exist in the latent? — done
   - Run 002 — done
   - First closed-loop run — done (TODO: wire in goal image, start position, video from goals/run1/)
   - The semantic pivot — done
   - Building a waypoint graph — done
   - Three failures on the way to the first graph success — done
7. Limitations — done
8. What comes next — done

## Outstanding asset placeholders

- `goals/run1/goal.png` — wire into First closed-loop run section
- `⏳ assets/dinov2_planner_demo.mp4` — short clip of DINOv2 flat planner working on robot
- `⏳ assets/subgoal-graph-anim.mp4` — graph build/route animation
- `⏳ on-robot success capture` — nearpurifier run trace or screen recording
- `⏳ additional on-robot success run videos`
- `🆕 assets/ps5-controller.jpg` — photo of DualSense controller
- TL;DR — not formally reviewed against style rules

## Prose style rules

- No em dashes. Rewrite as a separate sentence or use a comma with a conjunction.
- No colons to introduce a clause or list inline. Break into a new sentence instead, or fold the list items into prose.
- No semicolons. Split into two sentences.
- Sentences should connect and flow like a blog post, not read as a series of short punches. Use relative clauses, participial phrases, and conjunctions to keep paragraphs moving.
- Comments about what is "not novel" or what the system "cannot do" should be framed as deliberate scope decisions, not self-criticism.

## Structure rules

- Each section should state its *goal or motivation* before describing specifics or mechanics. The reader should understand why before how.
- Avoid embedding lists mid-sentence with a colon. Either convert to prose or use a proper bulleted list with a full-sentence introduction.
- Placeholder figures use the convention `[FIGURE: ✅/⏳/🆕 assets/filename — description]` followed by a caption in italics.

## What to omit / defer

- The overfit discussion ("But won't 160M parameters on 50 episodes overfit?") was removed from §4 to be reintroduced later with more context.
- Any hallucination references were removed from the draft entirely.

## Rebuild and publish workflow

```bash
python3 scripts/build_site.py   # writes docs/index.html
git add writeup/website_draft.md docs/index.html
git commit -m "doc: ..."
git push
```

GitHub Pages serves from `docs/` on `main` at:
https://kaushiktheprogrammer.github.io/NanoNAV/
