# Draft editing strategy

## Source of truth

`writeup/website_draft.md` is the only editing surface. `docs/index.html` is generated from it by `python3 scripts/build_site.py` and is never edited directly. Always rebuild and commit both files after any markdown change.

## Section review order

Sections are reviewed and cleaned one at a time, in document order:

1. Background — done
2. Problem Statement — done
   - Latent space (subsection ### ) — done
3. Robot Hardware — done
4. Data — done
5. The World Model — done
6. Road to Planning — done
   - Planning: MPC + CEM in latent space — done
   - Inference setup — done
   - Run 001 — done
   - Does translation exist in the latent? — done
   - Run 002 — done
   - First closed-loop run — done (TODO: wire in goal image, start position, video from goals/run1/)
   - The semantic pivot — done
   - Building a waypoint graph — done
   - Three notes on building a graph — done
7. Reflection (§6 Limitations + §7 What comes next merged) — done

## Current asset inventory (docs/assets/)

### Videos
- `plan-demo-6s.mp4` — 6 s graph-guided MPC run; hero slot + §5 FIGURE_PAIR left
- `topdown_graph_hamper-6s.mp4` — 6 s overhead hamper run, 152× speedup; §5 FIGURE_PAIR right
- `dinov2_planner_demo.mp4` — flat planner demo with controls (§5, before "Building a waypoint graph")
- `nograph_nearfan.mp4` — no-graph demo near the fan, controls (§5)
- `nograph_nearchair.mp4` — no-graph demo near the chair, controls (§5)
- `long_0_cmp.mp4` — world model rollout vs real, autoplay (§4 World Model)
- `rotation_0_cmp.mp4` + `translation_0_cmp.mp4` — motion tracking side-by-side (§5)
- `subgoal-graph-anim.mp4` — graph build/route animation, wide + controls (§5)

### Images
- `lekiwi-mount.jpg` — robot photo (§2)
- `world_trajectories.png` — dead-reckoned episode paths (§3)
- `chunk_deltas.png` — action distribution (§3)
- `stationary_latent_compare_f05.png` — f=5 latent overlap (§5)
- `stationary_latent_compare.png` — f=10 latent separation (§5)
- `action_diagnostic.png` — action test passed (§5)
- `sweep_diagram.png` — measurement grid (§5)
- `metric_comparison.png` — VAE vs DINOv2 metric (§5)
- `fsweep_chunk_distributions.png` — why VAE latent fails (§5)
- `c1_smoke_strip.png` — decoder visualization (§5)
- `route_montage.png` — route film strip, wide (§5)

### Other
- `viewer_blueprint.rbl` — Rerun layout file for .rrd playback

## Outstanding asset placeholders

- `🆕 assets/ps5-controller.jpg` — photo of DualSense controller (§3)
- `goals/run1/goal.png` — wire into First closed-loop run section (§5)
- `⏳ assets/subgoal-graph-anim.mp4` — copy from context/figures/ into docs/assets/ if not already present

## Figures needing improvement

- `assets/nanowm_arch.png` — generated with matplotlib; needs polish. Consider cleaner layout, better arrow routing, and making the AdaLN mechanism more visually prominent. Possibly recreate in Figma or Excalidraw for a publication-quality look.
- `assets/diffusion_forcing.png` — generated with matplotlib; needs polish. The two-panel comparison works but the noise bar visualization could be clearer. Consider animating the autoregressive rollout as a GIF to better convey the inference-time loop.

## Open content questions

- Whether to mention LeCun's JEPA / latent-space prediction philosophy in Background (user raised, decision pending). Suggested placement: after the Fei-Fei Li taxonomy paragraph, framing the DINOv2 pivot as the practical version of "predict in abstract space, not pixels."
- OOD generalization: worth testing and noting in §7 what actually happens outside the training distribution.

## Build pipeline notes

- `build_site.py` supports two figure marker types:
  - `[FIGURE: ✅ assets/foo.mp4 controls — desc]` — single figure; add `wide` for full-bleed
  - `[FIGURE_PAIR: ✅ assets/a.mp4 | assets/b.mp4 — desc]` — two videos stacked full-width
- Video cache-busting: rename the asset file if GitHub Pages serves a stale cached version under the old name.

## Prose style rules

- No em dashes. Rewrite as a separate sentence or use a comma with a conjunction.
- No colons to introduce a clause or list inline. Break into a new sentence instead, or fold the list items into prose.
- No semicolons. Split into two sentences.
- Sentences should connect and flow like a blog post, not read as a series of short punches. Use relative clauses, participial phrases, and conjunctions to keep paragraphs moving.
- Scope constraints should be framed as deliberate choices, not failures.
- First-person voice ("My goal…") is appropriate in §7 Reflection.

## Structure rules

- Each section should state its goal or motivation before describing specifics or mechanics.
- Avoid embedding lists mid-sentence with a colon. Either convert to prose or use a proper bulleted list with a full-sentence introduction.
- Placeholder figures use `[FIGURE: ✅/⏳/🆕 assets/filename — description]` followed by a caption in italics.

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
