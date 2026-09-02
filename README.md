# MIRAGE-X

**Task-grounded verification for imagined robot futures**

MIRAGE-X is a small, reproducible pilot around one question: **when a video world model imagines several plausible futures, how should an embodied agent decide whether any of them are trustworthy enough to execute?**

This is a single-scene research prototype, not a benchmark result and not an extension of GEM-4D. I used the open-source TesserAct world model as the backend because GEM-4D inference code was not public when the experiment was run.

## Experiment

The observation and instruction were held fixed:

> Move the cup near bottle Franka Emika Panda.

TesserAct generated eight stochastic futures using seeds 11, 22, 33, 44, 55, 66, 77, and 88. Each rollout contains 49 frames at 640 x 480. Mean generation time was **139.0 seconds per future** on one NVIDIA H100 PCIe, with **35.7 GB** peak allocated VRAM.

I then compared two evaluation strategies.

### 1. General plausibility ranking

Qwen2.5-VL scored task progress, object continuity, physical plausibility, manipulation feasibility, and temporal consistency. Scores varied between 5/10 and 6/10, with seed 33 ranked first. The critic separated candidates, but its own rationales exposed a weakness: a relatively high score could coexist with warnings such as `no movement` or `unrealistic grasp`.

### 2. Task-grounded verification

I tightened the rubric so that visible, causal completion of the requested manipulation dominates the score. Major failures include no meaningful progress, wrong-object interaction, impossible grasp, severe collision/interpenetration, and clearly incomplete manipulation.

The verifier sampled seven ordered frames from each rollout and ran three deterministic Qwen2.5-VL judgments per future. **All 24 judgments rejected the candidate being evaluated.**

The correct controller decision in this pilot is therefore:

> **ABSTAIN — no generated future cleared task-grounded verification.**

This is the main point of MIRAGE-X: a controller should not be forced to execute the least-bad rollout simply because a ranking function can order the candidates.

## Proposed control loop

`OBSERVE -> IMAGINE -> RANK -> VERIFY -> SELECT or ABSTAIN`

If no candidate clears verification, the next step should be to re-observe, change the candidate action set, invoke a stronger world model, or re-plan.

## Repository

- `paper/MIRAGE-X_two_page_note.md` — preliminary research note
- `results/generation_manifest.json` — generation settings, runtime, and memory
- `results/summary.json` — compact experiment summary
- `src/generate_futures.py` — controlled multi-future generation
- `src/plausibility_critic_qwen.py` — general multimodal critic
- `src/task_grounded_verifier_qwen.py` — failure-aware verifier
- `REPRODUCIBILITY.md` — environment and run details

The generated videos and rendered figures are retained as experiment artifacts; upstream model weights are not redistributed.

## Scope

This pilot does **not** include physical robot execution, inverse dynamics, GEM-4D integration, or evidence for a general test-time scaling law. Qwen2.5-VL is also an imperfect evaluator. The experiment is intended to isolate the distinction between **visual plausibility** and **task-grounded actionability**.

## Related work

- Zhen et al., *TesserAct: Learning 4D Embodied World Models* (ICCV 2025).
- Zhou et al., *GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation* (ECCV 2026), arXiv:2605.22882.
- Gu et al., *GeoWorld-VLM: Geometry from World Models for Vision-Language Models* (2026), arXiv:2605.16713.

## Author

Sufian Aldogom  
Preliminary research prototype, September 2026.
