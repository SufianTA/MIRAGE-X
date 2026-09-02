# MIRAGE-X

**Select-or-abstain verification for imagined robot futures**

MIRAGE-X is a preliminary research prototype for a simple controller question: **when a video world model generates several plausible robot futures, how should an embodied agent decide whether any of them are trustworthy enough to execute?**

The prototype separates candidate generation from permission to act:

`OBSERVE -> IMAGINE -> RANK -> VERIFY -> SELECT / ABSTAIN`

## Pilot experiment

A fixed observation and instruction were evaluated with the open TesserAct RGB world model backend:

> Move the cup near bottle Franka Emika Panda.

Eight stochastic futures were generated using seeds 11, 22, 33, 44, 55, 66, 77, and 88. Each rollout contains 49 frames at 640 x 480. Mean generation time was **139.0 seconds per future** on one NVIDIA H100 PCIe, with **35.7 GB** peak allocated VRAM.

Two evaluation stages were compared.

### General plausibility ranking

Qwen2.5-VL scored task progress, object continuity, physical plausibility, manipulation feasibility, and temporal consistency. Candidate scores ranged from 5/10 to 6/10, with seed 33 ranked first. The rationales showed that aggregate plausibility scores could still coexist with important failure indicators.

### Task-grounded verification

A stricter verifier required visible, causal completion of the requested manipulation. Major failures included no meaningful task progress, wrong-object interaction, impossible grasp, severe collision/interpenetration, or clearly incomplete manipulation.

Seven ordered frames were sampled from each rollout and judged three deterministic times with Qwen2.5-VL-7B-Instruct. **All 24 judgments rejected the evaluated candidate.**

The resulting controller decision was therefore:

> **ABSTAIN - no generated future cleared task-grounded verification.**

The result illustrates the distinction between **visual plausibility** and **task-grounded actionability**. A controller should not be forced to execute the least-bad rollout simply because a ranking function can order the candidates.

## Repository contents

- `paper/MIRAGE-X_two_page_note.md` - concise research note
- `results/generation_manifest.json` - generation configuration, runtime, and memory
- `results/summary.json` - compact experiment summary
- `results/plausibility_critic.json` - first-stage plausibility evaluation
- `results/task_grounded_verifier.json` - strict three-pass verification output
- `src/generate_futures.py` - controlled multi-future generation
- `src/plausibility_critic_qwen.py` - general multimodal critic
- `src/task_grounded_verifier_qwen.py` - failure-aware task-grounded verifier
- `REPRODUCIBILITY.md` - environment and run details

Generated videos and figures are experiment artifacts and can be reproduced from the generation script and manifest. Upstream model weights are not redistributed.

## Next experiment

The natural follow-up is to replace the TesserAct backend with a geometry-enhanced robot world model such as GEM-4D while keeping the select-or-abstain controller fixed. A larger evaluation should compare:

1. single-rollout execution,
2. best-of-K ranking, and
3. select-or-abstain verification.

Useful metrics include task success, false-accept rate, abstention rate, world-model calls, latency, and recovery after prediction error during partial execution.

## Scope

This is a single-scene proof of concept. It does not include physical robot execution, inverse dynamics, or GEM-4D integration, and it does not establish a general test-time scaling law. Qwen2.5-VL is an imperfect verifier and should ultimately be calibrated against simulator or real-robot outcomes.

## Related work

- Zhou et al., *GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation* (ECCV 2026), arXiv:2605.22882.
- Gu et al., *GeoWorld-VLM: Geometry from World Models for Vision-Language Models* (2026), arXiv:2605.16713.
- Zhen et al., *TesserAct: Learning 4D Embodied World Models* (ICCV 2025).

## Author

Sufian Aldogom  
Preliminary research prototype, September 2026.
