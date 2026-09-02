# MIRAGE-X: Select-or-Abstain Verification of Imagined Robot Futures
### Task-grounded verification for embodied world-model agents
**Sufian Aldogom | Preliminary research note | September 2026**

## Motivation
Video world models can generate visually coherent robot futures without guaranteeing that those futures are task-correct or executable. GEM-4D addresses the generation side of this problem through geometry-enhanced world modeling for robot manipulation. MIRAGE-X studies the downstream decision problem: **when several futures have been imagined, should the controller execute one of them at all?**

## Hypothesis
World-model rollouts should be treated as candidate hypotheses, not executable plans by default. A controller should rank candidates when useful, verify them against the requested task, and explicitly **select or abstain**.

`OBSERVE -> IMAGINE -> RANK -> VERIFY -> DECIDE`

## Experimental setup
TesserAct was used as the open world-model backend for this pilot. The observation and instruction were held fixed: *Move the cup near bottle Franka Emika Panda.* Eight stochastic futures were generated using seeds 11, 22, 33, 44, 55, 66, 77, and 88. Each rollout contained 49 frames at 640 x 480, using 50 inference steps and guidance scale 7.5.

- Futures: 8
- Mean generation: 139.0 s / future
- Peak allocated VRAM: 35.7 GB
- Strict verifier judgments: 24 (8 futures x 3 passes)

## Evaluation
**General plausibility critic.** Qwen2.5-VL scored task progress, object continuity, physical plausibility, manipulation feasibility, and temporal consistency. Candidate scores ranged from 5/10 to 6/10, with seed 33 ranked first. The rationales showed that a relatively high aggregate score could still coexist with important failure indicators.

**Task-grounded verifier.** A stricter rubric required visible, causal completion of the requested manipulation. Major failures included no meaningful task progress, wrong-object interaction, impossible grasp, severe collision/interpenetration, or clearly incomplete manipulation. Seven ordered frames were sampled from each rollout and judged three deterministic times. **All 24 judgments rejected the evaluated candidate.**

**Result: no imagined future cleared the task-grounded gate. The controller decision is ABSTAIN.**

## Interpretation
The pilot separates **visual plausibility** from **task-grounded actionability**. A ranking function can order candidate futures even when none is acceptable for execution. In that case, selecting the highest-ranked rollout would be a controller error; abstention is the safer and more informative outcome.

## Controller rule
MIRAGE-X separates candidate generation from permission to act. The controller may allocate additional imagination when candidates disagree, but execution is allowed only after task-grounded verification.

| Condition | Controller response |
|---|---|
| One or more futures pass verification | Select the highest-quality passing future; execute a short horizon. |
| Candidates disagree or verification is uncertain | Generate additional futures or acquire another observation. |
| No future passes verification | Abstain; revise the action proposal, re-observe, or invoke a stronger world model. |
| Observed state diverges from predicted state | Stop execution, update state, regenerate candidates, and verify again. |

## Why GEM-4D is the natural next experiment
GEM-4D is designed to produce geometry-enhanced video-world-model rollouts and convert them into robot trajectories. MIRAGE-X asks a complementary downstream question: **given a stronger rollout generator, when should the controller trust a predicted future enough to execute it?** A clean follow-up is to substitute GEM-4D for the TesserAct backend while keeping the select-or-abstain layer fixed.

## Proposed evaluation
Evaluate multiple manipulation tasks and controlled disturbances under three controller variants: (1) single-rollout execution, (2) best-of-K ranking, and (3) select-or-abstain verification. Measure task success, false-accept rate, abstention rate, world-model calls, latency, and recovery after prediction error. A closed-loop version would execute only a short horizon, compare the observed next state against the predicted state, and trigger re-imagination when divergence exceeds a threshold.

## Current scope
This is a single-scene proof of concept. It does not include physical robot execution, inverse dynamics, or GEM-4D integration, and it does not establish a general test-time scaling law. Qwen2.5-VL is an imperfect verifier and should ultimately be calibrated against simulator or real-robot outcomes. The contribution of this pilot is the **select-or-abstain verification formulation** and a reproducible failure case that motivates a larger study.

## Research direction
The broader question is whether embodied agents can learn when imagination is actionable. A useful world model should support prediction; a useful controller should know when to trust that prediction, when to seek more evidence, and when not to act.

## References
1. K. Zhou et al. *GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation.* ECCV 2026; arXiv:2605.22882.
2. R. Gu, K. Zhou, Y. Luo, M. Wang. *GeoWorld-VLM: Geometry from World Models for Vision-Language Models.* arXiv:2605.16713, 2026.
3. H. Zhen et al. *TesserAct: Learning 4D Embodied World Models.* ICCV 2025.
