# MIRAGE-X: Knowing When Not to Act
### Task-grounded verification of imagined futures for embodied agents
**Sufian Aldogom | Preliminary research note | September 2026**

## Motivation
Video world models can produce futures that look coherent without necessarily providing a trustworthy basis for action. Recent work such as GEM-4D addresses this gap by improving geometric consistency and extracting executable robot trajectories. This note studies a complementary decision problem: what should an agent do when it has several imagined futures, but none is clearly task-grounded enough to execute?

## Hypothesis
An embodied agent should treat world-model rollouts as hypotheses to verify rather than plans to execute. Ranking candidates is useful, but ranking alone is insufficient: the controller needs an explicit select-or-abstain decision.

`OBSERVE -> IMAGINE -> RANK -> VERIFY -> DECIDE`

## Experimental setup
I used the open-source TesserAct RGB world model as a temporary backend because GEM-4D inference code was not public at the time of the experiment. The observation and instruction were fixed: “Move the cup near bottle Franka Emika Panda.” Eight futures were generated using seeds 11, 22, 33, 44, 55, 66, 77, and 88. Each rollout contained 49 frames at 640 x 480, using 50 inference steps and guidance scale 7.5.

- Futures: 8
- Mean generation: 139.0 s / future
- Peak VRAM: 35.7 GB
- Strict judgments: 24 (8 x 3)

## Two evaluators, two different conclusions
**General plausibility critic.** Qwen2.5-VL scored task progress, object continuity, physical plausibility, manipulation feasibility, and temporal consistency. It produced non-identical scores (5/10 or 6/10) and ranked seed 33 first. The critic therefore separated candidates, but its own rationales revealed a weakness: a relatively high score could coexist with warnings such as “no movement” or “unrealistic grasp.”

**Task-grounded verifier.** I tightened the rubric so visible, causal completion of the requested manipulation dominated the score. A major failure was triggered by no meaningful progress, wrong-object interaction, impossible grasp, severe collision/interpenetration, or clearly incomplete manipulation. Seven ordered frames were sampled from each rollout and judged three deterministic times. All 24 judgments rejected the candidate being evaluated.

**The important result is not that one candidate won. It is that the correct controller decision was ABSTAIN.**

## Interpretation
This pilot exposes a distinction between visual plausibility and task-grounded actionability. A ranking function can order futures even when none should be executed. For an embodied system, “best among the samples” is not the same as “safe and relevant enough to act on.”

## MIRAGE-X control rule
The proposed controller separates candidate generation from permission to act. It first samples futures, optionally ranks them, then applies a task-grounded gate. If no candidate clears the gate, the system should abstain and spend its next unit of computation on information acquisition or replanning rather than execution.

| Condition | Controller response |
|---|---|
| At least one future passes verification | Select the highest-quality passing future; execute only a short horizon. |
| Futures disagree / verifier uncertain | Allocate more imagination or acquire another observation. |
| No future passes verification | ABSTAIN; change action proposal, re-observe, or invoke a stronger world model. |
| Reality diverges after partial execution | Stop and re-plan from the new observation. |

## Why GEM-4D is the natural next test
GEM-4D is designed to make video-world-model rollouts geometrically consistent enough to support robot manipulation, and reports substantially higher manipulation success than TesserAct on its evaluated tasks. MIRAGE-X asks a different but complementary question: given a stronger rollout generator, can an external verifier calibrate when the resulting future is trustworthy enough to execute? A useful next study would replace the TesserAct backend with GEM-4D while keeping the select-or-abstain layer fixed.

A controlled follow-up would measure: (1) verification pass rate, (2) calibration of verifier confidence against actual task success, (3) the relationship between geometric consistency and verifier acceptance, (4) compute spent before action, and (5) recovery after prediction error during partial execution.

## What this pilot establishes — and what it does not
**Established:** a reproducible pipeline generated eight stochastic robot futures from one fixed scene; a generic multimodal critic ranked them; a stricter task-grounded verifier rejected all eight across three deterministic judgments per future; and the resulting control decision was abstention.

**Not established:** a general scaling law, improved robot success, superiority over a baseline controller, or any result about GEM-4D itself. There was no physical execution or inverse-dynamics evaluation. Qwen2.5-VL is also an imperfect verifier and should ultimately be calibrated against simulator or real-robot outcomes.

## Research direction
The broader question is whether embodied agents can learn when imagination is actionable. A useful world model should not only support prediction; its downstream controller should know when to trust a predicted future, when to seek more evidence, and when not to act.

## References
1. H. Zhen et al. “TesserAct: Learning 4D Embodied World Models.” ICCV 2025.
2. K. Zhou et al. “GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation.” ECCV 2026; arXiv:2605.22882.
3. R. Gu, K. Zhou, Y. Luo, M. Wang. “GeoWorld-VLM: Geometry from World Models for Vision-Language Models.” arXiv:2605.16713, 2026.
