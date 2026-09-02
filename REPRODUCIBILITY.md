# Reproducibility notes

Hardware: one NVIDIA H100 PCIe (80 GB).

Generation: 640 x 480, 50 inference steps, guidance scale 7.5, 8 FPS, seeds 11/22/33/44/55/66/77/88.

The strict verifier samples seven ordered frames from each rollout and runs three deterministic Qwen2.5-VL-7B-Instruct judge passes. The full rubric is embedded in `src/task_grounded_verifier_qwen.py`.

Upstream model weights are not redistributed.
