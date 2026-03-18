# Project Status

## Goal

Build a standalone SkillRL-style fake news detection project that can:

- normalize raw datasets into a common schema,
- run a multi-step verification environment,
- use a Qwen VL agent for multimodal reasoning,
- support future SFT and RL refinement.

## What Is Done

- Standalone package structure created under `fake-news-skillrl/`.
- Environment, prompting, parser, skill memory, rollout trainer, and evaluation scripts implemented.
- Qwen VL model path added with automatic CUDA usage when available.
- Repeated inspection now incurs an invalid-action penalty.
- Fakett raw dataset adapter added:
  - reads `samples.jsonl`,
  - links rows to `[video_id].mp4`,
  - optionally extracts frames when OpenCV is installed.
- Parser hardened to treat malformed verdicts as verdict failures instead of misclassifying embedded inspect tags.
- Real-dataset metadata bias reduced by changing Fakett `task_type` from `misleading_caption` to `unknown`.
- Frame sampling improved to prefer non-blank mid-video frames and sample more frames by default.

## Current Known Issues

- Qwen VL still shows a tendency to loop or overpredict `fake` on small zero-shot runs.
- Evidence scoring is still weak for Fakett because `gold_evidence` is only lightly derived from `event` and `description`.
- VL quality depends heavily on actual extracted frames being present and informative.
- The project still uses a lightweight rollout/training scaffold rather than full `verl` integration.

## Recommended Next Moves

1. Regenerate normalized Fakett data with improved frame sampling on the server.
2. Re-run Qwen VL evaluation on the regenerated normalized file.
3. Inspect traces for:
   - invalid verdict formatting,
   - repeated invalid loops,
   - overuse of metadata vs visual evidence.
4. If the model still struggles zero-shot:
   - generate SFT trajectories from the real normalized data,
   - fine-tune on the action protocol,
   - then re-run evaluation.
5. Consider integrating `verl` once the task formatting and reward behavior are stable.

## Notes For Future Work

- Keep this file updated after each major experiment.
- Prefer behavior changes that improve trace quality before scaling training complexity.
