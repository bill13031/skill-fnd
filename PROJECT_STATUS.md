# Project Status

## Goal

Build a standalone SkillRL-style short-video credibility assessment project that can:

- normalize raw datasets into a common schema,
- run a multi-step verification environment,
- use a Qwen VL agent for multimodal reasoning,
- support future SFT and RL refinement.

## What Is Done

- Standalone package structure created under `fake-news-skillrl/`.
- Environment, prompting, parser, skill memory, rollout trainer, and evaluation scripts implemented.
- Qwen VL model path added with automatic CUDA usage when available.
- The environment now provides the full case package up front instead of making the agent inspect evidence item by item.
- The action protocol was redesigned around reasoning-control steps:
  - `<create>...</create>`
  - `<check>...</check>`
  - `<use_skill>...</use_skill>`
  - final `<verdict>...</verdict>`
- Fakett raw dataset adapter added:
  - reads `samples.jsonl`,
  - links rows to `[video_id].mp4`,
  - optionally extracts frames when OpenCV is installed.
- Task framing updated from generic fake-news detection to short-video credibility assessment:
  - misleading or non-factual content should be flagged,
  - playful exaggeration or metaphor can pass when it is not making a concrete misleading factual claim.
- Verdicts are now fake/real only with a short rationale, without an evidence list.
- Parser hardened to treat malformed verdicts as verdict failures instead of misclassifying embedded action-like text.
- Real-dataset metadata bias reduced by changing Fakett `task_type` from `misleading_caption` to `unknown`.
- Frame sampling improved to prefer non-blank mid-video frames and sample more frames by default.
- The Qwen VL message builder now tries to attach all available frame images directly to the prompt instead of gating them behind inspect actions.
- VL evaluation now defaults to attaching frames only on the first reasoning step and can force a fallback verdict after a small reasoning budget.
- The model-facing prompt now hides metadata, frame descriptions, and task-type hints to reduce shortcut bias.

## Current Known Issues

- Qwen VL still shows a tendency to loop or avoid reaching a verdict on small zero-shot runs.
- VL quality depends heavily on actual extracted frames being present and informative.
- The rollout is still sequential and can be slow on large normalized files even with the new runtime controls.
- The project still uses a lightweight rollout/training scaffold rather than full `verl` integration.

## Recommended Next Moves

1. Re-run Qwen VL evaluation with `--max-samples`, reduced `--max-new-tokens`, and the default first-step-only frame attachment.
2. Inspect whether frame images are actually being attached and used by the model on the server.
3. Review traces for:
   - malformed verdicts,
   - action loops,
   - confusion between misleading factual claims and harmless exaggeration.
4. If zero-shot behavior is still weak:
   - generate SFT trajectories from the real normalized data,
   - fine-tune on the new action protocol,
   - then re-run evaluation.
5. Consider `verl` integration once the task framing and action behavior are stable.

## Notes For Future Work

- Keep this file updated after each major experiment.
- Prefer behavior changes that improve trace quality before scaling training complexity.
