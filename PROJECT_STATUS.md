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
- The project has now been refactored from a single-agent stage pipeline into a controlled two-agent collaboration loop:
  - Analyzer report
  - Worker skill retrieval/creation
  - Analyzer final verdict
- Skill retrieval is now delayed until the Worker stage instead of being injected at reset.
- Intermediate collaboration stages now accept short plain-text outputs, while only the final verdict remains strictly parsed as XML+JSON.
- Local validation currently passes:
  - full manual test-function suite
  - heuristic smoke evaluation through the controlled pipeline

## Current Known Issues

- Qwen VL may still produce weak Analyzer/Worker outputs, but stage control no longer depends on the model choosing the next action correctly.
- VL quality depends heavily on actual extracted frames being present and informative.
- The rollout is still sequential and can be slow on large normalized files even with the new runtime controls.
- The project still uses a lightweight rollout/training scaffold rather than full `verl` integration.
- Dynamic skill creation is only prompt-level today; newly created skills are not yet persisted back into memory.

## Recommended Next Moves

1. Re-run Qwen VL evaluation under the two-agent controlled pipeline and inspect the Analyzer report, Worker skill, and final verdict together.
2. Check whether the Analyzer report is actually grounded in the attached frames rather than parroting generic captions.
3. Evaluate whether the dynamically retrieved Worker skill improves the Analyzer's final verdict rationale.
4. If zero-shot behavior is still weak:
   - generate SFT trajectories from the controlled pipeline,
   - fine-tune stage by stage,
   - then re-run evaluation.
5. Add persistent dynamic skill creation and memory update once the controlled pipeline is stable.
6. Consider `verl` integration once the controlled pipeline and dynamic skill stage are stable.

## Notes For Future Work

- Keep this file updated after each major experiment.
- Prefer behavior changes that improve trace quality before scaling training complexity.
