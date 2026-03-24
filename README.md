# Fake News SkillRL

`fake-news-skillrl` is a standalone, lightweight recreation of the core SkillRL idea for multimodal social-video credibility assessment. It does not depend on the `SkillRL` package at runtime, but it borrows several design ideas from it:

- structured verdict parsing,
- skill-bank retrieval for prompt injection,
- multi-step environment rollouts,
- prompt-first task framing for later VL model integration.

The current version focuses on a practical local pipeline:

- normalize short-video samples into a common schema,
- run a controlled two-agent episode over caption text, transcript/OCR if available, and sampled video frames,
- finish with a structured verdict:

```xml
<verdict>{"label":"fake","rationale":"..."}</verdict>
```

## Project Layout

```text
fake-news-skillrl/
  config/
  data/
  memory_data/fake_news/
  scripts/
  src/fake_news_skillrl/
  tests/
```

## Core Concepts

### Normalized Sample Schema

Each sample is normalized into a single record with the following keys:

- `sample_id`
- `post_text`
- `transcript`
- `ocr_text`
- `metadata`
- `frames`
- `label`
- `split`
- `data_source`

`frames` is a list of extracted frame image paths. In v1, this is frame-first rather than native video tensor input.

### Task Framing

The agent should behave like a short-video content credibility analyst, not a generic “fake news” classifier.

- `fake` means the post contains misleading or non-factual content presented as true or documentary.
- `real` means the post is factual, benign, or expressive without making a misleading factual claim.
Harmless exaggeration, metaphor, humor, or expressive social-video language should be allowed to pass when it is not making a concrete misleading factual claim.

### Controlled Two-Agent Pipeline

The environment provides the full case package up front and the controller decides the collaboration stage. The model does not choose the next stage itself.

The current controlled stages are:

- `analyzer_report`
- `worker_skill`
- `verdict`

The collaboration pattern is:

1. `Analyzer` reads the post and frames, then writes a short report:
   - what is visually shown
   - what claim the post makes
   - what kind of skill would help judge the case
2. `Worker` reads that report, dynamically retrieves relevant skills, and returns one short skill for the Analyzer.
3. `Analyzer` uses the original case plus the Worker-provided skill to return the final verdict.

Only the final stage is strictly structured:

```xml
<verdict>{"label":"fake|real","rationale":"..."}</verdict>
```

This design keeps stage control deterministic while still allowing dynamic skill retrieval at the `worker_skill` stage.

### Skill Bank

The skill bank uses the same high-level structure as SkillRL:

- `general_skills`
- `task_specific_skills`
- `common_mistakes`

Template retrieval is implemented first. Embedding retrieval is left as a future extension point.

Skills are no longer injected at reset. They are retrieved dynamically only when the episode reaches the `worker_skill` stage.

The repo now also includes a small DuckDuckGo utility in [web_search.py](/home/yang/SkillFND/fake-news-skillrl/src/fake_news_skillrl/web_search.py) so `web_search`-style skills can be grounded in a concrete search backend when you decide to wire search into the workflow.

## Quick Start

Create normalized smoke-test data:

```bash
python scripts/prepare_dataset.py \
  --input data/raw/smoke_samples.jsonl \
  --output data/normalized/smoke_samples.jsonl
```

Prepare Fakett-style real data:

```bash
python3 scripts/prepare_dataset.py \
  --dataset-format fakett \
  --input data/raw/samples.jsonl \
  --output data/normalized/samples.normalized.jsonl \
  --video-dir ~/datasets/fakett/video \
  --frames-dir data/frames/fakett \
  --fps 2 \
  --max-frames 16
```

Notes:

- If OpenCV is unavailable or frame extraction fails, the normalized rows will still be written, but `frames` may be empty and `metadata.frame_extraction_status` will indicate that.
- Frame extraction now samples by time using `--fps` and caps total saved frames with `--max-frames`. Frames are written into `frames-dir/<sample_id>/`.
- Fakett normalization currently keeps extra metadata for storage, but the model-facing prompt does not expose metadata or frame descriptions.

Generate SFT trajectories:

```bash
python scripts/generate_sft_data.py \
  --input data/normalized/smoke_samples.jsonl \
  --output data/sft/smoke_sft.jsonl
```

Run a lightweight rollout loop:

```bash
python scripts/train_rl.py \
  --input data/normalized/smoke_samples.jsonl \
  --skill-bank memory_data/fake_news/claude_style_skills.json
```

Evaluate the heuristic agent:

```bash
python scripts/evaluate.py \
  --input data/normalized/smoke_samples.jsonl \
  --skill-bank memory_data/fake_news/claude_style_skills.json
```

Evaluate Fakett-style normalized data with the Qwen VL path:

```bash
python scripts/evaluate.py \
  --input data/normalized/samples.normalized.jsonl \
  --skill-bank memory_data/fake_news/claude_style_skills.json \
  --agent-type qwen_vl \
  --model-name ./model/Qwen3.5-2B \
  --max-samples 8 \
  --max-new-tokens 192 \
  --repetition-penalty 1.02 \
  --attach-frames-first-step-only \
  --max-reasoning-steps-before-forced-verdict 4
```

The Qwen VL agent now moves the model to CUDA automatically when `torch.cuda.is_available()` is true, and falls back to CPU otherwise.

Runtime notes:

- Evaluation with a VL model is sequential in this standalone project, so reducing `--max-new-tokens` and `--max-samples` is often the fastest way to iterate.
- `--attach-frames-first-step-only` is enabled by default and prevents re-sending every frame image on later reasoning steps.
- `--max-reasoning-steps-before-forced-verdict` only applies once the collaboration reaches the verdict stage; it prevents the Analyzer from stalling without a valid verdict block.
- The Qwen path now logs per-step observations and raw model output so stage failures are easier to debug.

## Current Scope

This project is fully runnable and testable, but the model-training layer is intentionally lightweight:

- SFT data generation is implemented.
- RL-style episodic rollout, reward computation, and evaluation are implemented.
- The agent/model interface is modular and now includes a Qwen VL-backed path using `AutoProcessor` and `AutoModelForImageTextToText`.
- The model-facing prompt is intentionally simplified: no metadata, no frame descriptions, no evidence-list supervision, and no empty transcript/OCR placeholders.
- The controller now enforces the collaboration order while leaving skill retrieval dynamic at the Worker stage.

## Future Extensions

- stronger frame extraction and video preprocessing,
- embedding-based skill retrieval,
- dynamic skill creation and memory update from solved cases,
- richer Analyzer/Worker coordination once the basic collaboration loop is stable,
- richer reward shaping once better supervision is available,
- direct integration with a large-model trainer.
