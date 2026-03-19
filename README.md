# Fake News SkillRL

`fake-news-skillrl` is a standalone, lightweight recreation of the core SkillRL idea for multimodal social-video credibility assessment. It does not depend on the `SkillRL` package at runtime, but it borrows several design ideas from it:

- structured action parsing,
- skill-bank retrieval for prompt injection,
- multi-step environment rollouts,
- prompt-first task framing for later VL model integration.

The current version focuses on a practical local pipeline:

- normalize short-video samples into a common schema,
- run a short investigative episode over caption text, transcript/OCR if available, metadata, and sampled video frames,
- finish with a structured verdict:

```xml
<verdict>{"label":"fake","rationale":"...","evidence":["..."]}</verdict>
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

`gold_evidence` is optional and is currently not injected for Fakett-style data.

`frames` is a list of extracted frame descriptors. In v1, this is frame-first rather than native video tensor input.

### Task Framing

The agent should behave like a short-video content credibility analyst, not a generic “fake news” classifier.

- `fake` means the post contains misleading or non-factual content presented as true or documentary.
- `real` means the post is factual, benign, or expressive without making a misleading factual claim.
- `unverified` means the provided evidence is insufficient.

Harmless exaggeration, metaphor, humor, or expressive social-video language should be allowed to pass when it is not making a concrete misleading factual claim.

### Action Protocol

The environment provides the full case package up front. The agent should not ask to reveal evidence item by item.

Intermediate actions are:

- `<create>...</create>`
- `<check>...</check>`
- `<use_skill>...</use_skill>`

The episode ends with:

```xml
<verdict>{"label":"fake|real|unverified","rationale":"...","evidence":["..."]}</verdict>
```

### Skill Bank

The skill bank uses the same high-level structure as SkillRL:

- `general_skills`
- `task_specific_skills`
- `common_mistakes`

Template retrieval is implemented first. Embedding retrieval is left as a future extension point.

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
  --num-frames 4
```

Notes:

- If OpenCV is unavailable or frame extraction fails, the normalized rows will still be written, but `frames` may be empty and `metadata.frame_extraction_status` will indicate that.
- Fakett normalization keeps `task_type` neutral as `unknown`.
- Fakett normalization does not currently inject `gold_evidence`.

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
  --model-name ./model/Qwen3.5-2B
```

The Qwen VL agent now moves the model to CUDA automatically when `torch.cuda.is_available()` is true, and falls back to CPU otherwise.

## Current Scope

This project is fully runnable and testable, but the model-training layer is intentionally lightweight:

- SFT data generation is implemented.
- RL-style episodic rollout, reward computation, and evaluation are implemented.
- The agent/model interface is modular and now includes a Qwen VL-backed path using `AutoProcessor` and `AutoModelForImageTextToText`, while the heuristic fallback keeps the smoke workflow runnable.
- For Fakett-style data, evidence scoring is currently diagnostic only because no gold evidence annotations are provided.

## Future Extensions

- stronger frame extraction and video preprocessing,
- embedding-based skill retrieval,
- richer reward shaping once better supervision is available,
- direct integration with a large-model trainer.
