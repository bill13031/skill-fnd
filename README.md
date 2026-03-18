# Fake News SkillRL

`fake-news-skillrl` is a standalone, lightweight recreation of the core SkillRL idea for multimodal fake news detection. It does not depend on the `SkillRL` package at runtime, but it borrows several design ideas from it:

- structured action parsing,
- skill-bank retrieval for prompt injection,
- multi-step environment rollouts,
- prompt-first task framing for later VL model integration.

The first version focuses on a practical local pipeline:

- normalize fake-news samples into a common schema,
- run a short investigative episode over post text, transcript, OCR, metadata, and sampled video frames,
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
- `gold_evidence`
- `split`
- `data_source`

`frames` is a list of frame descriptors. In v1, this is frame-first rather than native video tensor input.

### Action Protocol

The environment supports local evidence inspection actions:

- `<inspect>post_text</inspect>`
- `<inspect>transcript</inspect>`
- `<inspect>ocr_text</inspect>`
- `<inspect>metadata</inspect>`
- `<inspect>frame:0</inspect>`

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

Generate SFT trajectories:

```bash
python scripts/generate_sft_data.py \
  --input data/normalized/smoke_samples.jsonl \
  --output data/sft/smoke_sft.jsonl
```

Run a lightweight RL-style rollout loop:

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

Evaluate with the Qwen VL model path:

```bash
python scripts/evaluate.py \
  --input data/raw/smoke_samples.jsonl \
  --skill-bank memory_data/fake_news/claude_style_skills.json \
  --agent-type qwen_vl \
  --model-name Qwen/Qwen3.5-2B
```

The Qwen VL agent now moves the model to CUDA automatically when `torch.cuda.is_available()` is true, and falls back to CPU otherwise.

## Current Scope

This project is fully runnable and testable, but the model-training layer is intentionally lightweight:

- SFT data generation is implemented.
- RL-style episodic rollout, reward computation, and evaluation are implemented.
- The agent/model interface is modular and now includes a Qwen VL-backed path using `AutoProcessor` and `AutoModelForImageTextToText`, while the heuristic fallback keeps the smoke workflow runnable.

## Future Extensions

- native image loading and VL processor integration,
- embedding-based skill retrieval,
- richer reward shaping with evidence alignment,
- direct integration with a large-model trainer.
