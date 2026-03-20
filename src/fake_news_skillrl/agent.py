from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .parser import parse_action
from .schema import FakeNewsSample, FrameRecord


DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen3.5-2B"


def select_inference_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


class BaseFakeNewsAgent(ABC):
    model_name: str

    @abstractmethod
    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        raise NotImplementedError

    def get_last_debug(self) -> Dict[str, Any]:
        return {}


@dataclass(slots=True)
class HeuristicFakeNewsAgent(BaseFakeNewsAgent):
    model_name: str = "heuristic-v1"
    last_debug: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        action_kinds = [item.split(":", 1)[0] for item in inspected_items]
        if "create" not in action_kinds:
            action = "<create>Break the post into its main factual claim and any credibility risks.</create>"
        elif "check" not in action_kinds:
            action = "<check>Compare the caption, transcript, OCR, and attached frames for support or contradiction of the main claim.</check>"
        elif "use_skill" not in action_kinds:
            action = "<use_skill>Apply the most relevant credibility skill: separate expressive exaggeration from concrete misleading factual claims.</use_skill>"
        else:
            action = self._verdict_action(sample)
        self.last_debug = {
            "agent_type": "heuristic",
            "model_name": self.model_name,
            "observation": observation,
            "inspected_items": list(inspected_items),
            "raw_output": action,
            "selected_action": action,
            "fallback_used": False,
        }
        return action

    def _verdict_action(self, sample: FakeNewsSample) -> str:
        rationale = self._build_rationale(sample)
        payload = {
            "label": self._predict_label(sample),
            "rationale": rationale,
        }
        return f"<verdict>{json.dumps(payload, ensure_ascii=True)}</verdict>"

    def _predict_label(self, sample: FakeNewsSample) -> str:
        combined = " ".join(
            [
                sample.post_text.lower(),
                sample.transcript.lower(),
                sample.ocr_text.lower(),
            ]
        )
        suspicious_patterns = [
            "guaranteed",
            "miracle",
            "every virus",
            "archive",
            "old footage",
            "100% cure",
        ]
        trustworthy_patterns = [
            "official",
            "emergency office",
            "pending inspection",
            "road closure",
        ]

        suspicious_score = sum(1 for pattern in suspicious_patterns if pattern in combined)
        trustworthy_score = sum(1 for pattern in trustworthy_patterns if pattern in combined)

        if suspicious_score > trustworthy_score:
            return "fake"
        return "real"

    def _build_rationale(self, sample: FakeNewsSample) -> str:
        label = self._predict_label(sample)
        if label == "fake":
            return "The post makes a misleading or non-factual claim when judged against the provided post information and frames."
        return "The post does not present a concrete misleading factual claim in the provided post information and frames."

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self.last_debug)


@dataclass(slots=True)
class QwenVLAgent(BaseFakeNewsAgent):
    model_name: str = DEFAULT_QWEN_VL_MODEL
    max_new_tokens: int = 160
    temperature: float = 0.0
    trust_remote_code: bool = False
    attach_frames_first_step_only: bool = True
    allow_heuristic_fallback: bool = False
    last_debug: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for QwenVLAgent. "
                "Install it first or use the heuristic agent."
            ) from exc

        self._device = select_inference_device(torch.cuda.is_available())
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        messages = self._build_messages(sample, observation, inspected_items)
        self.last_debug = {
            "agent_type": "qwen_vl",
            "model_name": self.model_name,
            "observation": observation,
            "inspected_items": list(inspected_items),
            "messages": messages,
            "raw_output": None,
            "selected_action": None,
            "fallback_used": False,
            "fallback_reason": None,
            "parse_failure_reason": None,
        }
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(self.temperature, 1e-5),
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
        generated = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
        ).strip()
        self.last_debug["raw_output"] = generated
        action = self._extract_first_action(generated, sample)
        if action is None:
            parse_failure_reason = self._explain_parse_failure(generated, sample)
            invalid_action = generated.strip().replace("<|im_end|>", "").strip()
            self.last_debug["selected_action"] = invalid_action
            self.last_debug["parse_failure_reason"] = parse_failure_reason
            if self.allow_heuristic_fallback:
                fallback = HeuristicFakeNewsAgent()
                fallback_action = fallback.next_action(sample, inspected_items, observation)
                self.last_debug["selected_action"] = fallback_action
                self.last_debug["fallback_used"] = True
                self.last_debug["fallback_reason"] = "model_output_did_not_parse_to_a_valid_action"
                self.last_debug["fallback_debug"] = fallback.get_last_debug()
                return fallback_action
            return invalid_action
        self.last_debug["selected_action"] = action
        return action

    def _build_messages(
        self,
        sample: FakeNewsSample,
        observation: str,
        inspected_items: List[str],
    ) -> List[dict]:
        content: List[dict] = [
            {
                "type": "text",
                "text": (
                    f"{observation}\n"
                    "Respond with exactly one valid action block and no extra commentary.\n"
                    "All provided inputs are already available in the case.\n"
                    "Use create, check, use_skill, and verdict actions only.\n"
                ),
            }
        ]

        should_attach_frames = (not self.attach_frames_first_step_only) or not inspected_items
        for frame in sample.frames:
            if should_attach_frames:
                image_part = self._frame_to_content_part(frame)
                if image_part is not None:
                    content.append(image_part)

        return [{"role": "user", "content": content}]

    @staticmethod
    def _frame_to_content_part(frame: FrameRecord) -> dict | None:
        path = frame.path.strip()
        if not path:
            return None
        if path.startswith("http://") or path.startswith("https://"):
            return {"type": "image", "url": path}
        if Path(path).exists():
            return {"type": "image", "path": path}
        return None

    @staticmethod
    def _extract_first_action(text: str, sample: FakeNewsSample) -> str | None:
        candidates: Sequence[str] = text.splitlines() if "\n" in text else [text]
        max_frame_index = max(0, len(sample.frames) - 1)
        for candidate in candidates:
            candidate = candidate.strip().replace("<|im_end|>", "").strip()
            if not candidate:
                continue
            parsed = parse_action(candidate, max_frame_index=max_frame_index)
            if parsed.is_valid:
                return candidate
        stripped = text.strip().replace("<|im_end|>", "").strip()
        parsed = parse_action(stripped, max_frame_index=max_frame_index)
        if parsed.is_valid:
            return stripped
        return None

    @staticmethod
    def _explain_parse_failure(text: str, sample: FakeNewsSample) -> str:
        candidates: Sequence[str] = text.splitlines() if "\n" in text else [text]
        max_frame_index = max(0, len(sample.frames) - 1)
        candidate_errors: List[str] = []
        for candidate in candidates:
            cleaned = candidate.strip().replace("<|im_end|>", "").strip()
            if not cleaned:
                continue
            parsed = parse_action(cleaned, max_frame_index=max_frame_index)
            if parsed.is_valid:
                return "unexpected_parse_success"
            candidate_errors.append(f"{cleaned[:120]} -> {parsed.error}")
        stripped = text.strip().replace("<|im_end|>", "").strip()
        if stripped:
            parsed = parse_action(stripped, max_frame_index=max_frame_index)
            if parsed.error:
                candidate_errors.append(f"full_output -> {parsed.error}")
        if candidate_errors:
            return " | ".join(candidate_errors[:5])
        return "model_output_was_empty_after_cleanup"

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self.last_debug)


def build_agent(
    agent_type: str,
    model_name: str | None = None,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    trust_remote_code: bool = False,
    attach_frames_first_step_only: bool = True,
    allow_heuristic_fallback: bool = False,
) -> BaseFakeNewsAgent:
    if agent_type == "heuristic":
        return HeuristicFakeNewsAgent(model_name=model_name or "heuristic-v1")
    if agent_type in {"transformers", "qwen_vl"}:
        return QwenVLAgent(
            model_name=model_name or DEFAULT_QWEN_VL_MODEL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            trust_remote_code=trust_remote_code,
            attach_frames_first_step_only=attach_frames_first_step_only,
            allow_heuristic_fallback=allow_heuristic_fallback,
        )
    raise ValueError(f"Unsupported agent type: {agent_type}")
