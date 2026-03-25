from __future__ import annotations

import json
import base64
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .parser import parse_action
from .schema import FakeNewsSample, FrameRecord
from .parser import CREATE_RE, CHECK_RE, USE_SKILL_RE, VERDICT_RE


DEFAULT_QWEN_VL_MODEL = "./model/Qwen3.5-2B"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_DASHSCOPE_MODEL = "qwen3.5-plus"
CURRENT_STAGE_RE = re.compile(r"## Current Stage\n([a-z_]+)", re.IGNORECASE)
CURRENT_ROLE_RE = re.compile(r"## Current Role\n([a-z_]+)", re.IGNORECASE)


def select_inference_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def _clean_generation_text(text: str) -> str:
    return text.strip().replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def repair_verdict_output(text: str) -> str | None:
    cleaned = _clean_generation_text(text)
    if not cleaned:
        return None
    if cleaned.startswith("<verdict>") and cleaned.endswith("</verdict>"):
        return cleaned
    if cleaned.startswith("{") and cleaned.endswith("}"):
        parsed = parse_action(f"<verdict>{cleaned}</verdict>", max_frame_index=0)
        if parsed.is_valid:
            return f"<verdict>{cleaned}</verdict>"

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1].strip()
        parsed = parse_action(f"<verdict>{candidate}</verdict>", max_frame_index=0)
        if parsed.is_valid:
            return f"<verdict>{candidate}</verdict>"
    return None


def _frame_path_to_image_url(path: str) -> str | None:
    normalized = path.strip()
    if not normalized:
        return None
    if normalized.startswith("http://") or normalized.startswith("https://") or normalized.startswith("file://"):
        return normalized
    file_path = Path(normalized)
    if not file_path.exists():
        return None
    return f"file://{file_path.resolve()}"


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
class AnalyzerAgent(BaseFakeNewsAgent):
    backend: BaseFakeNewsAgent
    model_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_name = f"analyzer::{self.backend.model_name}"

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        return self.backend.next_action(sample, inspected_items, observation)

    def get_last_debug(self) -> Dict[str, Any]:
        debug = getattr(self.backend, "get_last_debug", lambda: {})()
        return {
            **debug,
            "agent_wrapper_role": "analyzer",
        }


@dataclass(slots=True)
class WorkerAgent(BaseFakeNewsAgent):
    backend: BaseFakeNewsAgent
    model_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_name = f"worker::{self.backend.model_name}"

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        return self.backend.next_action(sample, inspected_items, observation)

    def get_last_debug(self) -> Dict[str, Any]:
        debug = getattr(self.backend, "get_last_debug", lambda: {})()
        return {
            **debug,
            "agent_wrapper_role": "worker",
        }


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
        current_stage = self._detect_stage(observation)
        if current_stage == "event_extraction":
            event_text = sample.event_text.strip() or "The post reports a concrete real-world event."
            action = f"Event: {event_text}"
        elif current_stage == "frame_description":
            frame_label = sample.frames[0].description.strip() if sample.frames else ""
            frame_ocr = sample.frames[0].ocr_text.strip() if sample.frames else ""
            action = (
                f"Frame description: {frame_label or 'A frame from the post showing the main visible scene.'}\n"
                f"OCR: {frame_ocr or '[none]'}"
            )
        elif current_stage == "preliminary_analysis":
            action = (
                "Preliminary reasoning: The extracted event makes a concrete factual claim whose credibility depends on whether the post provides trustworthy support rather than merely related imagery or framing.\n"
                "Need: Need a verification skill that helps judge whether the extracted event is credibly supported or misleading."
            )
        elif current_stage == "worker_skill":
            action = "Skill: Topic-matched or stylized imagery does not verify an extraordinary real-world claim without documentary support."
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
            return "The post makes an unsupported extraordinary factual claim that is not credibly verified by the provided inputs."
        return "The post does not present a concrete misleading factual claim in the provided post information and frames."

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self.last_debug)

    @staticmethod
    def _detect_stage(observation: str) -> str:
        match = CURRENT_STAGE_RE.search(observation)
        if match is None:
            return "verdict"
        return match.group(1).strip().lower()

    @staticmethod
    def _detect_role(observation: str) -> str:
        match = CURRENT_ROLE_RE.search(observation)
        if match is None:
            return "analyzer"
        return match.group(1).strip().lower()


@dataclass(slots=True)
class QwenVLAgent(BaseFakeNewsAgent):
    model_name: str = DEFAULT_QWEN_VL_MODEL
    max_new_tokens: int = 192
    temperature: float = 0.0
    repetition_penalty: float = 1.02
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
        current_stage = HeuristicFakeNewsAgent._detect_stage(observation)
        messages = self._build_messages(sample, observation, inspected_items, current_stage)
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
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
        generated = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
        ).strip()
        self.last_debug["raw_output"] = generated
        action = self._extract_stage_output(generated, sample, current_stage)
        if action is None:
            parse_failure_reason = self._explain_parse_failure(generated, sample)
            invalid_action = _clean_generation_text(generated)
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
        current_stage: str = "preliminary_analysis",
    ) -> List[dict]:
        if current_stage == "verdict":
            response_rule = (
                "Return exactly one <verdict>{\"label\":\"fake|real\",\"rationale\":\"...\"}</verdict> block and nothing else.\n"
            )
        else:
            response_rule = (
                "Return only the requested stage output as plain text.\n"
                "Do not wrap it in XML unless the stage explicitly asks for a verdict block.\n"
            )
        content: List[dict] = [
            {
                "type": "text",
                "text": (
                    f"{observation}\n"
                    f"{response_rule}"
                    "All provided inputs are already available in the case.\n"
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
        max_frame_index = max(0, len(sample.frames) - 1)
        first_complete_block = QwenVLAgent._extract_first_complete_block(text)
        if first_complete_block is not None:
            parsed = parse_action(first_complete_block, max_frame_index=max_frame_index)
            if parsed.is_valid:
                return first_complete_block

        candidates: Sequence[str] = text.splitlines() if "\n" in text else [text]
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
    def _extract_first_complete_block(text: str) -> str | None:
        cleaned = _clean_generation_text(text)
        if not cleaned:
            return None
        matches: List[tuple[int, str]] = []
        for pattern in (CREATE_RE, CHECK_RE, USE_SKILL_RE, VERDICT_RE):
            match = pattern.search(cleaned)
            if match is not None:
                matches.append((match.start(), match.group(0).strip()))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    @staticmethod
    def _extract_stage_output(text: str, sample: FakeNewsSample, current_stage: str) -> str | None:
        if current_stage == "verdict":
            direct = QwenVLAgent._extract_first_action(text, sample)
            if direct is not None:
                return direct
            return repair_verdict_output(text)
        cleaned = _clean_generation_text(text)
        if not cleaned:
            return None
        first_complete_block = QwenVLAgent._extract_first_complete_block(cleaned)
        if first_complete_block is None:
            return cleaned
        parsed = parse_action(first_complete_block, max_frame_index=max(0, len(sample.frames) - 1))
        if parsed.is_valid and parsed.action_type != "verdict":
            return str(parsed.payload.get("content", "")).strip()
        return cleaned

    @staticmethod
    def _explain_parse_failure(text: str, sample: FakeNewsSample) -> str:
        candidates: Sequence[str] = text.splitlines() if "\n" in text else [text]
        max_frame_index = max(0, len(sample.frames) - 1)
        cleaned = _clean_generation_text(text)
        opening_tag_count = sum(cleaned.lower().count(tag) for tag in ("<create>", "<check>", "<use_skill>", "<verdict>"))
        if opening_tag_count > 1:
            first_complete_block = QwenVLAgent._extract_first_complete_block(cleaned)
            if first_complete_block is not None:
                return (
                    "model_output_contains_multiple_action_blocks_in_one_turn; "
                    f"first_complete_block={first_complete_block[:200]}"
                )
            return "model_output_contains_multiple_action_blocks_in_one_turn"
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


@dataclass(slots=True)
class OpenAIResponsesAgent(BaseFakeNewsAgent):
    model_name: str = DEFAULT_OPENAI_MODEL
    max_output_tokens: int = 300
    attach_frames_first_step_only: bool = True
    last_debug: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAIResponsesAgent. Install it first or use another backend."
            ) from exc

        self._client = OpenAI()

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        current_stage = HeuristicFakeNewsAgent._detect_stage(observation)
        user_input = self._build_responses_input(sample, observation, inspected_items, current_stage)
        self.last_debug = {
            "agent_type": "openai_responses",
            "model_name": self.model_name,
            "observation": observation,
            "inspected_items": list(inspected_items),
            "input": user_input,
            "raw_output": None,
            "selected_action": None,
            "fallback_used": False,
            "fallback_reason": None,
            "parse_failure_reason": None,
        }
        response = self._client.responses.create(
            model=self.model_name,
            input=user_input,
            max_output_tokens=self.max_output_tokens,
        )
        generated = _clean_generation_text(getattr(response, "output_text", "") or "")
        self.last_debug["raw_output"] = generated
        action = QwenVLAgent._extract_stage_output(generated, sample, current_stage)
        if action is None:
            parse_failure_reason = QwenVLAgent._explain_parse_failure(generated, sample)
            invalid_action = _clean_generation_text(generated)
            self.last_debug["selected_action"] = invalid_action
            self.last_debug["parse_failure_reason"] = parse_failure_reason
            return invalid_action
        self.last_debug["selected_action"] = action
        return action

    def _build_responses_input(
        self,
        sample: FakeNewsSample,
        observation: str,
        inspected_items: List[str],
        current_stage: str,
    ) -> List[dict]:
        if current_stage == "verdict":
            response_rule = (
                "Return exactly one JSON verdict object or one <verdict>{...}</verdict> block with label fake|real and a short rationale.\n"
            )
        else:
            response_rule = (
                "Return only the requested stage output as plain text.\n"
            )
        content: List[dict] = [
            {
                "type": "input_text",
                "text": (
                    f"{observation}\n"
                    f"{response_rule}"
                    "All provided inputs are already available in the case.\n"
                ),
            }
        ]
        should_attach_frames = (not self.attach_frames_first_step_only) or not inspected_items
        for frame in sample.frames:
            if not should_attach_frames:
                continue
            image_part = self._frame_to_responses_part(frame)
            if image_part is not None:
                content.append(image_part)
        return [{"role": "user", "content": content}]

    @staticmethod
    def _frame_to_responses_part(frame: FrameRecord) -> dict | None:
        path = frame.path.strip()
        if not path:
            return None
        if path.startswith("http://") or path.startswith("https://"):
            return {"type": "input_image", "image_url": path}
        file_path = Path(path)
        if not file_path.exists():
            return None
        mime_type, _ = mimetypes.guess_type(file_path.name)
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return {"type": "input_image", "image_url": f"data:{mime_type};base64,{encoded}"}

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self.last_debug)


@dataclass(slots=True)
class DashScopeMultiModalAgent(BaseFakeNewsAgent):
    model_name: str = DEFAULT_DASHSCOPE_MODEL
    max_output_tokens: int = 300
    attach_frames_first_step_only: bool = True
    api_key: str | None = None
    base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    last_debug: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as exc:
            raise ImportError(
                "dashscope is required for DashScopeMultiModalAgent. Install it first or use another backend."
            ) from exc
        dashscope.base_http_api_url = self.base_http_api_url
        self._dashscope = dashscope
        self._conversation_cls = MultiModalConversation

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        current_stage = HeuristicFakeNewsAgent._detect_stage(observation)
        messages = self._build_dashscope_messages(sample, observation, inspected_items, current_stage)
        self.last_debug = {
            "agent_type": "dashscope_multimodal",
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
        response = self._conversation_cls.call(
            api_key=self.api_key or __import__("os").environ.get("DASHSCOPE_API_KEY"),
            model=self.model_name,
            messages=messages,
        )
        generated = self._extract_text_from_dashscope_response(response)
        self.last_debug["raw_output"] = generated
        action = QwenVLAgent._extract_stage_output(generated, sample, current_stage)
        if action is None:
            parse_failure_reason = QwenVLAgent._explain_parse_failure(generated, sample)
            invalid_action = _clean_generation_text(generated)
            self.last_debug["selected_action"] = invalid_action
            self.last_debug["parse_failure_reason"] = parse_failure_reason
            return invalid_action
        self.last_debug["selected_action"] = action
        return action

    def _build_dashscope_messages(
        self,
        sample: FakeNewsSample,
        observation: str,
        inspected_items: List[str],
        current_stage: str,
    ) -> List[dict]:
        if current_stage == "verdict":
            response_rule = (
                "Return exactly one JSON verdict object or one <verdict>{...}</verdict> block with label fake|real and a short rationale.\n"
            )
        else:
            response_rule = "Return only the requested stage output as plain text.\n"
        content: List[dict] = [{"text": f"{observation}\n{response_rule}All provided inputs are already available in the case.\n"}]
        should_attach_frames = (not self.attach_frames_first_step_only) or not inspected_items
        for frame in sample.frames:
            if not should_attach_frames:
                continue
            image_url = _frame_path_to_image_url(frame.path)
            if image_url is not None:
                content.insert(0, {"image": image_url})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_text_from_dashscope_response(response: Any) -> str:
        try:
            content = response.output.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
        return _clean_generation_text("\n".join(text_parts))

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self.last_debug)


def build_agent(
    agent_type: str,
    model_name: str | None = None,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    repetition_penalty: float = 1.02,
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
            repetition_penalty=repetition_penalty,
            trust_remote_code=trust_remote_code,
            attach_frames_first_step_only=attach_frames_first_step_only,
            allow_heuristic_fallback=allow_heuristic_fallback,
        )
    if agent_type in {"aliyun_sdk", "dashscope", "dashscope_multimodal"}:
        return DashScopeMultiModalAgent(
            model_name=model_name or DEFAULT_DASHSCOPE_MODEL,
            max_output_tokens=max_new_tokens,
            attach_frames_first_step_only=attach_frames_first_step_only,
        )
    if agent_type in {"openai", "openai_sdk", "openai_responses"}:
        return OpenAIResponsesAgent(
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            max_output_tokens=max_new_tokens,
            attach_frames_first_step_only=attach_frames_first_step_only,
        )
    raise ValueError(f"Unsupported agent type: {agent_type}")


def build_agent_pair(
    agent_type: str,
    model_name: str | None = None,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    repetition_penalty: float = 1.02,
    trust_remote_code: bool = False,
    attach_frames_first_step_only: bool = True,
    allow_heuristic_fallback: bool = False,
) -> tuple[AnalyzerAgent, WorkerAgent]:
    backend = build_agent(
        agent_type=agent_type,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        trust_remote_code=trust_remote_code,
        attach_frames_first_step_only=attach_frames_first_step_only,
        allow_heuristic_fallback=allow_heuristic_fallback,
    )
    return AnalyzerAgent(backend=backend), WorkerAgent(backend=backend)
