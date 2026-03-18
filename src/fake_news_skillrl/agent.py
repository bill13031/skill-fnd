from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

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


@dataclass(slots=True)
class HeuristicFakeNewsAgent(BaseFakeNewsAgent):
    model_name: str = "heuristic-v1"

    def next_action(
        self,
        sample: FakeNewsSample,
        inspected_items: List[str],
        observation: str,
    ) -> str:
        del observation
        inspection_order = ["post_text", "transcript", "metadata"]
        for target in inspection_order:
            if target not in inspected_items:
                return f"<inspect>{target}</inspect>"

        return self._verdict_action(sample)

    def _verdict_action(self, sample: FakeNewsSample) -> str:
        evidence = list(sample.gold_evidence[:3])
        rationale = self._build_rationale(sample)
        payload = {
            "label": self._predict_label(sample),
            "rationale": rationale,
            "evidence": evidence,
        }
        return f"<verdict>{json.dumps(payload, ensure_ascii=True)}</verdict>"

    def _predict_label(self, sample: FakeNewsSample) -> str:
        combined = " ".join(
            [
                sample.post_text.lower(),
                sample.transcript.lower(),
                sample.ocr_text.lower(),
                " ".join(frame.description.lower() for frame in sample.frames),
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
        if trustworthy_score > suspicious_score:
            return "real"
        return "unverified"

    def _build_rationale(self, sample: FakeNewsSample) -> str:
        label = self._predict_label(sample)
        if label == "fake":
            return "The post contains unsupported or contradictory cues across text, transcript, OCR, or frame context."
        if label == "real":
            return "The available text, metadata, and frame evidence are internally consistent and source signals look credible."
        return "The provided evidence package is insufficient for a confident factual judgment."


@dataclass(slots=True)
class QwenVLAgent(BaseFakeNewsAgent):
    model_name: str = DEFAULT_QWEN_VL_MODEL
    max_new_tokens: int = 160
    temperature: float = 0.0
    trust_remote_code: bool = False

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
        action = self._extract_first_action(generated, sample)
        if action is None:
            fallback = HeuristicFakeNewsAgent()
            return fallback.next_action(sample, inspected_items, observation)
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
                    "If frame images are attached, use them together with the textual evidence.\n"
                ),
            }
        ]

        attached_frames = self._frames_for_items(sample, inspected_items)
        for frame in attached_frames:
            image_part = self._frame_to_content_part(frame)
            if image_part is not None:
                content.append(image_part)
            if frame.description:
                content.append(
                    {
                        "type": "text",
                        "text": f"Frame {frame.frame_id} description: {frame.description}",
                    }
                )

        if not attached_frames:
            for frame in sample.frames[:1]:
                if frame.description:
                    content.append(
                        {
                            "type": "text",
                            "text": f"Available frame context: {frame.description}",
                        }
                    )

        return [{"role": "user", "content": content}]

    @staticmethod
    def _frames_for_items(sample: FakeNewsSample, inspected_items: List[str]) -> List[FrameRecord]:
        frames: List[FrameRecord] = []
        for item in inspected_items:
            if not item.startswith("frame:"):
                continue
            index = int(item.split(":", 1)[1])
            if 0 <= index < len(sample.frames):
                frames.append(sample.frames[index])
        return frames

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
            candidate = candidate.strip()
            if not candidate:
                continue
            parsed = parse_action(candidate, max_frame_index=max_frame_index)
            if parsed.is_valid:
                return candidate
        stripped = text.strip()
        parsed = parse_action(stripped, max_frame_index=max_frame_index)
        if parsed.is_valid:
            return stripped
        return None


def build_agent(
    agent_type: str,
    model_name: str | None = None,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    trust_remote_code: bool = False,
) -> BaseFakeNewsAgent:
    if agent_type == "heuristic":
        return HeuristicFakeNewsAgent(model_name=model_name or "heuristic-v1")
    if agent_type in {"transformers", "qwen_vl"}:
        return QwenVLAgent(
            model_name=model_name or DEFAULT_QWEN_VL_MODEL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f"Unsupported agent type: {agent_type}")
