from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence


class MultimodalBackend(str, Enum):
    """
    Supported multimodal backends.

    - OPENAI_CHAT_COMPLETIONS: OpenAI-compatible chat completions API
      This works for local VLLM-served models and OpenRouter-compatible endpoints.
    - OPENAI_API: direct OpenAI API access for hosted GPT-style models
    - GEMINI_GOOGLE: Google Gemini client
    """

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_API = "openai_api"
    GEMINI_GOOGLE = "google_gemini"


@dataclass(frozen=True)
class MultimodalLLMConfig:
    backend: MultimodalBackend = MultimodalBackend.OPENAI_CHAT_COMPLETIONS
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_s: float = 60.0
    system_prompt: Optional[str] = None
    reasoning_effort: str = "low"
    openrouter_site_url: Optional[str] = None
    openrouter_app_name: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> "MultimodalLLMConfig":
        backend_value = backend or os.getenv("DEMO_VLM_BACKEND") or MultimodalBackend.OPENAI_CHAT_COMPLETIONS.value
        backend_enum = _parse_backend(backend_value)

        default_model = {
            MultimodalBackend.OPENAI_CHAT_COMPLETIONS: "gpt-4o-mini",
            MultimodalBackend.OPENAI_API: "gpt-5",
            MultimodalBackend.GEMINI_GOOGLE: "google/gemini-3.1-flash-lite-preview",
        }[backend_enum]

        if backend_enum == MultimodalBackend.GEMINI_GOOGLE:
            # Gemini is routed through OpenRouter in demo mode.
            api_key = os.getenv("DEMO_VLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        else:
            api_key = os.getenv("DEMO_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        return cls(
            backend=backend_enum,
            model=model or os.getenv("DEMO_VLM_MODEL") or default_model,
            api_key=api_key,
            base_url=os.getenv("DEMO_VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            temperature=float(os.getenv("DEMO_VLM_TEMPERATURE", "0.0")),
            max_output_tokens=int(os.getenv("DEMO_VLM_MAX_OUTPUT_TOKENS", "512")),
            timeout_s=float(os.getenv("DEMO_VLM_TIMEOUT_S", "60.0")),
            system_prompt=system_prompt,
            reasoning_effort=os.getenv("DEMO_VLM_REASONING_EFFORT", "low"),
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL"),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME"),
        )


def call_multimodal_llm(
    *,
    prompt: str,
    frame_b64: Optional[str] = None,
    frame_mime_type: str = "image/png",
    system_prompt: Optional[str] = None,
    config: Optional[MultimodalLLMConfig] = None,
) -> str:
    """
    Call a multimodal LLM with an optional image frame and a text prompt.

    This helper is intentionally backend-agnostic so the same entrypoint can be
    reused for student-checking and hint generation.
    """
    cfg = config or MultimodalLLMConfig.from_env(system_prompt=system_prompt)
    resolved_system_prompt = system_prompt if system_prompt is not None else cfg.system_prompt

    if cfg.backend == MultimodalBackend.GEMINI_GOOGLE:
        return _call_gemini_openrouter(
            prompt=prompt,
            frame_b64=frame_b64,
            frame_mime_type=frame_mime_type,
            model=cfg.model,
            api_key=cfg.api_key,
            system_prompt=resolved_system_prompt,
            timeout_s=cfg.timeout_s,
            reasoning_effort=cfg.reasoning_effort,
        )

    return _call_openai_compatible(
        prompt=prompt,
        frame_b64=frame_b64,
        frame_mime_type=frame_mime_type,
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        system_prompt=resolved_system_prompt,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        timeout_s=cfg.timeout_s,
        openrouter_site_url=cfg.openrouter_site_url,
        openrouter_app_name=cfg.openrouter_app_name,
    )


def encode_image_to_base64(image) -> str:
    """
    Convert a PIL image into PNG base64 for transport/storage.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_openai_compatible(
    *,
    prompt: str,
    frame_b64: Optional[str],
    frame_mime_type: str,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    max_output_tokens: int,
    timeout_s: float,
    openrouter_site_url: Optional[str] = None,
    openrouter_app_name: Optional[str] = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("openai package is required for OpenAI-compatible VLM calls") from exc

    client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout_s:
        client_kwargs["timeout"] = timeout_s

    if openrouter_site_url:
        client_kwargs.setdefault("default_headers", {})
        client_kwargs["default_headers"].update({"HTTP-Referer": openrouter_site_url})
    if openrouter_app_name:
        client_kwargs.setdefault("default_headers", {})
        client_kwargs["default_headers"].update({"X-Title": openrouter_app_name})

    client = OpenAI(**client_kwargs)

    content = [{"type": "text", "text": prompt}]
    if frame_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_uri(frame_b64, frame_mime_type)},
            }
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )

    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    content_text = getattr(message, "content", None) if message is not None else None
    return str(content_text or "").strip()


def _call_gemini_openrouter(
    *,
    prompt: str,
    frame_b64: Optional[str],
    frame_mime_type: str,
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    timeout_s: float,
    reasoning_effort: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("openai package is required for OpenRouter Gemini calls") from exc

    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter Gemini calls")

    request_id, request_path = _dump_gemini_request(
        prompt=prompt,
        frame_b64=frame_b64,
        frame_mime_type=frame_mime_type,
        model=model,
        system_prompt=system_prompt,
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=resolved_api_key,
        timeout=timeout_s,
    )

    messages: list[dict] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    user_content: list[dict] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    if frame_b64:
        data_url = f"data:{frame_mime_type};base64,{frame_b64}"
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            extra_body={
                "reasoning": {
                    "enabled": True,
                    "effort": reasoning_effort,
                },
            },
        )

        message = response.choices[0].message
        text = message.content
        output_text = str(text or "").strip()
        _dump_gemini_output(request_path=request_path, output_text=output_text)
        return output_text
    except Exception as exc:
        _dump_gemini_output(request_path=request_path, output_text=f"[ERROR] {exc}")
        raise


def _data_uri(frame_b64: str, frame_mime_type: str) -> str:
    return f"data:{frame_mime_type};base64,{frame_b64}"


def _dump_gemini_request(
    *,
    prompt: str,
    frame_b64: Optional[str],
    frame_mime_type: str,
    model: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], Optional[Path]]:
    dump_dir = _gemini_dump_dir()
    if dump_dir is None:
        return None, None

    request_id = _timestamp_request_id()
    dump_dir.mkdir(parents=True, exist_ok=True)

    request_path = dump_dir / f"{request_id}.json"
    request_payload = {
        "request_id": request_id,
        "model": model,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "frame_mime_type": frame_mime_type,
        "has_image": bool(frame_b64),
        "output_text": None,
    }
    request_path.write_text(json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if frame_b64:
        image_path = dump_dir / f"{request_id}.png"
        image_path.write_bytes(base64.b64decode(frame_b64))

    return request_id, request_path


def _dump_gemini_output(*, request_path: Optional[Path], output_text: str) -> None:
    if request_path is None:
        return

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    payload["output_text"] = output_text
    request_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _gemini_dump_dir() -> Optional[Path]:
    raw_dir = os.getenv("DEMO_GEMINI_DUMP_DIR")
    if raw_dir:
        return Path(raw_dir)

    if os.getenv("DEMO_GEMINI_DUMP_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    return Path("tmp") / "gemini_calls"


def _timestamp_request_id() -> str:
    """
    Build a human-readable request id based on UTC time.

    Example: 20260413T142233.123456Z
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parse_backend(raw_value: str) -> MultimodalBackend:
    value = str(raw_value).strip().lower()
    for backend in MultimodalBackend:
        if backend.value == value:
            return backend
    return MultimodalBackend.OPENAI_CHAT_COMPLETIONS
