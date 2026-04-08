from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
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
    GEMINI_GOOGLE = "gemini_google"


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
            MultimodalBackend.GEMINI_GOOGLE: "gemini-2.0-flash",
        }[backend_enum]

        if backend_enum == MultimodalBackend.GEMINI_GOOGLE:
            api_key = os.getenv("DEMO_VLM_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
        return _call_gemini_google(
            prompt=prompt,
            frame_b64=frame_b64,
            frame_mime_type=frame_mime_type,
            model=cfg.model,
            api_key=cfg.api_key,
            system_prompt=resolved_system_prompt,
            timeout_s=cfg.timeout_s,
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


def _call_gemini_google(
    *,
    prompt: str,
    frame_b64: Optional[str],
    frame_mime_type: str,
    model: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    timeout_s: float,
) -> str:
    image_bytes = base64.b64decode(frame_b64) if frame_b64 else None
    image_object = None
    if image_bytes is not None:
        try:
            from PIL import Image

            image_object = Image.open(BytesIO(image_bytes))
        except Exception:
            image_object = None

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        genai = None
        types = None

    if genai is not None:
        client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        contents: list[object] = []
        if system_prompt:
            contents.append(system_prompt)
        contents.append(prompt)
        if image_object is not None:
            contents.append(image_object)
        elif image_bytes is not None:
            contents.append(
                {
                    "mime_type": frame_mime_type,
                    "data": image_bytes,
                }
            )
        config = {}
        if types is not None:
            config = {
                "temperature": 0.0,
                "max_output_tokens": 512,
                "response_mime_type": "text/plain",
            }
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config) if types is not None else None,
        )
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()

    try:
        import google.generativeai as genai_legacy  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Google Gemini client is required for Gemini VLM calls") from exc

    genai_legacy.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
    model_client = genai_legacy.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
    )

    parts: list[object] = [prompt]
    if image_object is not None:
        parts.append(image_object)
    elif image_bytes is not None:
        parts.append(
            {
                "mime_type": frame_mime_type,
                "data": image_bytes,
            }
        )

    response = model_client.generate_content(parts, generation_config={"temperature": 0.0})
    text = getattr(response, "text", None)
    return str(text or "").strip()


def _data_uri(frame_b64: str, frame_mime_type: str) -> str:
    return f"data:{frame_mime_type};base64,{frame_b64}"


def _parse_backend(raw_value: str) -> MultimodalBackend:
    value = str(raw_value).strip().lower()
    for backend in MultimodalBackend:
        if backend.value == value:
            return backend
    return MultimodalBackend.OPENAI_CHAT_COMPLETIONS
