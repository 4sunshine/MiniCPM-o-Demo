from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import List, Optional

from core.demo.annotation_models import DecisionType, Event, EventType
from core.demo.event_factory import (
    build_correction_feedback_event,
    build_correction_verified_event,
    build_hint_feedback_event,
    build_mistake_detected_event,
    build_segment_checked_event,
    build_task_completed_event,
)
from core.demo.multimodal_llm import MultimodalLLMConfig, call_multimodal_llm
from core.demo.session_state import SessionState


# =========================
# Internal result container
# =========================

@dataclass
class SegmentCheckResult:
    """
    Internal normalized result of a reasoning-side silent check.
    """
    decision_type: DecisionType
    error_type: Optional[str] = None
    explanation: Optional[str] = None
    feedback_text: Optional[str] = None


# =========================
# Public API
# =========================

def handle_reasoning_for_event(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Main reasoning policy entrypoint.

    For a single incoming event, returns zero or more emitted reasoning-side events.

    Rules:
    - user.writing.ended / user.editing.ended:
        always emit agent.analysis.segment_checked
        optionally emit mistake_detected + visible correction feedback
        optionally emit correction verified feedback
        optionally emit task completed feedback
    - user.activity.inactivity:
        optionally emit visible hint feedback
    """
    event_type = _enum_value(event.event_type)

    if event_type in {
        EventType.USER_WRITING_ENDED.value,
        EventType.USER_EDITING_ENDED.value,
    }:
        return handle_completed_segment(state, event)

    if event_type == EventType.USER_ACTIVITY_INACTIVITY.value:
        return handle_inactivity(state, event)

    return []


def handle_completed_segment(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Run a silent reasoning check for a completed user segment.

    Output pattern:
    - always emit segment_checked
    - if incorrect:
        emit mistake_detected
        emit visible correction feedback
    - if correction_verified:
        emit visible correction_verified feedback
    - if task_completed_candidate:
        emit visible task_completed feedback
    """
    segment_id = event.context.segment_id
    if segment_id is None:
        return []

    content_text = event.payload.content_text
    result = evaluate_segment_text(state, event)

    emitted: List[Event] = []

    checked_event = build_segment_checked_event(
        state,
        timestamp=event.timestamp + 100,
        reference_segment_id=segment_id,
        reference_event_id=event.event_id,
        content_text=content_text,
        decision_type=result.decision_type,
        error_type=result.error_type,
        check_scope="latest_segment",
    )
    emitted.append(checked_event)

    if result.decision_type == DecisionType.INCORRECT:
        mistake_event = build_mistake_detected_event(
            state,
            timestamp=event.timestamp + 101,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            content_text=content_text,
            error_type=result.error_type or "unknown_error",
            explanation=result.explanation,
        )
        emitted.append(mistake_event)

        feedback_event = build_correction_feedback_event(
            state,
            timestamp=event.timestamp + 102,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            content_text=content_text,
            feedback_text=result.feedback_text or "There is a mistake here. Please check this step.",
            error_type=result.error_type,
        )
        emitted.append(feedback_event)

    elif result.decision_type == DecisionType.CORRECT_SO_FAR:
        if is_correction_of_previous_error(state, event):
            verified_event = build_correction_verified_event(
                state,
                timestamp=event.timestamp + 101,
                reference_segment_id=segment_id,
                reference_event_id=event.event_id,
                content_text=content_text,
                feedback_text=result.feedback_text or "This is correct now",
            )
            emitted.append(verified_event)

    elif result.decision_type == DecisionType.TASK_COMPLETED_CANDIDATE:
        completed_event = build_task_completed_event(
            state,
            timestamp=event.timestamp + 101,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            feedback_text=result.feedback_text or "You've successfully completed the task",
        )
        emitted.append(completed_event)

    return emitted


def handle_inactivity(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Build a visible hint in response to inactivity observation.

    Expected input:
    - event_type = user.activity.inactivity
    - usually emitted by the Proactivity Agent
    """
    reference_segment_id = event.context.reference_segment_id
    reference_event_id = event.context.reference_event_id
    content_text = event.payload.content_text

    hint_text = generate_hint_text(state, event)

    if not hint_text:
        return []

    return [
        build_hint_feedback_event(
            state,
            timestamp=event.timestamp + 1,
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
            content_text=content_text,
            feedback_text=hint_text,
        )
    ]


# =========================
# Segment evaluation
# =========================

def evaluate_segment_text(
    state: SessionState,
    event: Event,
) -> SegmentCheckResult:
    """
    Multimodal evaluator for a completed student segment.

    The preferred path is an external VLM call with a task-specific prompt and
    the latest frame from short-term memory. If that call fails or returns an
    unusable result, we fall back to deterministic local heuristics.
    """
    text = normalize_text(event.payload.content_text)
    frame = state.latest_frame()

    prompt = _build_segment_evaluation_prompt(
        state=state,
        event=event,
        normalized_text=text,
    )

    print("FRAME", frame)

    try:
        raw_response = call_multimodal_llm(
            prompt=prompt,
            frame_b64=frame.frame_b64 if frame else None,
            frame_mime_type=frame.mime_type if frame else "image/png",
            config=_build_segment_vlm_config(),
        )
        parsed = _parse_segment_check_result(raw_response)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    return _fallback_segment_check(text)


# =========================
# Hint generation
# =========================

def generate_hint_text(
    state: SessionState,
    inactivity_event: Event,
) -> Optional[str]:
    """
    Hint generation for inactivity.

    Uses the same multimodal helper as correctness checking, but with a
    separate prompt and a shorter learner-facing output contract.
    """
    frame = state.latest_frame()
    prompt = _build_inactivity_hint_prompt(state, inactivity_event)

    try:
        hint_text = call_multimodal_llm(
            prompt=prompt,
            frame_b64=frame.frame_b64 if frame else None,
            frame_mime_type=frame.mime_type if frame else "image/png",
            config=_build_hint_vlm_config(),
        )
        hint_text = hint_text.strip()
        if hint_text:
            return hint_text
    except Exception:
        pass

    text = normalize_text(inactivity_event.payload.content_text)

    if text == "x2=":
        return "You’re very close. Use the minus case: x=(−1−5)/2"

    if text.endswith("="):
        return "Try continuing from the expression you already wrote."

    if text:
        return "You are close. Check the previous step and continue carefully."

    return "Take a look at your last completed step and continue from there."


# =========================
# Utility / heuristics
# =========================

def is_correction_of_previous_error(
    state: SessionState,
    event: Event,
) -> bool:
    """
    Heuristic:
    return True if there was a recent incorrect segment before this event.

    This is useful for emitting a visible 'correction verified' feedback.
    """
    current_segment_id = event.context.segment_id
    if current_segment_id is None:
        return False

    for seg_id, record in state.segments.items():
        if seg_id == current_segment_id:
            continue
        if record.reasoning_status.value == "incorrect":
            return True
    return False


def normalize_text(text: str) -> str:
    """
    Lightweight normalization for rule-based matching.
    """
    return " ".join(text.strip().lower().split())


def _build_segment_vlm_config() -> MultimodalLLMConfig:
    return MultimodalLLMConfig.from_env(
        backend=None,
        model=_resolve_env_model(
            [
                "DEMO_SEGMENT_VLM_MODEL",
                "DEMO_VLM_MODEL",
            ],
            default="gpt-4o-mini",
        ),
        system_prompt="You are a tutoring correctness evaluator.",
    )


def _build_hint_vlm_config() -> MultimodalLLMConfig:
    return MultimodalLLMConfig.from_env(
        backend=None,
        model=_resolve_env_model(
            [
                "DEMO_HINT_VLM_MODEL",
                "DEMO_VLM_MODEL",
            ],
            default="gpt-4o-mini",
        ),
        system_prompt="You are a tutoring hint generator.",
    )


def _build_segment_evaluation_prompt(
    *,
    state: SessionState,
    event: Event,
    normalized_text: str,
) -> str:
    recent_frames = state.recent_frames[-5:]
    frame_summary = "\n".join(
        [
            f"- frame_index={frame.frame_index}, timestamp_ms={frame.timestamp_ms}, mime_type={frame.mime_type}"
            for frame in recent_frames
        ]
    ) or "- no recent frames available"

    return (
        "Evaluate the correctness of the student's latest completed step.\n"
        "Return ONLY valid JSON with keys:\n"
        '{ "decision_type": "correct_so_far|incorrect|incomplete|task_completed_candidate", '
        '"error_type": string_or_null, '
        '"explanation": string_or_null, '
        '"feedback_text": string_or_null }\n\n'
        f"Student step: {event.payload.content_text}\n"
        f"Normalized step: {normalized_text}\n"
        f"Current segment id: {event.context.segment_id}\n"
        f"Recent frames:\n{frame_summary}\n"
        "Use the frame when needed, but judge the student's work primarily from the latest step and visible context.\n"
    )


def _build_inactivity_hint_prompt(state: SessionState, inactivity_event: Event) -> str:
    latest_frame = state.latest_frame()
    frame_summary = (
        f"timestamp_ms={latest_frame.timestamp_ms}, frame_index={latest_frame.frame_index}"
        if latest_frame is not None
        else "no recent frame available"
    )

    return (
        "Write one short, helpful hint for a student who has paused work.\n"
        "Return plain text only.\n\n"
        f"Observed text: {inactivity_event.payload.content_text}\n"
        f"Recent frame summary: {frame_summary}\n"
        "Keep the hint concise, encouraging, and specific to the last visible step."
    )


def _parse_segment_check_result(raw_response: str) -> Optional[SegmentCheckResult]:
    if not raw_response:
        return None

    candidate = _extract_json_object(raw_response)
    if not candidate:
        return None

    try:
        data = json.loads(candidate)
    except Exception:
        return None

    decision_type = _parse_decision_type(data.get("decision_type"))
    if decision_type is None:
        return None

    return SegmentCheckResult(
        decision_type=decision_type,
        error_type=_optional_str(data.get("error_type")),
        explanation=_optional_str(data.get("explanation")),
        feedback_text=_optional_str(data.get("feedback_text")),
    )


def _fallback_segment_check(text: str) -> SegmentCheckResult:
    print("FALLBACK_CHECKED")
    if text in {"x2=", "x1=", "d=", "sqrt(d)="} or text.endswith("="):
        return SegmentCheckResult(
            decision_type=DecisionType.INCOMPLETE,
        )

    if "1-24=-23" in text or "1 - 24 = -23" in text:
        return SegmentCheckResult(
            decision_type=DecisionType.INCORRECT,
            error_type="sign_error_in_discriminant",
            explanation="Because c=-6, the product 4ac is negative, so D=1-(-24)=25.",
            feedback_text=(
                "You made a mistake in the sign. Notice that c=−6, "
                "so 4ac=4⋅1⋅(−6)=−24. That means D=1−(−24), not 1−24. "
                "Try correcting this step."
            ),
        )

    if text in {
        "d=1-(-24)=25",
        "d = 1 - (-24) = 25",
    }:
        return SegmentCheckResult(
            decision_type=DecisionType.CORRECT_SO_FAR,
            feedback_text="This is correct now",
        )

    if text in {
        "final answer/check completed",
        "task completed",
        "done",
    }:
        return SegmentCheckResult(
            decision_type=DecisionType.TASK_COMPLETED_CANDIDATE,
            feedback_text="You've successfully completed the task",
        )

    return SegmentCheckResult(
        decision_type=DecisionType.CORRECT_SO_FAR,
    )


def _extract_json_object(raw_response: str) -> Optional[str]:
    text = raw_response.strip()
    if not text:
        return None

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def _parse_decision_type(raw_value: object) -> Optional[DecisionType]:
    if raw_value is None:
        return None

    value = str(raw_value).strip().lower()
    for decision in DecisionType:
        if decision.value == value:
            return decision
    return None


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_env_model(keys: list[str], *, default: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
