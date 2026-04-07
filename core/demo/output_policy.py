from __future__ import annotations

from typing import Iterable, List

from core.demo.annotation_models import DecisionType, Event, EventType, Visibility
from core.demo.session_state import SessionState


def is_visible_feedback_event(event: Event) -> bool:
    """
    Return True when an event should surface to the learner.
    """
    event_type = _enum_value(event.event_type)
    if event_type != EventType.AGENT_FEEDBACK_SENT.value:
        return False

    if event.decision is None:
        return bool(_feedback_text(event).strip())

    visibility = _enum_value(event.decision.visibility) if event.decision.visibility is not None else ""
    if visibility == Visibility.VISIBLE.value:
        return True

    return bool(_feedback_text(event).strip())


def render_visible_text(events: Iterable[Event]) -> str:
    """
    Render learner-facing text from visible feedback events.
    """
    texts = [
        _feedback_text(event)
        for event in events
        if is_visible_feedback_event(event) and _feedback_text(event).strip()
    ]
    return "\n".join(texts)


def build_demo_payload(
    state: SessionState,
    *,
    timestamp_ms: int,
    events: List[Event],
    pending_audio_samples: int = 0,
    pending_frame_count: int = 0,
) -> dict:
    """
    Build the replay payload returned by worker_demo.
    """
    feedback_events = [event for event in events if is_visible_feedback_event(event)]
    analysis_events = [
        event for event in events
        if _enum_value(event.event_type).startswith("agent.analysis.")
    ]
    user_events = [
        event for event in events
        if _enum_value(event.event_type).startswith("user.")
    ]

    return {
        "timestamp_ms": timestamp_ms,
        "user_events": [event.model_dump(mode="json") for event in user_events],
        "analysis_events": [event.model_dump(mode="json") for event in analysis_events],
        "feedback_events": [event.model_dump(mode="json") for event in feedback_events],
        "visible_text": render_visible_text(feedback_events),
        "is_listen": len(feedback_events) == 0,
        "pending_audio_samples": pending_audio_samples,
        "pending_frame_count": pending_frame_count,
        "latest_feedback_text": state.latest_feedback_text,
        "latest_hint_text": state.latest_hint_text,
        "latest_error_type": state.latest_error_type,
        "task_completed": state.task_completed,
    }


def _feedback_text(event: Event) -> str:
    return str(event.payload.feedback_text or event.payload.content_text or "")


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
