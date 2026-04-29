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

    if not bool(event.meta.get("generated_by_policy")):
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
    latest_event = _latest_visible_feedback_event(events)
    if latest_event is None:
        return ""
    return _feedback_text(latest_event)


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
    latest_feedback_event = _latest_visible_feedback_event(feedback_events)
    analysis_events = [
        event for event in events
        if _enum_value(event.event_type).startswith("agent.analysis.")
    ]
    user_events = [
        event for event in events
        if _enum_value(event.event_type).startswith("user.")
    ]

    latest_feedback_text = _feedback_text(latest_feedback_event) if latest_feedback_event else None
    latest_hint_text = _latest_feedback_by_decision(feedback_events, DecisionType.SEND_HINT)

    return {
        "timestamp_ms": timestamp_ms,
        "user_events": [event.model_dump(mode="json") for event in user_events],
        "analysis_events": [event.model_dump(mode="json") for event in analysis_events],
        "feedback_events": [latest_feedback_event.model_dump(mode="json")] if latest_feedback_event else [],
        "visible_text": render_visible_text(feedback_events),
        "is_listen": latest_feedback_event is None,
        "pending_audio_samples": pending_audio_samples,
        "pending_frame_count": pending_frame_count,
        "latest_feedback_text": latest_feedback_text,
        "latest_hint_text": latest_hint_text,
        "latest_error_type": state.latest_error_type,
        "task_completed": state.task_completed,
    }


def _feedback_text(event: Event) -> str:
    return str(event.payload.feedback_text or event.payload.content_text or "")


def _latest_visible_feedback_event(events: Iterable[Event]) -> Event | None:
    latest_event: Event | None = None
    for event in events:
        if not is_visible_feedback_event(event):
            continue
        if _feedback_text(event).strip():
            latest_event = event
    return latest_event


def _latest_feedback_by_decision(events: List[Event], decision_type: DecisionType) -> str | None:
    for event in reversed(events):
        decision = event.decision
        if decision is None:
            continue
        if _enum_value(decision.decision_type) == decision_type.value:
            text = _feedback_text(event).strip()
            if text:
                return text
    return None


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
