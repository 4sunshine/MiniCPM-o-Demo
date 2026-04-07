from __future__ import annotations

from typing import Any, Iterable, List

from core.demo.annotation_models import (
    Actor,
    Context,
    Decision,
    Event,
    EventType,
    Payload,
    Source,
)


def normalize_timeline_event(raw_event: dict[str, Any]) -> Event:
    """
    Convert a raw replay timeline entry into the shared demo Event schema.
    """
    payload = raw_event.get("payload")
    payload_data = payload if isinstance(payload, dict) else {}

    context = raw_event.get("context")
    context_data = context if isinstance(context, dict) else {}

    decision = raw_event.get("decision")
    decision_data = decision if isinstance(decision, dict) else None

    meta = raw_event.get("meta")
    meta_data = meta if isinstance(meta, dict) else {}

    return Event(
        event_id=str(raw_event.get("event_id", "")),
        timestamp=int(raw_event.get("timestamp", 0)),
        source=_parse_enum(Source, raw_event.get("source"), Source.UI_OBSERVER),
        actor=_parse_enum(Actor, raw_event.get("actor"), Actor.USER),
        agent_name=str(raw_event.get("agent_name", "")),
        event_type=_parse_event_type(raw_event.get("event_type", "")),
        payload=Payload(
            content_text=str(payload_data.get("content_text") or ""),
            feedback_text=payload_data.get("feedback_text"),
            content_ref=payload_data.get("content_ref"),
        ),
        context=Context(**context_data),
        decision=Decision(**decision_data) if decision_data is not None else None,
        meta=dict(meta_data),
    )


def normalize_timeline_events(raw_events: Iterable[dict[str, Any]]) -> List[Event]:
    """
    Normalize and sort replay timeline events by timestamp.
    """
    normalized = [normalize_timeline_event(event) for event in raw_events]
    return sorted(normalized, key=lambda event: event.timestamp)


def _parse_enum(enum_cls, raw_value: Any, default):
    if raw_value is None:
        return default

    try:
        return enum_cls(raw_value)
    except Exception:
        return default if raw_value == "" else raw_value


def _parse_event_type(raw_value: Any) -> EventType | str:
    try:
        return EventType(raw_value)
    except Exception:
        return str(raw_value)
