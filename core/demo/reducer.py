from __future__ import annotations

from typing import Optional

from core.demo.annotation_models import (
    Actor,
    DecisionType,
    Event,
    EventType,
)
from core.demo.session_state import (
    ReasoningStatus,
    SegmentRecord,
    SegmentStatus,
    SessionState,
)


# =========================
# Public API
# =========================

def apply_event(state: SessionState, event: Event) -> SessionState:
    """
    Apply one normalized event to SessionState.

    Rules:
    - Event log is append-only
    - Derived state is updated deterministically from the event
    - User events drive segment lifecycle
    - Reasoning events update segment reasoning state
    - Feedback events update latest visible outputs
    """
    state.events.append(event)

    _update_last_activity(state, event)
    _update_active_modality(state, event)

    event_type = _enum_value(event.event_type)

    if event_type in {
        EventType.USER_AUDIO_STARTED.value,
        EventType.USER_WRITING_STARTED.value,
        EventType.USER_EDITING_STARTED.value,
    }:
        _handle_segment_started(state, event)

    elif event_type in {
        EventType.USER_AUDIO_ENDED.value,
        EventType.USER_WRITING_ENDED.value,
        EventType.USER_EDITING_ENDED.value,
    }:
        _handle_segment_ended(state, event)

    elif event_type == EventType.AGENT_ANALYSIS_SEGMENT_CHECKED.value:
        _handle_segment_checked(state, event)

    elif event_type == EventType.AGENT_ANALYSIS_MISTAKE_DETECTED.value:
        _handle_mistake_detected(state, event)

    elif event_type == EventType.AGENT_FEEDBACK_SENT.value:
        _handle_feedback_sent(state, event)

    elif event_type == EventType.SYSTEM_TASK_COMPLETED.value:
        state.task_completed = True

    return state


def apply_events(state: SessionState, events: list[Event]) -> SessionState:
    """
    Apply a list of events in order.
    """
    for event in events:
        state = apply_event(state, event)
    return state


# =========================
# Top-level state updates
# =========================

def _update_last_activity(state: SessionState, event: Event) -> None:
    if event.actor == Actor.USER:
        state.last_user_activity_ts = event.timestamp
    elif event.actor == Actor.ASSISTANT:
        state.last_assistant_activity_ts = event.timestamp


def _update_active_modality(state: SessionState, event: Event) -> None:
    if event.context.modality is not None:
        state.active_modality = event.context.modality


# =========================
# Segment lifecycle
# =========================

def _handle_segment_started(state: SessionState, event: Event) -> None:
    segment_id = event.context.segment_id
    if segment_id is None:
        return

    record = state.segments.get(segment_id)
    if record is None:
        record = SegmentRecord(segment_id=segment_id)
        state.segments[segment_id] = record

    record.modality = event.context.modality or record.modality
    record.kind = _infer_segment_kind(str(event.event_type))
    record.started_at = event.timestamp
    record.target = event.context.target or record.target
    record.state = event.context.state or record.state
    record.status = SegmentStatus.OPEN
    record.last_event_id = event.event_id

    state.open_segment_id = segment_id


def _handle_segment_ended(state: SessionState, event: Event) -> None:
    segment_id = event.context.segment_id
    if segment_id is None:
        return

    record = state.segments.get(segment_id)
    if record is None:
        record = SegmentRecord(
            segment_id=segment_id,
            kind=_infer_segment_kind(str(event.event_type)),
        )
        state.segments[segment_id] = record

    record.modality = event.context.modality or record.modality
    record.kind = record.kind or _infer_segment_kind(str(event.event_type))
    record.ended_at = event.timestamp
    record.content_text = event.payload.content_text
    record.target = event.context.target or record.target
    record.state = event.context.state or record.state
    record.status = SegmentStatus.CLOSED
    record.last_event_id = event.event_id

    if state.open_segment_id == segment_id:
        state.open_segment_id = None


def _infer_segment_kind(event_type: str) -> str:
    if event_type.startswith("user.writing."):
        return "writing"
    if event_type.startswith("user.editing."):
        return "editing"
    if event_type.startswith("user.audio."):
        return "audio"
    return "unknown"


# =========================
# Reasoning-side events
# =========================

def _handle_segment_checked(state: SessionState, event: Event) -> None:
    segment_id = _resolve_reference_segment_id(event)
    if segment_id is None:
        return

    record = _get_or_create_segment(state, segment_id)
    record.check_event_id = event.event_id
    record.last_event_id = event.event_id

    decision = event.decision
    if decision is None:
        return

    decision_type = _enum_value(decision.decision_type)

    if decision_type == DecisionType.CORRECT_SO_FAR.value:
        record.reasoning_status = ReasoningStatus.CORRECT_SO_FAR

    elif decision_type == DecisionType.INCORRECT.value:
        record.reasoning_status = ReasoningStatus.INCORRECT
        state.latest_error_type = decision.error_type

    elif decision_type == DecisionType.INCOMPLETE.value:
        record.reasoning_status = ReasoningStatus.INCOMPLETE

    elif decision_type == DecisionType.TASK_COMPLETED_CANDIDATE.value:
        record.reasoning_status = ReasoningStatus.TASK_COMPLETED_CANDIDATE


def _handle_mistake_detected(state: SessionState, event: Event) -> None:
    segment_id = _resolve_reference_segment_id(event)
    if segment_id is None:
        return

    record = _get_or_create_segment(state, segment_id)
    record.mistake_event_id = event.event_id
    record.last_event_id = event.event_id
    record.reasoning_status = ReasoningStatus.INCORRECT
    state.pending_correction_segment_id = segment_id

    if event.decision and event.decision.error_type:
        state.latest_error_type = event.decision.error_type


def _handle_feedback_sent(state: SessionState, event: Event) -> None:
    segment_id = _resolve_reference_segment_id(event, allow_direct_segment=True)
    if segment_id is not None:
        record = _get_or_create_segment(state, segment_id)
        record.feedback_event_id = event.event_id
        record.last_event_id = event.event_id

    feedback_text = event.payload.feedback_text
    generated_by_policy = bool(event.meta.get("generated_by_policy"))

    if generated_by_policy and feedback_text:
        state.latest_feedback_text = feedback_text

    decision = event.decision
    if decision is None:
        return

    decision_type = _enum_value(decision.decision_type)

    if decision_type == DecisionType.SEND_HINT.value:
        if generated_by_policy:
            state.latest_hint_text = feedback_text

    elif decision_type == DecisionType.CORRECTION_VERIFIED.value:
        if generated_by_policy and segment_id is not None:
            record = _get_or_create_segment(state, segment_id)
            record.reasoning_status = ReasoningStatus.CORRECT_SO_FAR
            state.pending_correction_segment_id = None

    elif decision_type == DecisionType.TASK_COMPLETED.value:
        if generated_by_policy:
            state.task_completed = True
            state.pending_correction_segment_id = None


# =========================
# Helpers
# =========================

def _resolve_reference_segment_id(
    event: Event,
    *,
    allow_direct_segment: bool = False,
) -> Optional[int]:
    """
    For reasoning-side events, prefer reference_segment_id.
    For some feedback events, segment_id may be used directly.
    """
    if event.context.reference_segment_id is not None:
        return event.context.reference_segment_id

    if allow_direct_segment and event.context.segment_id is not None:
        return event.context.segment_id

    return None


def _get_or_create_segment(state: SessionState, segment_id: int) -> SegmentRecord:
    record = state.segments.get(segment_id)
    if record is None:
        record = SegmentRecord(segment_id=segment_id)
        state.segments[segment_id] = record
    return record


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
