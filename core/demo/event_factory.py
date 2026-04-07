from __future__ import annotations

from typing import Optional

from core.demo.annotation_models import (
    Actor,
    Decision,
    DecisionType,
    Event,
    EventType,
    Payload,
    Context,
    Source,
    Visibility,
)
from core.demo.session_state import SessionState


# =========================
# Generic helpers
# =========================

def next_event_id(state: SessionState) -> str:
    """
    Generate a simple sequential event id based on current event log length.

    Example:
        evt_0001
        evt_0002
    """
    return f"evt_{len(state.events) + 1:04d}"


def make_event(
    *,
    event_id: str,
    timestamp: int,
    source: Source | str,
    actor: Actor,
    agent_name: str,
    event_type: EventType | str,
    payload: Optional[Payload] = None,
    context: Optional[Context] = None,
    decision: Optional[Decision] = None,
    meta: Optional[dict] = None,
) -> Event:
    """
    Low-level generic event constructor.
    """
    return Event(
        event_id=event_id,
        timestamp=timestamp,
        source=source,
        actor=actor,
        agent_name=agent_name,
        event_type=event_type,
        payload=payload or Payload(),
        context=context or Context(),
        decision=decision,
        meta=meta or {},
    )


# =========================
# Common decision builders
# =========================

def make_decision(
    *,
    decision_type: DecisionType,
    producer: str,
    visibility: Optional[Visibility] = None,
    trigger: Optional[str] = None,
    error_type: Optional[str] = None,
) -> Decision:
    return Decision(
        decision_type=decision_type,
        producer=producer,
        visibility=visibility,
        trigger=trigger,
        error_type=error_type,
    )


# =========================
# Reasoning-side event builders
# =========================

def build_segment_checked_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: int,
    reference_event_id: str,
    content_text: str,
    decision_type: DecisionType,
    error_type: Optional[str] = None,
    check_scope: str = "latest_segment",
) -> Event:
    """
    Build a silent reasoning check result.

    Typical decision_type values:
    - CORRECT_SO_FAR
    - INCORRECT
    - INCOMPLETE
    - TASK_COMPLETED_CANDIDATE
    """
    return make_event(
        event_id=next_event_id(state),
        timestamp=timestamp,
        source=Source.REASONING_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Streaming Reasoning Agent",
        event_type=EventType.AGENT_ANALYSIS_SEGMENT_CHECKED,
        payload=Payload(content_text=content_text),
        context=Context(
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
        ),
        decision=make_decision(
            decision_type=decision_type,
            producer="Streaming Reasoning Agent",
            visibility=Visibility.SILENT,
            error_type=error_type,
        ),
        meta={"check_scope": check_scope},
    )


def build_mistake_detected_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: int,
    reference_event_id: str,
    content_text: str,
    error_type: str,
    explanation: Optional[str] = None,
) -> Event:
    """
    Build a reasoning-side mistake_detected event.

    This is still silent/internal.
    Visible learner-facing text should be emitted separately via agent.feedback.sent.
    """
    meta = {}
    if explanation:
        meta["explanation"] = explanation

    return make_event(
        event_id=next_event_id(state),
        timestamp=timestamp,
        source=Source.REASONING_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Streaming Reasoning Agent",
        event_type=EventType.AGENT_ANALYSIS_MISTAKE_DETECTED,
        payload=Payload(content_text=content_text),
        context=Context(
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
        ),
        decision=make_decision(
            decision_type=DecisionType.MISTAKE_DETECTED,
            producer="Streaming Reasoning Agent",
            visibility=Visibility.SILENT,
            error_type=error_type,
        ),
        meta=meta,
    )


def build_feedback_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: Optional[int],
    reference_event_id: Optional[str],
    feedback_text: str,
    content_text: str = "",
    decision_type: DecisionType,
    trigger: Optional[str] = None,
    error_type: Optional[str] = None,
) -> Event:
    """
    Build a visible assistant feedback event.

    Typical decision_type values:
    - SEND_CORRECTION_FEEDBACK
    - SEND_HINT
    - CORRECTION_VERIFIED
    - TASK_COMPLETED
    - TASK_POLICY_ACKNOWLEDGED
    """
    return make_event(
        event_id=next_event_id(state),
        timestamp=timestamp,
        source=Source.REASONING_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Streaming Reasoning Agent",
        event_type=EventType.AGENT_FEEDBACK_SENT,
        payload=Payload(
            content_text=content_text,
            feedback_text=feedback_text,
        ),
        context=Context(
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
        ),
        decision=make_decision(
            decision_type=decision_type,
            producer="Streaming Reasoning Agent",
            visibility=Visibility.VISIBLE,
            trigger=trigger,
            error_type=error_type,
        ),
        meta={},
    )


# =========================
# Proactivity-side event builders
# =========================

def build_inactivity_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: Optional[int],
    reference_event_id: Optional[str],
    content_text: str = "",
    inactivity_ms: Optional[int] = None,
    alert_for_reasoning_agent: bool = True,
) -> Event:
    """
    Build a proactivity-side inactivity observation.

    This describes learner inactivity.
    It does not itself perform correctness checking.
    """
    meta = {
        "alert_for_reasoning_agent": alert_for_reasoning_agent,
    }
    if inactivity_ms is not None:
        meta["inactivity_ms"] = inactivity_ms

    return make_event(
        event_id=next_event_id(state),
        timestamp=timestamp,
        source=Source.PROACTIVITY_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Streaming Proactivity Agent",
        event_type=EventType.USER_ACTIVITY_INACTIVITY,
        payload=Payload(content_text=content_text),
        context=Context(
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
        ),
        decision=make_decision(
            decision_type=DecisionType.SUGGEST_HINT,
            producer="Streaming Proactivity Agent",
            visibility=Visibility.SILENT,
        ),
        meta=meta,
    )


# =========================
# Convenience wrappers
# =========================

def build_correction_feedback_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: int,
    reference_event_id: str,
    content_text: str,
    feedback_text: str,
    error_type: Optional[str] = None,
) -> Event:
    return build_feedback_event(
        state,
        timestamp=timestamp,
        reference_segment_id=reference_segment_id,
        reference_event_id=reference_event_id,
        content_text=content_text,
        feedback_text=feedback_text,
        decision_type=DecisionType.SEND_CORRECTION_FEEDBACK,
        trigger="mistake_detected",
        error_type=error_type,
    )


def build_hint_feedback_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: Optional[int],
    reference_event_id: Optional[str],
    content_text: str,
    feedback_text: str,
) -> Event:
    return build_feedback_event(
        state,
        timestamp=timestamp,
        reference_segment_id=reference_segment_id,
        reference_event_id=reference_event_id,
        content_text=content_text,
        feedback_text=feedback_text,
        decision_type=DecisionType.SEND_HINT,
        trigger="thinking_too_long",
    )


def build_correction_verified_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: int,
    reference_event_id: str,
    content_text: str,
    feedback_text: str = "This is correct now",
) -> Event:
    return build_feedback_event(
        state,
        timestamp=timestamp,
        reference_segment_id=reference_segment_id,
        reference_event_id=reference_event_id,
        content_text=content_text,
        feedback_text=feedback_text,
        decision_type=DecisionType.CORRECTION_VERIFIED,
        trigger="correction_verified",
    )


def build_task_completed_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_segment_id: Optional[int],
    reference_event_id: Optional[str],
    feedback_text: str = "You've successfully completed the task",
) -> Event:
    return build_feedback_event(
        state,
        timestamp=timestamp,
        reference_segment_id=reference_segment_id,
        reference_event_id=reference_event_id,
        content_text="",
        feedback_text=feedback_text,
        decision_type=DecisionType.TASK_COMPLETED,
        trigger="task_completed",
    )


def build_policy_acknowledged_event(
    state: SessionState,
    *,
    timestamp: int,
    reference_event_id: Optional[str],
    feedback_text: str = "Got it. Proceed.",
) -> Event:
    return build_feedback_event(
        state,
        timestamp=timestamp,
        reference_segment_id=None,
        reference_event_id=reference_event_id,
        content_text="",
        feedback_text=feedback_text,
        decision_type=DecisionType.TASK_POLICY_ACKNOWLEDGED,
        trigger="task_policy_acknowledged",
    )
