from __future__ import annotations

from core.demo.annotation_models import Actor, DecisionType, Event, EventType, Source, Visibility
from core.demo.event_factory import make_decision, make_event, next_event_id
from core.demo.session_state import SessionState


def execute_feedback_action(event: Event, state: SessionState) -> Event:
    """
    Stub feedback executor.

    Routes SEND_HINT and SEND_CORRECTION_FEEDBACK decisions into a
    visible agent.feedback.sent event. The text comes from the input event
    when available; otherwise a small stub message is generated.
    """
    decision = event.decision
    decision_type = decision.decision_type if decision else None
    decision_value = getattr(decision_type, "value", decision_type)

    if decision_value == DecisionType.SEND_HINT.value:
        feedback_text = event.payload.feedback_text or event.payload.content_text or "[STUB] hint executed"
    elif decision_value == DecisionType.SEND_CORRECTION_FEEDBACK.value:
        feedback_text = event.payload.feedback_text or event.payload.content_text or "[STUB] correction feedback executed"
    else:
        feedback_text = event.payload.feedback_text or event.payload.content_text or "[STUB] feedback executed"

    return make_event(
        event_id=next_event_id(state),
        timestamp=event.timestamp,
        source=Source.REASONING_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Feedback Executor",
        event_type=EventType.AGENT_FEEDBACK_SENT,
        payload=event.payload.model_copy(update={
            "content_text": event.payload.content_text,
            "feedback_text": feedback_text,
        }),
        context=event.context.model_copy(),
        decision=make_decision(
            decision_type=decision_type or DecisionType.SEND_HINT,
            producer="Feedback Executor",
            visibility=Visibility.VISIBLE,
            trigger=decision.trigger if decision else None,
            error_type=decision.error_type if decision else None,
        ),
        meta=dict(event.meta),
    )
