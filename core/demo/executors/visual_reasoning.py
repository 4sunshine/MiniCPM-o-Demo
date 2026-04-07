from __future__ import annotations

from core.demo.annotation_models import Actor, DecisionType, Event, EventType, Source, Visibility
from core.demo.event_factory import make_decision, make_event, next_event_id
from core.demo.session_state import SessionState


def execute_visual_reasoning_action(event: Event, state: SessionState) -> Event:
    """
    Stub visual reasoning executor.

    This does not call Gemini or any ML backend. It emits a synthetic
    segment check event so the reducer can continue the event-driven flow.
    """
    reference_segment_id = event.context.reference_segment_id
    if reference_segment_id is None:
        reference_segment_id = event.context.segment_id if event.context.segment_id is not None else 0

    reference_event_id = event.context.reference_event_id or event.event_id

    return make_event(
        event_id=next_event_id(state),
        timestamp=event.timestamp,
        source=Source.REASONING_NODE,
        actor=Actor.ASSISTANT,
        agent_name="Visual Reasoning Executor",
        event_type=EventType.AGENT_ANALYSIS_SEGMENT_CHECKED,
        payload=event.payload.model_copy(update={
            "content_text": "Visual reasoning executed",
        }),
        context=event.context.model_copy(update={
            "reference_segment_id": reference_segment_id,
            "reference_event_id": reference_event_id,
        }),
        decision=make_decision(
            decision_type=DecisionType.CORRECT_SO_FAR,
            producer="Visual Reasoning Executor",
            visibility=Visibility.SILENT,
        ),
        meta=dict(event.meta),
    )
