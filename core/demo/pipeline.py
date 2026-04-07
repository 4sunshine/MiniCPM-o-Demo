from __future__ import annotations

from typing import List, Tuple

from core.demo.action_dispatcher import ActionDispatcher
from core.demo.annotation_models import Event
from core.demo.reducer import apply_event, apply_events
from core.demo.reasoning_policy import handle_reasoning_for_event
from core.demo.session_state import SessionState


def process_demo_event(
    state: SessionState,
    event: Event,
    dispatcher: ActionDispatcher | None = None,
) -> Tuple[SessionState, List[Event], List[Event]]:
    """
    Run the demo event pipeline:

        event -> reducer -> reasoning_policy -> dispatcher -> new events -> reducer

    Returns:
        (updated_state, policy_events, dispatched_events)
    """
    dispatcher = dispatcher or ActionDispatcher()

    state = apply_event(state, event)
    policy_events = handle_reasoning_for_event(state, event)

    dispatched_events: List[Event] = []
    for policy_event in policy_events:
        dispatched_events.extend(dispatcher.dispatch(policy_event, state))

    if dispatched_events:
        state = apply_events(state, dispatched_events)

    return state, policy_events, dispatched_events
