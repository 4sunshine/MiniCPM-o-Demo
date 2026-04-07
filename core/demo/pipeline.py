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
    run_policy: bool = True,
) -> Tuple[SessionState, List[Event], List[Event]]:
    """
    Run the demo event pipeline:

        event -> reducer -> reasoning_policy -> dispatcher -> new events -> reducer

    Returns:
        (updated_state, policy_events, dispatched_events)
    """
    dispatcher = dispatcher or ActionDispatcher()

    state = apply_event(state, event)
    policy_events = handle_reasoning_for_event(state, event) if run_policy else []

    dispatched_events: List[Event] = []
    if run_policy:
        for policy_event in policy_events:
            dispatched_events.extend(dispatcher.dispatch(policy_event, state))

    if dispatched_events:
        state = apply_events(state, dispatched_events)

    return state, policy_events, dispatched_events


def process_demo_events(
    state: SessionState,
    events: List[Event],
    dispatcher: ActionDispatcher | None = None,
    run_policy: bool = True,
) -> Tuple[SessionState, List[Event], List[Event]]:
    """
    Process a batch of events in order.
    """
    dispatcher = dispatcher or ActionDispatcher()

    policy_events: List[Event] = []
    dispatched_events: List[Event] = []

    for event in events:
        state, event_policy_events, event_dispatched_events = process_demo_event(
            state,
            event,
            dispatcher=dispatcher,
            run_policy=run_policy,
        )
        policy_events.extend(event_policy_events)
        dispatched_events.extend(event_dispatched_events)

    return state, policy_events, dispatched_events
