from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, Optional

from core.demo.annotation_models import DecisionType, Event
from core.demo.executors.feedback import execute_feedback_action
from core.demo.executors.visual_reasoning import execute_visual_reasoning_action
from core.demo.session_state import SessionState


ExecutorFn = Callable[[Event, SessionState], Event]


class ActionDispatcher:
    """
    Execute decision-bearing events through the appropriate stub executor.

    The dispatcher is intentionally permissive:
    - known action decisions are executed by stubs
    - non-action decisions are passed through unchanged so reducer-only
      events can still flow back into the system
    """

    def __init__(self, executors: Optional[Dict[str, ExecutorFn]] = None):
        self._executors = executors or {
            "run_visual_reasoning": execute_visual_reasoning_action,
            DecisionType.SEND_HINT.value: execute_feedback_action,
            DecisionType.SEND_CORRECTION_FEEDBACK.value: execute_feedback_action,
        }

    def dispatch(self, event: Event, state: SessionState) -> list[Event]:
        decision = event.decision
        if decision is None:
            return [event]

        decision_key = _decision_key(decision.decision_type)
        executor = self._executors.get(decision_key)
        if executor is None:
            return [event]

        return [executor(event, state)]


def _decision_key(decision_type: object) -> str:
    if isinstance(decision_type, Enum):
        return str(decision_type.value)
    return str(getattr(decision_type, "value", decision_type))
