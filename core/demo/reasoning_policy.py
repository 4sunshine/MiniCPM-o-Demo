from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.demo.annotation_models import DecisionType, Event, EventType
from core.demo.event_factory import (
    build_correction_feedback_event,
    build_correction_verified_event,
    build_hint_feedback_event,
    build_mistake_detected_event,
    build_segment_checked_event,
    build_task_completed_event,
)
from core.demo.session_state import SessionState


# =========================
# Internal result container
# =========================

@dataclass
class SegmentCheckResult:
    """
    Internal normalized result of a reasoning-side silent check.
    """
    decision_type: DecisionType
    error_type: Optional[str] = None
    explanation: Optional[str] = None
    feedback_text: Optional[str] = None


# =========================
# Public API
# =========================

def handle_reasoning_for_event(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Main reasoning policy entrypoint.

    For a single incoming event, returns zero or more emitted reasoning-side events.

    Rules:
    - user.writing.ended / user.editing.ended:
        always emit agent.analysis.segment_checked
        optionally emit mistake_detected + visible correction feedback
        optionally emit correction verified feedback
        optionally emit task completed feedback
    - user.activity.inactivity:
        optionally emit visible hint feedback
    """
    event_type = _enum_value(event.event_type)

    if event_type in {
        EventType.USER_WRITING_ENDED.value,
        EventType.USER_EDITING_ENDED.value,
    }:
        return handle_completed_segment(state, event)

    if event_type == EventType.USER_ACTIVITY_INACTIVITY.value:
        return handle_inactivity(state, event)

    return []


def handle_completed_segment(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Run a silent reasoning check for a completed user segment.

    Output pattern:
    - always emit segment_checked
    - if incorrect:
        emit mistake_detected
        emit visible correction feedback
    - if correction_verified:
        emit visible correction_verified feedback
    - if task_completed_candidate:
        emit visible task_completed feedback
    """
    segment_id = event.context.segment_id
    if segment_id is None:
        return []

    content_text = event.payload.content_text
    result = evaluate_segment_text(state, event)

    emitted: List[Event] = []

    checked_event = build_segment_checked_event(
        state,
        timestamp=event.timestamp + 100,
        reference_segment_id=segment_id,
        reference_event_id=event.event_id,
        content_text=content_text,
        decision_type=result.decision_type,
        error_type=result.error_type,
        check_scope="latest_segment",
    )
    emitted.append(checked_event)

    if result.decision_type == DecisionType.INCORRECT:
        mistake_event = build_mistake_detected_event(
            state,
            timestamp=event.timestamp + 101,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            content_text=content_text,
            error_type=result.error_type or "unknown_error",
            explanation=result.explanation,
        )
        emitted.append(mistake_event)

        feedback_event = build_correction_feedback_event(
            state,
            timestamp=event.timestamp + 102,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            content_text=content_text,
            feedback_text=result.feedback_text or "There is a mistake here. Please check this step.",
            error_type=result.error_type,
        )
        emitted.append(feedback_event)

    elif result.decision_type == DecisionType.CORRECT_SO_FAR:
        if is_correction_of_previous_error(state, event):
            verified_event = build_correction_verified_event(
                state,
                timestamp=event.timestamp + 101,
                reference_segment_id=segment_id,
                reference_event_id=event.event_id,
                content_text=content_text,
                feedback_text=result.feedback_text or "This is correct now",
            )
            emitted.append(verified_event)

    elif result.decision_type == DecisionType.TASK_COMPLETED_CANDIDATE:
        completed_event = build_task_completed_event(
            state,
            timestamp=event.timestamp + 101,
            reference_segment_id=segment_id,
            reference_event_id=event.event_id,
            feedback_text=result.feedback_text or "You've successfully completed the task",
        )
        emitted.append(completed_event)

    return emitted


def handle_inactivity(
    state: SessionState,
    event: Event,
) -> List[Event]:
    """
    Build a visible hint in response to inactivity observation.

    Expected input:
    - event_type = user.activity.inactivity
    - usually emitted by the Proactivity Agent
    """
    reference_segment_id = event.context.reference_segment_id
    reference_event_id = event.context.reference_event_id
    content_text = event.payload.content_text

    hint_text = generate_hint_text(state, event)

    if not hint_text:
        return []

    return [
        build_hint_feedback_event(
            state,
            timestamp=event.timestamp + 1,
            reference_segment_id=reference_segment_id,
            reference_event_id=reference_event_id,
            content_text=content_text,
            feedback_text=hint_text,
        )
    ]


# =========================
# Segment evaluation
# =========================

def evaluate_segment_text(
    state: SessionState,
    event: Event,
) -> SegmentCheckResult:
    """
    Baseline rule-based evaluator.

    This is intentionally simple and deterministic.
    Later you can replace this function with:
    - ChatGPT API
    - Gemini API
    - hybrid symbolic + LLM logic

    Current behavior:
    - specific quadratic-demo rules
    - correction verification heuristic
    - incomplete line heuristic
    - task completion heuristic
    - fallback: correct_so_far
    """
    text = normalize_text(event.payload.content_text)

    # Incomplete line
    if text in {"x2=", "x1=", "d=", "sqrt(d)="} or text.endswith("="):
        return SegmentCheckResult(
            decision_type=DecisionType.INCOMPLETE,
        )

    # Known sign mistake example
    if "1-24=-23" in text or "1 - 24 = -23" in text:
        return SegmentCheckResult(
            decision_type=DecisionType.INCORRECT,
            error_type="sign_error_in_discriminant",
            explanation="Because c=-6, the product 4ac is negative, so D=1-(-24)=25.",
            feedback_text=(
                "You made a mistake in the sign. Notice that c=−6, "
                "so 4ac=4⋅1⋅(−6)=−24. That means D=1−(−24), not 1−24. "
                "Try correcting this step."
            ),
        )

    # Known corrected line
    if text in {
        "d=1-(-24)=25",
        "d = 1 - (-24) = 25",
    }:
        return SegmentCheckResult(
            decision_type=DecisionType.CORRECT_SO_FAR,
            feedback_text="This is correct now",
        )

    # Generic task completion markers
    if text in {
        "final answer/check completed",
        "task completed",
        "done",
    }:
        return SegmentCheckResult(
            decision_type=DecisionType.TASK_COMPLETED_CANDIDATE,
            feedback_text="You've successfully completed the task",
        )

    # Default baseline
    return SegmentCheckResult(
        decision_type=DecisionType.CORRECT_SO_FAR,
    )


# =========================
# Hint generation
# =========================

def generate_hint_text(
    state: SessionState,
    inactivity_event: Event,
) -> Optional[str]:
    """
    Baseline rule-based hint generation.

    Later this can be replaced by an external reasoning backend.
    """
    text = normalize_text(inactivity_event.payload.content_text)

    if text == "x2=":
        return "You’re very close. Use the minus case: x=(−1−5)/2"

    if text.endswith("="):
        return "Try continuing from the expression you already wrote."

    if text:
        return "You are close. Check the previous step and continue carefully."

    return "Take a look at your last completed step and continue from there."


# =========================
# Utility / heuristics
# =========================

def is_correction_of_previous_error(
    state: SessionState,
    event: Event,
) -> bool:
    """
    Heuristic:
    return True if there was a recent incorrect segment before this event.

    This is useful for emitting a visible 'correction verified' feedback.
    """
    current_segment_id = event.context.segment_id
    if current_segment_id is None:
        return False

    for seg_id, record in state.segments.items():
        if seg_id == current_segment_id:
            continue
        if record.reasoning_status.value == "incorrect":
            return True
    return False


def normalize_text(text: str) -> str:
    """
    Lightweight normalization for rule-based matching.
    """
    return " ".join(text.strip().lower().split())


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))
