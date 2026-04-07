from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.demo.annotation_models import Event, Modality


# =========================
# Segment lifecycle enums
# =========================

class SegmentStatus(str, Enum):
    """
    Lifecycle state of a user segment.
    """
    OPEN = "open"
    CLOSED = "closed"


class ReasoningStatus(str, Enum):
    """
    Latest reasoning interpretation of a completed or active segment.

    These values should be updated from reasoning-side decisions such as:
    - correct_so_far
    - incorrect
    - incomplete
    - task_completed_candidate
    """
    UNKNOWN = "unknown"
    CORRECT_SO_FAR = "correct_so_far"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    TASK_COMPLETED_CANDIDATE = "task_completed_candidate"


# =========================
# Segment record
# =========================

class SegmentRecord(BaseModel):
    """
    Derived state for a single student work segment.

    A segment is typically formed by:
    - user.writing.started  -> user.writing.ended
    - user.editing.started  -> user.editing.ended

    Notes:
    - reasoning_status is derived from reasoning-side analysis events
    - *_event_id fields allow back-references to important emitted events
    """
    model_config = ConfigDict(extra="allow")

    segment_id: int

    modality: Optional[Modality] = None
    kind: Optional[str] = None  # e.g. "writing", "editing", "audio"

    started_at: Optional[int] = None
    ended_at: Optional[int] = None

    content_text: str = ""
    target: Optional[str] = None
    state: Optional[str] = None

    status: SegmentStatus = SegmentStatus.OPEN
    reasoning_status: ReasoningStatus = ReasoningStatus.UNKNOWN

    last_event_id: Optional[str] = None
    check_event_id: Optional[str] = None
    mistake_event_id: Optional[str] = None
    feedback_event_id: Optional[str] = None


# =========================
# Tutor policy snapshot
# =========================

class TutorPolicy(BaseModel):
    """
    High-level policy snapshot for the session.

    This should stay lightweight and descriptive.
    Concrete event/decision routing belongs in reducer / agent logic.
    """
    model_config = ConfigDict(extra="allow")

    mistake_detection_agent: str = "Streaming Reasoning Agent"
    proactivity_role: str = "observe_and_alert"
    thinking_policy: str = "hint_on_prolonged_inactivity"


# =========================
# Session state
# =========================

class SessionState(BaseModel):
    """
    Derived runtime state for one replay/demo session.

    This model is the main persistent state used by:
    - replay/orchestration logic
    - reducer updates
    - LangGraph nodes

    Design notes:
    - events stores the normalized event history
    - segments stores derived segment-level state
    - latest_* fields provide fast access for orchestration and debugging
    """
    model_config = ConfigDict(extra="allow")

    # Identity
    session_id: Optional[str] = None
    task_id: Optional[str] = None

    # Event log
    events: List[Event] = Field(default_factory=list)

    # Activity tracking
    last_user_activity_ts: Optional[int] = None
    last_assistant_activity_ts: Optional[int] = None
    active_modality: Optional[Modality] = None

    # Segment tracking
    open_segment_id: Optional[int] = None
    segments: Dict[int, SegmentRecord] = Field(default_factory=dict)

    # Policy snapshot
    policy: TutorPolicy = Field(default_factory=TutorPolicy)

    # Fast-access outputs / recent state
    latest_feedback_text: Optional[str] = None
    latest_hint_text: Optional[str] = None
    latest_error_type: Optional[str] = None

    # Session completion
    task_completed: bool = False
