from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


# =========================
# Core origins / roles
# =========================

class Actor(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Source(str, Enum):
    UI_OBSERVER = "ui_observer"
    PROACTIVITY_NODE = "proactivity_node"
    REASONING_NODE = "reasoning_node"
    SYSTEM = "system"


class Modality(str, Enum):
    AUDIO = "audio"
    HANDWRITING = "handwriting"
    DRAWING = "drawing"
    TEXT = "text"
    MIXED = "mixed"


class Visibility(str, Enum):
    SILENT = "silent"
    VISIBLE = "visible"


# =========================
# Event taxonomy
# =========================

class EventType(str, Enum):
    """
    Event types are grouped semantically by what the event describes,
    not by who emitted it.

    Prefix conventions:
    - user.*       : observable learner activity
    - agent.*      : reasoning-side analysis or visible assistant output
    - system.*     : orchestration/session-level events
    """

    # ---------------------------------
    # User-observable activity events
    # ---------------------------------
    USER_AUDIO_STARTED = "user.audio.started"
    USER_AUDIO_ENDED = "user.audio.ended"

    USER_WRITING_STARTED = "user.writing.started"
    USER_WRITING_ENDED = "user.writing.ended"

    USER_EDITING_STARTED = "user.editing.started"
    USER_EDITING_ENDED = "user.editing.ended"

    USER_ACTIVITY_INACTIVITY = "user.activity.inactivity"

    # ---------------------------------
    # Reasoning-side internal analysis
    # ---------------------------------
    AGENT_ANALYSIS_SEGMENT_CHECKED = "agent.analysis.segment_checked"
    AGENT_ANALYSIS_MISTAKE_DETECTED = "agent.analysis.mistake_detected"

    # ---------------------------------
    # Visible assistant output
    # ---------------------------------
    AGENT_FEEDBACK_SENT = "agent.feedback.sent"

    # ---------------------------------
    # Optional system/session events
    # ---------------------------------
    SYSTEM_TASK_STARTED = "system.task.started"
    SYSTEM_TASK_COMPLETED = "system.task.completed"


# =========================
# Decision taxonomy
# =========================

class DecisionType(str, Enum):
    """
    Decision types are grouped semantically by decision function.

    These values are used inside the shared Decision object.
    Not every event has a decision.
    """

    # ---------------------------------
    # Reasoning evaluation outcomes
    # ---------------------------------
    CORRECT_SO_FAR = "correct_so_far"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    MISTAKE_DETECTED = "mistake_detected"
    CORRECTION_VERIFIED = "correction_verified"

    # ---------------------------------
    # Progress / completion outcomes
    # ---------------------------------
    TASK_COMPLETED_CANDIDATE = "task_completed_candidate"
    TASK_COMPLETED = "task_completed"

    # ---------------------------------
    # Proactivity outcomes
    # ---------------------------------
    SUGGEST_HINT = "suggest_hint"

    # ---------------------------------
    # Visible feedback actions
    # ---------------------------------
    SEND_CORRECTION_FEEDBACK = "send_correction_feedback"
    SEND_HINT = "send_hint"

    # ---------------------------------
    # Session/policy acknowledgements
    # ---------------------------------
    TASK_POLICY_ACKNOWLEDGED = "task_policy_acknowledged"

    # ---------------------------------
    # External reasoning / tool execution
    # ---------------------------------
    RUN_VISUAL_REASONING = "run_visual_reasoning"
    RUN_TEXT_REASONING = "run_text_reasoning"   # optional but useful


# =========================
# Shared event submodels
# =========================

class Payload(BaseModel):
    """
    Generic event payload.

    Use:
    - content_text   for user-produced text or observed text content
    - feedback_text  for learner-facing assistant output
    - content_ref    for optional external references (image/frame/etc.)

    extra='allow' permits future extension without breaking replay.
    """
    model_config = ConfigDict(extra="allow")

    content_text: str = ""
    feedback_text: Optional[str] = None
    content_ref: Optional[str] = None


class Context(BaseModel):
    """
    Contextual linking information for an event.

    This is intentionally generic and can be extended.
    """
    model_config = ConfigDict(extra="allow")

    modality: Optional[Modality] = None

    segment_id: Optional[int] = None
    reference_segment_id: Optional[int] = None
    reference_event_id: Optional[str] = None

    target: Optional[str] = None
    state: Optional[str] = None

    task_id: Optional[str] = None
    session_id: Optional[str] = None


class Decision(BaseModel):
    """
    Semantic interpretation or action decision associated with an event.
    """
    model_config = ConfigDict(extra="allow")

    decision_type: DecisionType
    producer: str
    visibility: Optional[Visibility] = None
    trigger: Optional[str] = None
    error_type: Optional[str] = None


class Event(BaseModel):
    """
    Shared event envelope for:
    - annotation events
    - internal emitted events
    - assistant outputs
    """
    model_config = ConfigDict(extra="allow")

    event_id: str
    timestamp: int = Field(..., ge=0, description="Milliseconds since replay start")

    source: Source | str
    actor: Actor

    agent_name: str
    event_type: EventType | str

    payload: Payload
    context: Context = Field(default_factory=Context)
    decision: Optional[Decision] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

