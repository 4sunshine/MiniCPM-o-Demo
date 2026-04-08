# SOLUTION_STATE

## Scope

Reviewed `core/demo/` and `worker_demo.py` against the demo-mode requirements in `AGENTS.md`, then refactored the replay path into dedicated demo modules.

## Current Demo Structure

### `core/demo/annotation_models.py`

Defines the shared event schema for the demo pipeline.

- `Actor`, `Source`, `Modality`, `Visibility`
- `EventType`
- `DecisionType`
- `Payload`, `Context`, `Decision`, `Event`

This is the canonical contract for replay and reasoning events.

### `core/demo/session_state.py`

Defines derived replay state.

- `SegmentRecord`
- `TutorPolicy`
- `SessionState`

This is the mutable session state used by reducer, policy, and replay orchestration.

### `core/demo/reducer.py`

Applies normalized events to `SessionState`.

- appends events to the event log
- updates activity timestamps
- tracks open/closed segments
- updates reasoning status from analysis and feedback events
- sets task completion flags

The reducer is deterministic and does not make policy decisions.

### `core/demo/event_factory.py`

Builds normalized demo events and decisions.

- reasoning events
- hint / correction feedback events
- inactivity observations
- task-completion events

### `core/demo/reasoning_policy.py`

Contains rule-based reasoning logic.

- completed user segments are evaluated
- inactivity can trigger hints
- incorrect work emits mistake and correction feedback events
- correct or completed work can emit verification or completion feedback

Current reasoning is deterministic and local, not API-backed yet.

### `core/demo/action_dispatcher.py`

Routes decisions to executors.

- `RUN_VISUAL_REASONING` -> visual reasoning executor
- `SEND_HINT` -> feedback executor
- `SEND_CORRECTION_FEEDBACK` -> feedback executor

### `core/demo/executors/visual_reasoning.py`

Stub visual reasoning executor.

- no ML backend call
- emits a synthetic `agent.analysis.segment_checked`

### `core/demo/executors/feedback.py`

Stub visible-feedback executor.

- emits `agent.feedback.sent`
- preserves feedback text from the input event when present

### `core/demo/timeline_normalizer.py`

New module for replay normalization.

- converts raw timeline dicts into normalized `Event` instances
- parses `source`, `actor`, `event_type`, `context`, `decision`, and `payload`
- sorts normalized replay events by timestamp

This is the right place for replay JSON -> schema conversion.

### `core/demo/pipeline.py`

Event orchestration layer.

Current flow:

`event -> reducer -> reasoning_policy -> dispatcher -> new events -> reducer`

New support:

- `process_demo_event(..., run_policy=...)`
- `process_demo_events(..., run_policy=...)`

For replay slices that already contain annotated reasoning events, `run_policy=False` avoids duplicating policy outputs.

### `core/demo/output_policy.py`

New module for learner-facing output shaping.

- detects visible feedback events
- renders visible text from agent outcomes
- builds the replay payload fields used by `worker_demo.py`

This keeps output shaping separate from schema conversion and state updates.

### `core/demo/__init__.py`

Exports the core demo helpers, including the new normalizer and output-policy helpers.

## `worker_demo.py`

`worker_demo.py` remains the transport/orchestration layer for the demo worker.

Key behavior:

- `load_model()` now normalizes the replay file into `Event` objects via `normalize_timeline_events(...)`
- `duplex_prefill()` collects due events by replay timestamp
- `_build_demo_payload()` now hands the due slice through `process_demo_events(..., run_policy=False)` and `build_demo_payload(...)`
- visible text comes from the new output-policy module instead of inline filtering

So the worker is now thinner and no longer owns the replay normalization logic.

## Architectural Gaps Remaining

### 1. Replay and live policy flow are still separate

The worker is using replay slices directly, which is correct for annotated demo playback. The full live event-generation path still needs a clean integration point if we want one unified orchestration layer.

### 2. `RUN_VISUAL_REASONING` is still unused in policy

The schema and dispatcher support it, but `reasoning_policy.py` does not emit it yet.

### 3. API-backed reasoning is still stubbed

The demo logic is still deterministic/rule-based. `AGENTS.md` allows external reasoning APIs later, but the current code path is still local and deterministic.

## Refactor Outcome

The replay path is now split into three responsibilities:

1. `timeline_normalizer.py` converts raw replay data into schema events.
2. `pipeline.py` updates state and optionally runs policy/dispatch logic.
3. `output_policy.py` renders learner-facing output from replay outcomes.

This matches the architectural boundaries requested in `AGENTS.md` and keeps `worker_demo.py` focused on timing and websocket orchestration.

