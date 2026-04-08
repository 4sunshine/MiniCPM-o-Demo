# Agentic System Diagram

This is a conceptual map of the current demo replay architecture.

```text
Replay annotations / user events
        |
        v
+--------------------+
|   Observer layer   |
| (demo replay input)|
+--------------------+
        |
        v
+--------------------+
|   Pipeline /       |
|   Orchestrator     |
|  core/demo/pipeline|
+--------------------+
        |
        +------------------------------+
        |                              |
        v                              v
+--------------------+        +----------------------+
|       Memory       |        |   Planner / Critic   |
| SessionState +     |<-------| reasoning_policy.py  |
| reducer.py         |        +----------------------+
+--------------------+                 |
        ^                              |
        |                              v
        |                      +----------------------+
        |                      |     Dispatcher       |
        |                      | action_dispatcher.py |
        |                      +----------------------+
        |                              |
        |                              v
        |                      +----------------------+
        |                      |      Executor        |
        |                      | executors/*.py       |
        |                      +----------------------+
        |                              |
        +------------------------------+
                       |
                       v
              Updated session state
```

## Role Mapping

- `Observer`: replay annotations and the incoming demo event stream
- `Planner`: `core/demo/reasoning_policy.py`
- `Critic`: `core/demo/reasoning_policy.py`
- `Dispatcher`: `core/demo/action_dispatcher.py`
- `Executor`: `core/demo/executors/`
- `Memory`: `core/demo/session_state.py` + `core/demo/reducer.py`

## Reading the Flow

1. The replay stream produces annotated events.
2. `pipeline.py` applies the incoming event to state.
3. `reasoning_policy.py` decides whether the event should trigger analysis, hints, or feedback.
4. `action_dispatcher.py` turns those decisions into concrete emitted events.
5. `reducer.py` folds the emitted events back into `SessionState`.

## Important Nuance

In this codebase, the planner and critic are intentionally blended into one module.
That is common in small agent systems where the same policy both evaluates work and decides the next pedagogical action.
