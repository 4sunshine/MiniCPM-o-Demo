# AGENTS.md

## Project purpose

This repository contains a real-time multimodal assistant based on MiniCPM-O.

In addition to the live runtime, this repository supports a **demo replay mode** whose purpose is to emulate an educational assistant session using:

- a **pre-recorded video** instead of a live webcam stream
- a **pre-annotated event timeline** instead of real MiniCPM trigger generation
- external **Reasoning Agent responses** generated through API requests to ChatGPT or Gemini

The demo mode is intended to validate orchestration, event flow, agent behavior, and pedagogical timing without depending on live perception quality. However understanding of Video Content should be perfect.

---

## Demo mode: conceptual model

In demo mode, the system replays a prerecorded session.

### Inputs
Demo mode uses two main offline inputs:

1. **Pre-recorded video**
   - replaces the live webcam/video stream
   - frames are fed into the system according to replay time

2. **Annotation timeline**
   - replaces real MiniCPM-triggered proactive events
   - acts as the source of user activity and Proactive Agent observations

### Outputs
Demo mode produces:

- simulated assistant actions
- silent internal checks
- visible feedback to the learner
- optional logs or saved replay outputs

---

## Agent roles

### Streaming Proactivity Agent
The Streaming Proactivity Agent is an **observation and timing layer**.

Its role in demo mode is to consume annotation events that represent:
- speaking starts/stops
- writing starts/stops
- editing starts/stops
- inactivity or hesitation

In demo mode, proactive observations come from the annotation file rather than being generated live by MiniCPM perception.

The Streaming Proactivity Agent:
- observes activity timing
- detects inactivity / hesitation
- can request a hint pathway
- must not perform correctness checking

### Streaming Reasoning Agent
The Streaming Reasoning Agent is a **correctness and pedagogical feedback layer**.

Its role is to:
- silently evaluate completed user work segments
- detect mistakes
- generate correction feedback when needed
- generate hints when triggered by inactivity flow
- generate final pedagogical responses

In demo mode, Reasoning Agent outputs should be generated through API requests to external reasoning models such as ChatGPT or Gemini.

The Streaming Reasoning Agent:
- checks correctness after stable user work boundaries
- may emit silent internal analysis
- may emit visible learner-facing feedback
- must not be responsible for low-level timing observation

### Feedback policy layer
The feedback policy layer converts internal agent decisions into learner-facing messages.

It is responsible for tone, phrasing, and output style.

It must not own correctness logic or temporal observation logic.

---

## Architectural boundaries

The system should remain modular.

### Live path
The live runtime path is centered on the existing MiniCPM worker.

### Demo path
The demo path is centered on a dedicated replay entrypoint that:
- reads prerecorded video
- replays annotation events over time
- routes events through the tutoring logic
- emits simulated assistant outputs

### Separation of concerns
Keep the following boundaries strict:

- schema/model definitions must be separate from business logic
- event/state update logic must be separate from reasoning logic
- reasoning logic must be separate from orchestration wiring
- replay/video timing logic must be separate from pedagogical logic

Do not collapse these responsibilities into one large file.

---

## Core replay assumptions

Demo mode should follow these assumptions:

1. The prerecorded video replaces the webcam stream
2. Annotation events replace live proactive triggers
3. The replay clock is the source of truth for timing
4. User activity and assistant behavior are modeled as events
5. Completed user work should be checked at stable boundaries
6. External reasoning APIs may be used to generate assistant responses
7. The event log should remain inspectable and reproducible

---

## Design principles

### Determinism
Replay behavior should be reproducible from:
- a video file
- an annotation file
- a configuration

### Inspectability
Internal events, silent checks, and visible outputs should be observable in logs or saved outputs.

### Thin orchestration
Entry-point files and LangGraph wiring should remain thin.
Most decision logic should live in dedicated modules.

### Explicit contracts
Do not invent event semantics ad hoc.
If a new behavior is added, document it in the appropriate schema and documentation files.

---

## Coding guidance

When implementing demo mode:

- prefer pure functions where possible
- keep replay logic modular
- use typed models for structured data
- keep worker demo code thin
- avoid mixing production live logic with demo-specific replay logic
- make it easy to replace stubbed logic with real API-backed reasoning

---

## What belongs where

### AGENTS.md
This file defines:
- project purpose
- demo intent
- agent role boundaries
- architectural rules

### Documentation files
Documentation files should define:
- event semantics
- replay behavior
- annotation examples
- operational notes

### Schema/model files
Schema/model files should define:
- event structures
- session state structures
- graph state structures
- validation rules

### Policy/logic files
Policy/logic files should define:
- reasoning behavior
- proactivity behavior
- feedback generation
- routing conditions

---

## Important constraints

- Proactive annotations in demo mode are taken from the annotation file, not from live MiniCPM triggering
- Reasoning Agent responses in demo mode are generated through external reasoning APIs
- The prerecorded video is the visual source for replay instead of a live webcam
- Demo mode must be implemented as an addition to the current repository, not as a separate disconnected project

---

## Goal of this repository extension

The goal is to create a demo-capable educational assistant pipeline that can:

- replay a recorded student session
- simulate proactive observations from annotation
- generate reasoning feedback from external LLM APIs
- validate orchestration before full live integration

## Demo Time Semantics

In demo mode, `timestamp_ms` MUST be derived from replayed video timeline (frame index / fps or decoder timestamps), never from wall-clock time.

