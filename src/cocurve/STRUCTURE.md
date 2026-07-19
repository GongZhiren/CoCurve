# CoCurve Code Layout

The codebase keeps backward-compatible flat modules while exposing structured sub-packages:

- `cocurve.core`
  - model/unit registry
  - calibration/ablation
  - Fisher matrix and solver
  - runtime and physical pruning
- `cocurve.evaluation`
  - quality gates
  - prompting/chat-template adapter
  - task-aware scoring
- `cocurve.infra`
  - config loader/validator
  - artifact I/O
  - shared types/utilities

Entry point orchestration stays in `cocurve.pipeline`.
