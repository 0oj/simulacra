# research/

FYP-era code and one-off experiments. Preserved verbatim so historical
runs can be reproduced; not maintained as part of the v0.1 library API.

The library itself (`src/simulacra/`) does not import from this
directory. Files here may reference paths or modules that have moved.

## Contents

- `scenarios/` — non-hopper scenarios (sounding rocket, pitch-over,
  tipover edge cases). Runnable via `uv run research/scenarios/<name>.py`
  as long as the simulacra package resolves.
- `backend_scenarios/` — older `backend/scenarios/` directory (demos
  and standalone PID tests).
- `docs/` — FYP report figures, literature review, session reports,
  ablation study writeups.
- `runs/` — historical training outputs (large; consider .gitignore
  for any future re-runs).
