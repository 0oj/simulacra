# Simulacra v0.1 — Plan 1: Repository Restructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mechanically reshape the repo from `backend/`-subordinate (Python package buried two levels deep) to a top-level `src/simulacra/` library layout, with scenarios moved inside the package and FYP-era code archived to `research/`, **without changing a single line of behaviour**. All existing tests pass after the move; all existing scenarios still produce identical NDJSON output; the frontend renders unchanged.

**Architecture:** Pure file relocation. The Python package `simulacra` keeps the same module tree; only its location on disk changes. Imports in tests and scenarios already use absolute `simulacra.*` paths, so they continue to resolve. `pyproject.toml` already declares `packages = ["src/simulacra"]` relative to its own directory — moving both the package and the `pyproject.toml` together preserves that relationship. Research artefacts and FYP-specific code move to a top-level `research/` directory that the library itself does not import from.

**Tech Stack:** `git mv` (to preserve history), `uv` for Python environment, `pytest` for verification.

---

## Pre-flight: branch and isolation

This plan runs on the existing `feature/v0.1-sdk` branch. The user's uncommitted FYP files in the working tree are **left untouched** by every move below (only tracked files are moved by `git mv`; untracked files stay where they are). After Plan 1 lands, those uncommitted files may end up referencing paths that no longer exist — that's expected, and the user manages those separately.

No worktree is used. Reason: the user's flow has them switching between `feature/v0.1-sdk` and `feature/neat-waypoint-corridor` to continue FYP work; a worktree would invert their mental model.

---

## Target layout (end state after Plan 1)

```
simulacra/
├── pyproject.toml                  (moved from backend/)
├── uv.lock                         (moved from backend/)
├── README.md                       (unchanged on disk, content updated in Task 13)
├── ONBOARDING.md                   (unchanged)
├── CLAUDE.md                       (paths updated in Task 13)
├── CHANGELOG.md                    (unchanged)
├── AGENTS.md                       (unchanged)
├── .gitignore                      (extended to ignore root-level *.ndjson)
├── src/
│   └── simulacra/                  (moved from backend/src/simulacra/)
│       ├── noumena/
│       ├── phenomena/
│       ├── onboard/
│       ├── physics/
│       ├── scenarios/              (NEW — empty package marker only in Plan 1)
│       └── (existing top-level .py files unchanged)
├── tests/                          (moved from backend/tests/)
├── scripts/                        (moved from backend/scripts/)
├── examples/                       (moved from backend/examples/ — plotting/visual helpers)
├── frontend/                       (unchanged)
├── docs/                           (unchanged — already at root)
├── references/                     (unchanged)
└── research/                       (NEW)
    ├── scenarios/                  (moved from top-level scenarios/ minus hopper_*)
    │   ├── pitch_over.py
    │   ├── sounding_rocket.py
    │   ├── nose_heavy.py
    │   ├── engine_heavy.py
    │   └── tipover_test.py
    ├── backend_scenarios/          (moved from backend/scenarios/)
    │   ├── demos/
    │   ├── hover_tvc_test.py
    │   └── pid_hover_test.py
    ├── docs/                       (moved from backend/docs/)
    └── runs/                       (moved from backend/runs/)
```

After Plan 1, `src/simulacra/scenarios/` exists but is empty except for `__init__.py`. Plan 2 will populate it with `hopper_short.py` and `hopper_long.py` adapted to the new `Controller` interface. Until then, the top-level `scenarios/hopper_short.py` and `scenarios/hopper_long.py` continue to work via `uv run scenarios/hopper_short.py`.

---

## Task 1: Capture baseline state

**Files:**
- Capture only — no files touched.

**Rationale:** Before any move, freeze proof of "what currently passes" so we can verify nothing regresses.

- [ ] **Step 1.1: Confirm clean working tree state for moves**

Run:
```bash
cd /Users/ooj/Projects/Othisis/simulacra
git status --short | grep -v '^??' | grep -v '^ M backend/' | grep -v '^ M frontend/' | grep -v '^ M scenarios/' | head
```

Expected: empty (only untracked or in-progress FYP modifications are present, which is fine — they don't block moves).

- [ ] **Step 1.2: Capture current pytest result**

Run:
```bash
cd backend && uv run pytest -q --tb=no 2>&1 | tail -10 > /tmp/simulacra_baseline_pytest.txt
cat /tmp/simulacra_baseline_pytest.txt
```

Expected: a summary line like `N passed, M failed in X.XXs`. Note both the pass count and any failures — these are the baseline. We do **not** require the baseline to be 100% green; we require the post-move count to **match** the baseline.

- [ ] **Step 1.3: Capture current hopper_short NDJSON output**

Run:
```bash
cd /Users/ooj/Projects/Othisis/simulacra/backend
uv run ../scenarios/hopper_short.py > /tmp/simulacra_baseline_hopper_short.ndjson
wc -l /tmp/simulacra_baseline_hopper_short.ndjson
```

Expected: a non-empty NDJSON file (typically a few thousand lines). Note the line count.

- [ ] **Step 1.4: Commit checkpoint marker**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git commit --allow-empty -m "checkpoint: Plan 1 restructure begins"
```

Expected: empty commit succeeds. Note the commit SHA — this is your "if anything goes catastrophically wrong, reset here" anchor.

---

## Task 2: Move the Python package to top-level `src/`

**Files:**
- Move: `backend/src/simulacra/` → `src/simulacra/`

- [ ] **Step 2.1: Move the package**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
mkdir -p src
git mv backend/src/simulacra src/simulacra
git status --short | head -5
```

Expected: many `R  backend/src/simulacra/... -> src/simulacra/...` rename lines. No `??` lines for files inside the package (all should be tracked).

- [ ] **Step 2.2: Remove the now-empty `backend/src/` directory**

```bash
rmdir backend/src 2>/dev/null && echo "removed empty backend/src" || ls backend/src
```

Expected: "removed empty backend/src". If `ls` runs instead, something is still in there — investigate before continuing.

- [ ] **Step 2.3: Commit the package move**

```bash
git commit -m "$(cat <<'EOF'
restructure: move simulacra package from backend/src/ to src/

Pure relocation, no content changes. The package's import paths
(simulacra.*) are unchanged because they were always absolute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands.

---

## Task 3: Move tests to top-level

**Files:**
- Move: `backend/tests/` → `tests/`

- [ ] **Step 3.1: Move the tests directory**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/tests tests
ls tests/ | head -5
```

Expected: test files visible in top-level `tests/` (e.g. `test_kernel3d.py`, `test_quat.py`).

- [ ] **Step 3.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: move tests from backend/tests/ to tests/

Pure relocation. Tests use absolute simulacra.* imports so they
continue to resolve once the package is also at the top level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Move `pyproject.toml` and `uv.lock` to root

**Files:**
- Move: `backend/pyproject.toml` → `pyproject.toml`
- Move: `backend/uv.lock` → `uv.lock`

The existing `pyproject.toml` already says `packages = ["src/simulacra"]` (relative to its own location). Once it sits at the repo root next to `src/`, that path resolves correctly.

- [ ] **Step 4.1: Move pyproject and lockfile**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/pyproject.toml pyproject.toml
git mv backend/uv.lock uv.lock
ls -la pyproject.toml uv.lock
```

Expected: both files exist at root.

- [ ] **Step 4.2: Verify `uv sync` still works**

```bash
rm -rf .venv backend/.venv
uv sync 2>&1 | tail -5
```

Expected: `Resolved N packages` and `Installed N packages` lines, no errors. The venv now lives at `/Users/ooj/Projects/Othisis/simulacra/.venv` (root level).

- [ ] **Step 4.3: Run tests from the new layout**

```bash
uv run pytest -q --tb=no tests/ 2>&1 | tail -5
```

Expected: same pass/fail counts as `/tmp/simulacra_baseline_pytest.txt` (Task 1.2). If counts differ, **stop** and diagnose — something about the move broke a discoverable path.

- [ ] **Step 4.4: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: move pyproject.toml and uv.lock to repo root

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Move developer `scripts/`

**Files:**
- Move: `backend/scripts/` → `scripts/`

- [ ] **Step 5.1: Move scripts**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/scripts scripts
ls scripts/ | head -10
```

Expected: `analyze_flight.py`, `eval_hopper.py`, training scripts, etc.

- [ ] **Step 5.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: move backend/scripts/ to scripts/

Includes training scripts, eval helpers, and analysis utilities.
Some of these will move to research/training/ in a later plan, but
for now they stay accessible at the top level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Move plotting/visual `examples/`

**Files:**
- Move: `backend/examples/` → `examples/`

These are plotting helpers (`plot_gimbal.py`, `plot_telemetry.py`, `mass_visualizer.py`, `gimbal_notebook.ipynb`). Their purpose overlaps with the future `examples/lqr_skeleton.py` engineer-facing examples but distinct enough to keep separate sub-folders later. For Plan 1, just relocate as-is.

- [ ] **Step 6.1: Move examples**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/examples examples
ls examples/ | head -10
```

Expected: plotting scripts visible at `examples/plot_*.py`, etc.

- [ ] **Step 6.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: move backend/examples/ to examples/

These are existing plotting and visualization helpers. Engineer-facing
controller skeletons (lqr_skeleton.py, etc.) will be added in Plan 3
alongside them, organised by subdirectory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Create `src/simulacra/scenarios/` as empty package

**Files:**
- Create: `src/simulacra/scenarios/__init__.py`

This is the target home for `hopper_short.py` and `hopper_long.py` after Plan 2 adapts them. For Plan 1, we just create the empty package so the import path `simulacra.scenarios` resolves.

- [ ] **Step 7.1: Create the package**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
mkdir -p src/simulacra/scenarios
cat > src/simulacra/scenarios/__init__.py <<'EOF'
"""Canonical scenarios that ship with simulacra.

Each scenario module exposes a `run(controller, out, seed)` function so
engineers can use the scenario with their own controller. Scripts run
directly via `uv run -m simulacra.scenarios.<name>` execute with a default
controller declared in each scenario's __main__ block.

This package is empty in v0.1 until Plan 2 ports hopper_short and
hopper_long to the new Controller interface. The current top-level
scenarios/hopper_short.py and scenarios/hopper_long.py continue to work
under the old interface.
"""
EOF
git add src/simulacra/scenarios/__init__.py
```

- [ ] **Step 7.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: add empty simulacra.scenarios package

Placeholder for hopper_short and hopper_long to land in once Plan 2
adapts them to the new Controller interface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Create `research/` archive with FYP-era scenarios

**Files:**
- Create: `research/README.md`
- Move: `scenarios/pitch_over.py`, `scenarios/sounding_rocket.py`, `scenarios/nose_heavy.py`, `scenarios/engine_heavy.py`, `scenarios/tipover_test.py` → `research/scenarios/`

These are non-hopper scenarios from the FYP era. They stay runnable (paths inside them use `simulacra.*` absolute imports which work from anywhere) but are no longer first-class library scenarios.

- [ ] **Step 8.1: Create research dir with README**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
mkdir -p research/scenarios
cat > research/README.md <<'EOF'
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
EOF
git add research/README.md
```

- [ ] **Step 8.2: Move the non-hopper scenarios**

```bash
git mv scenarios/pitch_over.py research/scenarios/pitch_over.py
git mv scenarios/sounding_rocket.py research/scenarios/sounding_rocket.py
git mv scenarios/nose_heavy.py research/scenarios/nose_heavy.py
git mv scenarios/engine_heavy.py research/scenarios/engine_heavy.py
git mv scenarios/tipover_test.py research/scenarios/tipover_test.py
ls scenarios/
```

Expected: only `hopper_short.py` and `hopper_long.py` remain in top-level `scenarios/` (plus `__pycache__/` which we'll handle later).

- [ ] **Step 8.3: Verify a research scenario still runs**

```bash
uv run research/scenarios/sounding_rocket.py 2>&1 | head -3
```

Expected: NDJSON header line printed to stdout (a `{"kind":"metadata",...}` line). If it errors with an import error, **stop** and diagnose.

- [ ] **Step 8.4: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: archive non-hopper scenarios under research/scenarios/

Hopper scenarios stay at top-level scenarios/ until Plan 2 moves them
into the simulacra.scenarios package. Other FYP-era scenarios are
preserved here for reproducibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Move `backend/scenarios/` (demos + standalone tests) to `research/backend_scenarios/`

**Files:**
- Move: `backend/scenarios/` → `research/backend_scenarios/`

This is the older scenarios directory inside `backend/` (containing `demos/`, `hover_tvc_test.py`, `pid_hover_test.py`). Pre-dates the top-level `scenarios/` directory.

- [ ] **Step 9.1: Move it**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/scenarios research/backend_scenarios
ls research/backend_scenarios/
```

Expected: `demos/`, `hover_tvc_test.py`, `pid_hover_test.py`.

- [ ] **Step 9.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: archive backend/scenarios/ as research/backend_scenarios/

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Archive `backend/docs/`

**Files:**
- Move: `backend/docs/` → `research/docs/`

The existing `backend/docs/` contains FYP report figures, literature review, ablation study writeup, session reports. Distinct from the top-level `docs/` directory (which holds superpowers specs and plans).

- [ ] **Step 10.1: Move it**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/docs research/docs
ls research/docs/ | head -5
```

Expected: FYP figures, markdown files. The top-level `docs/` (superpowers content) is untouched.

- [ ] **Step 10.2: Commit**

```bash
git commit -m "$(cat <<'EOF'
restructure: archive backend/docs/ (FYP figures and writeups) under research/docs/

Distinct from top-level docs/ which holds library design specs and
implementation plans.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Archive `backend/runs/` and add `.gitignore` for future runs

**Files:**
- Move: `backend/runs/` → `research/runs/`
- Modify: `.gitignore` (add patterns for runtime outputs)

`backend/runs/` contains training outputs (checkpoints, tensorboard logs, eval CSVs). Large. Keep what's already committed for reproducibility, but make sure future runs don't pollute the new root layout.

- [ ] **Step 11.1: Move existing runs**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git mv backend/runs research/runs 2>/dev/null || echo "backend/runs already moved or empty"
ls research/runs/ 2>/dev/null | head -3
```

Expected: training run subdirectories, or "already moved or empty" if nothing was there.

- [ ] **Step 11.2: Extend `.gitignore`**

Read the current `.gitignore`:
```bash
cat .gitignore
```

Then append (using Edit tool, not echo):
```
# Runtime outputs — large NDJSON dumps, training runs, temp files
/*.ndjson
/runs/
/temp/
backend/
```

The `backend/` line is defensive: nothing should live there after Plan 1, but if a stale `__pycache__` reappears we don't want it tracked.

- [ ] **Step 11.3: Stage and commit**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
restructure: archive backend/runs/ under research/runs/, gitignore future outputs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Remove now-empty `backend/` directory and clean up root-level junk

**Files:**
- Remove: `backend/` (should be empty or contain only `__pycache__` / `.venv` / temp dirs)
- Remove from tracking: root-level `hopper_short.ndjson`, `hopper_long.ndjson`, `Wed-18-March-2026.pdf`, `Wed-18-March-2026.typ`

- [ ] **Step 12.1: Check what's left in `backend/`**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
ls -la backend/ 2>/dev/null
```

Expected: only `__pycache__/`, `.venv/`, or `temp/` (none of which should be tracked). If any tracked file remains, stop and decide whether to move it.

- [ ] **Step 12.2: Remove `backend/` from the working tree**

```bash
rm -rf backend/
ls backend/ 2>&1
```

Expected: "No such file or directory".

- [ ] **Step 12.3: Untrack root-level NDJSON dumps and stray PDFs**

```bash
git rm --cached hopper_short.ndjson hopper_long.ndjson 2>/dev/null
git rm --cached "Wed-18-March-2026.pdf" "Wed-18-March-2026.typ" 2>/dev/null
git status --short | grep -E 'D  |^M ' | head -5
```

Expected: deletions of those root-level files in the index. The files themselves remain on disk (untracked) so the user keeps their working copies; they just won't be in commits anymore.

- [ ] **Step 12.4: Commit cleanup**

```bash
git commit -m "$(cat <<'EOF'
restructure: remove now-empty backend/, untrack root-level dumps

backend/ contained only build cache after the moves; deleted in
working tree. NDJSON dumps and the Typst draft are untracked but
preserved on local disk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update `CLAUDE.md` paths and run command section

**Files:**
- Modify: `CLAUDE.md`

The `CLAUDE.md` "Commands" section says "All commands run from `backend/`" — false after Plan 1. Update both that section and the three-layer architecture summary at the top, which says `backend/src/simulacra/...`.

- [ ] **Step 13.1: Update the architecture descriptions**

In `CLAUDE.md`, replace these three lines:
- `- **Noumena** (\`backend/src/simulacra/noumena/\`)` → `- **Noumena** (\`src/simulacra/noumena/\`)`
- `- **Phenomena** (\`backend/src/simulacra/phenomena/\`)` → `- **Phenomena** (\`src/simulacra/phenomena/\`)`
- `- **Onboard** (\`backend/src/simulacra/onboard/\`)` → `- **Onboard** (\`src/simulacra/onboard/\`)`

Use the Edit tool with each line as old_string and the corrected version as new_string.

- [ ] **Step 13.2: Update the Commands section**

Find the block under "### Backend (Python, managed with `uv`)" that starts with "All commands run from `backend/`:" and update both the prose and the example block.

Replace:
```
All commands run from `backend/`:

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_quat.py       # run a single test file
uv run pytest tests/test_quat.py -k "test_name"  # run a single test
uv run -m examples.rocket_wasd_3d      # run WASD example (outputs NDJSON to stdout)
uv run -m examples.plot_wasd_3d output.ndjson  # plot from NDJSON log
```
```

With:
```
All commands run from the repo root:

```bash
uv run pytest                                          # run all tests
uv run pytest tests/test_quat.py                       # run a single test file
uv run pytest tests/test_quat.py -k "test_name"        # run a single test
uv run scenarios/hopper_short.py > /tmp/out.ndjson     # run a scenario (NDJSON to stdout)
uv run python examples/plot_telemetry.py /tmp/out.ndjson  # plot from NDJSON log
```
```

- [ ] **Step 13.3: Update the two trailing path references**

Find these two lines lower in `CLAUDE.md`:
- `Results saved to \`backend/runs/\`. Each run contains:` → `Results saved to \`research/runs/\` (historical) and future runs are gitignored.`
- `Comprehensive study design: \`backend/docs/feature_ablation_study.md\`` → `Comprehensive study design: \`research/docs/feature_ablation_study.md\``

- [ ] **Step 13.4: Stage and commit**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update CLAUDE.md paths and command examples for new layout

backend/* references replaced with the new top-level paths so commands
work as written from the repo root.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Final verification (no commit; gates the plan)

**Files:**
- Verify only.

This task is the gate: if any sub-step fails, the restructure is not done and the cause needs investigation.

- [ ] **Step 14.1: Tests still pass with same count as baseline**

```bash
cd /Users/ooj/Projects/Othisis/simulacra
uv run pytest -q --tb=no tests/ 2>&1 | tail -5 > /tmp/simulacra_post_pytest.txt
diff /tmp/simulacra_baseline_pytest.txt /tmp/simulacra_post_pytest.txt
```

Expected: `diff` produces no output (the summary lines match — same count of passes, failures, and warnings).

- [ ] **Step 14.2: `hopper_short` scenario produces identical NDJSON**

```bash
uv run scenarios/hopper_short.py > /tmp/simulacra_post_hopper_short.ndjson
wc -l /tmp/simulacra_baseline_hopper_short.ndjson /tmp/simulacra_post_hopper_short.ndjson
diff /tmp/simulacra_baseline_hopper_short.ndjson /tmp/simulacra_post_hopper_short.ndjson | head
```

Expected: identical line counts. The `diff` should be empty (line-for-line identical NDJSON output). If timestamps or seeds vary, investigate before continuing — that means something behavioural changed during the move, which is the failure mode this plan exists to prevent.

- [ ] **Step 14.3: Top-level layout matches the target**

```bash
ls /Users/ooj/Projects/Othisis/simulacra/ | grep -v -E '^(\.|temp$|hopper_.*\.ndjson|Wed-.*)'
```

Expected output (in some order):
```
AGENTS.md
CHANGELOG.md
CLAUDE.md
ONBOARDING.md
README.md
deploy_ucl.sh
docs
examples
frontend
pyproject.toml
references
research
scenarios
scripts
src
tests
uv.lock
```

No `backend/`. No straggler files at root.

- [ ] **Step 14.4: Tag the restructure**

```bash
git tag -a v0.1-restructure -m "Plan 1 complete: repo restructured to top-level src/ layout"
```

This tag is a clean rollback point if Plan 2 turns out to need rework.

- [ ] **Step 14.5: Push to `rocketry` remote**

```bash
git push rocketry feature/v0.1-sdk
git push rocketry v0.1-restructure
```

Expected: branch and tag both push successfully.

---

## Done criteria for Plan 1

All boxes above ticked, and:

1. Top-level `src/simulacra/`, `tests/`, `pyproject.toml`, `uv.lock` exist.
2. No `backend/` directory.
3. `research/` exists with `scenarios/`, `backend_scenarios/`, `docs/`, `runs/`, and a `README.md`.
4. `uv run pytest` from root passes with the same counts as Task 1.2 baseline.
5. `uv run scenarios/hopper_short.py > /tmp/post.ndjson` produces output byte-identical to Task 1.3 baseline.
6. `CLAUDE.md` no longer references `backend/` paths.
7. `v0.1-restructure` tag exists and is pushed to `rocketry`.

When all of the above are true, Plan 2 (API + reference migration) can begin.
