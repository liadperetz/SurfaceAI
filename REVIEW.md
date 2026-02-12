# Codebase Review — SurfaceAI

Review conducted for publication readiness alongside the paper
*"Architectural Causes of Jailbreak Vulnerabilities in Multi-Agent Systems"*.

---

## Overall Assessment

The codebase is well-structured and clearly maps to the paper's experiment design.
The three-layer evaluation framework (LLM, OpenHands, MAS) is cleanly separated,
Pydantic models enforce consistent data flow, and the orchestrators faithfully
implement each architecture described in the paper.

The changes below are a polish pass — no architectural issues were found.

---

## Changes Made

### 1. Extracted duplicated `parse_steps()` into shared utility

**Before:** Identical `_parse_steps()` function (regex parsing of numbered/bulleted
steps) was copy-pasted in all 4 orchestrator files.

**After:** Single implementation in `src/surfaceai/mas/orchestrators/utils.py`,
imported by `planner_executor.py`, `centralized.py`, `fully_connected.py`, and
`sequential_chain.py`.

### 2. Extracted duplicated `parse_json()` into shared utility

**Before:** Identical JSON-from-LLM parsing logic (markdown code block extraction +
fallback regex) duplicated 3 times across `Judge._parse_json()`,
`FiveLevelJudge._parse_json()`, and `MASJudge._parse_json()`.

**After:** Single `parse_json()` function in `src/surfaceai/utils.py` with a
configurable `fallback` parameter. Each judge passes its own fallback dict.

### 3. Removed keyword-based heuristics — all classification is LLM-based

**Before:** `Judge` and `FiveLevelJudge` accepted `llm_client=None` and fell back to
keyword-matching heuristics using a hardcoded `REFUSAL_PATTERNS` list. The E2 planner's
refusal detection (`LLMAgent._check_refusal`) also used the same keyword list.

**After:**
- `Judge` and `FiveLevelJudge` now require `llm_client` (no `None` default).
  The heuristic fallback paths and `REFUSAL_PATTERNS` constant are removed entirely.
- E2 refusal detection uses an LLM call with a dedicated classifier prompt instead of
  keyword matching, making it consistent with the rest of the evaluation pipeline.

### 4. Updated `MASExperiment` enum comments to match paper terminology

**Before:** E3-E9 all labeled "placeholder".

**After:** E3-E5 labeled with their architecture names (Centralized Orchestration,
Fully Connected Communication, Sequential Action Chain). E6-E9 labeled with paper
section names plus "(not yet implemented)".

### 5. Updated E6-E9 placeholder prompt strings

**Before:** Vague "TODO: E6 - Shared Memory Planning" strings.

**After:** Clear "[NOT IMPLEMENTED] E6 — Private Agent State: Planner" format matching
paper terminology.

### 6. Fixed import ordering in `providers.py`

**Before:** `from surfaceai.config.settings import get_settings` appeared *after* the
`logger = logging.getLogger(__name__)` line (non-standard).

**After:** Import moved above the logger definition, following standard import ordering.

### 7. Deleted internal development documents

Removed `PLAN.md` and `snug-munching-eclipse.md` — internal dev planning docs that
should not ship with the research artifact.

### 8. Fixed pre-existing lint warnings

Removed unused imports: `Literal` in `schemas.py`, `Any` in `llm_agent.py` and
`openhands_agent.py`, `Optional` in `mas/judge.py`. Removed unused exception variable
in `judge.py`.

---

## Files Created

| File | Purpose |
|------|---------|
| `src/surfaceai/utils.py` | Shared `parse_json()` |
| `src/surfaceai/mas/orchestrators/utils.py` | Shared `parse_steps()` |

## Files Modified

| File | Change |
|------|--------|
| `src/surfaceai/judge.py` | Require `llm_client`, remove heuristic fallback, use shared `parse_json` |
| `src/surfaceai/mas/judge.py` | Import `parse_json` from utils, remove duplicated method |
| `src/surfaceai/mas/agents/llm_agent.py` | Replace keyword refusal check with LLM-based classifier |
| `src/surfaceai/mas/orchestrators/planner_executor.py` | Import `parse_steps` from utils, remove local copy |
| `src/surfaceai/mas/orchestrators/centralized.py` | Import `parse_steps` from utils, remove local copy |
| `src/surfaceai/mas/orchestrators/fully_connected.py` | Import `parse_steps` from utils, remove local copy |
| `src/surfaceai/mas/orchestrators/sequential_chain.py` | Import `parse_steps` from utils, remove local copy |
| `src/surfaceai/config/schemas.py` | Updated enum comments, removed unused `Literal` import |
| `src/surfaceai/mas/prompts.py` | Clearer E6-E9 placeholder strings |
| `src/surfaceai/providers.py` | Fixed import ordering |
| `src/surfaceai/mas/agents/openhands_agent.py` | Removed unused `Any` import |

## Files Deleted

| File | Reason |
|------|--------|
| `PLAN.md` | Internal dev planning doc |
| `snug-munching-eclipse.md` | Internal dev planning doc |

---

## Paper Alignment

The implemented experiments (E1-E5) map correctly to the paper's architectures:

| Experiment | Paper Section | Implementation |
|------------|--------------|----------------|
| E1 | Planner-Executor Decomposition | `PlannerExecutorOrchestrator` |
| E2 | Planner with Safety Filtering | Same orchestrator, `safety_filtering=True` |
| E3 | Centralized Orchestration | `CentralizedOrchestrator` |
| E4 | Fully Connected Communication | `FullyConnectedOrchestrator` |
| E5 | Sequential Action Chain | `SequentialChainOrchestrator` |

E6-E9 are correctly marked as not yet implemented in both the enum and prompt files.

---

## Code Quality Notes

- No lint errors (ruff clean).
- All imports resolve correctly.
- Test infrastructure is in place (`pytest`, markers for `integration` and `docker`).
- Pydantic models enforce schema consistency between trace records and summary stats.
- Orchestrators are deterministic Python logic (not LLM-driven), matching the paper's design.
