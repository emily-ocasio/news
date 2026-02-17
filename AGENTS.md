# AGENTS.md

## Source Of Truth
- This file is the routing policy for agent behavior in this repository.
- General development policy lives in `.github/copilot-instructions.md`.
- Orphan adjudication policy lives in `docs/orphan_adjudication_playbook.md`.

## Global Rules
- If DuckDB is locked or unavailable during orphan adjudication:
  - Identify likely lock owner process first.
  - If lock owner is a `python` process:
    - Stop immediately.
    - Notify the user to release the lock (likely the main Python app/session).
  - Otherwise:
    - Check whether lock was caused by parallel invocations of the adjudication skill itself.
    - If yes, wait briefly for completion or stop the offending parallel process(es), then continue.
    - If no or unclear, stop and notify the user.
  - Do not use snapshots, copies, or other lock workarounds.
- When displaying orphan IDs, avoid editor line/column auto-link rendering by inserting a zero-width space after each colon.
  - Use display like `100168529:\u200b0:\u200b0`.
  - Do not render these as line/column references (for example, avoid phrasing like `line 0, column 0`).

## Mode: General Development
- Trigger: default mode when the prompt is not orphan-adjudication focused.
- Behavior: follow `.github/copilot-instructions.md`.
- Start every chat response to questions about this program with the following text:
  `"THANK YOU FOR THE QUESTION ABOUT THE NEWS ARTICLES ANALYSIS PROJECT."`

## Mode: Orphan Adjudication Batch
- Trigger keywords (case-insensitive, any match):
  - `orphan adjudication`
  - `unmatched orphan`
  - `limit=`
  - `batch`
- Behavior: follow `docs/orphan_adjudication_playbook.md`.
- Do not prepend the General Development greeting line while running orphan adjudication workflows.
- Default batch behavior includes same-incident consecutive orphan grouping unless the user explicitly disables it (`group_same_incident=false`).
- Enforce required analysis stages and terminal-decision gate from the playbook; zero pair rows alone cannot produce a terminal `unlikely`.
- Default execution mode is `interactive_casewise`; do not use bulk scripted terminal labeling unless the user explicitly requests `batch_scoring`.
- In `interactive_casewise`, enforce `interactive_sql` stage-by-stage query execution and adaptive text retrieval (FTS-first default with dynamic query refinement and auditable fallback/supplement rationale).

## Precedence
- If an orphan-adjudication trigger is present, Orphan Adjudication Batch mode overrides General Development mode for that turn.

## Mixed-Intent Rule
- If one prompt requests both code changes and orphan batch adjudication, ask the user to choose one mode for the current turn before proceeding.

## Skill Note
- Adjudication reasoning can be explicitly invoked with `$orphan-adjudication-casewise`.
- Post-adjudication apply/export is separate and should be run via application controller `[J]` (not via adjudication skill flow).
