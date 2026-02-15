# Orphan Adjudication Playbook

## Source Of Truth
- This file is the authoritative workflow for orphan adjudication batch mode.
- Routing and precedence are defined in `AGENTS.md`.
- General development policy remains in `.github/copilot-instructions.md`.

## Purpose
Perform post-Splink adjudication for unmatched orphans, persist decisions for replay, and produce transparent evidence-based outputs.

## Pipeline Position
This workflow runs after the core extraction and linkage pipeline steps have already completed:
1. GPT extraction completed for in-scope articles (structured victim/incident data populated in SQLite, including `gptVictimJson` where applicable).
2. Incident setup completed (incident records/fields staged into DuckDB inputs).
3. Initial clustering completed for named-victim records (entity clusters formed).
4. Orphan linkage completed (unnamed-victim records attempted against existing entities).
5. This adjudication step investigates the remaining unlinked orphans for likely missed matches beyond expected deterministic/Splink coverage.

Downstream expectation:
- Adjudication outputs are persisted so they can be re-applied after future reruns of incident setup, clustering, and orphan linkage without repeating agent reasoning.

## Scope
- This workflow is for post-Splink adjudication only.
- It is not bug-finding and not a replacement for Splink clustering/linkage.
- It is an interactive agent workflow with direct-write persistence.

## Execution Mode
- Default mode: `interactive_casewise`.
- Optional mode: `batch_scoring` (only when explicitly requested by user).
- If no mode is specified in the user prompt, use `interactive_casewise`.
- In `interactive_casewise`, the agent must adjudicate each orphan/group with direct narrative reasoning and case-specific evidence.
- In `batch_scoring`, outputs are screening-only unless user explicitly approves terminal adjudication criteria.
- Primary task type: qualitative textual analysis and reasoning per case; scripting is allowed only for data retrieval support, not for autonomous terminal adjudication.

## Query Execution Policy
- Default query mode in `interactive_casewise` is `interactive_sql`.
- Use direct incremental SQL per stage (DuckDB and SQLite) for each orphan/group.
- Do not use a single monolithic Python batch script to perform terminal adjudication in `interactive_casewise`.
- Scripted helpers are allowed only for retrieval support and must still emit stage-by-stage SQL summaries and row counts.
- Retrieval strategy is adaptive: agents may iteratively reformulate queries based on intermediate findings as long as they log what changed and why.

## Inputs
- Required:
  - `limit` (number of unmatched orphans to process in this run)
- Optional:
  - `starting_after_orphan_id`
  - `dry_run` (`true` or `false`)
  - filter hints (for example, `dataset=CLASS_WP`, `gptClass=M`)
  - `group_same_incident` (`true` or `false`, default `true`)

## Orphan Queue Ordering
- Queue selection must be deterministic.
- Default ordering for "next N unmatched orphans" must mirror `orphan_matches_final.xlsx` generation order:
  1. `group_midpoint_day` ascending, `NULLS LAST`
  2. `match_id` ascending
  3. `CASE rec_type WHEN 'entity' THEN 0 ELSE 1 END` ascending
  4. `uid` ascending
- Use unmatched rows from `orphan_matches_final_current` where `rec_type='orphan'`.
- Queue eligibility must exclude previously adjudicated orphans with terminal decisions:
  - exclude if `orphan_id` exists in `orphan_adjudication_overrides` with `resolution_label IN ('likely_missed_match','possible_but_weak','unlikely')`.
  - allow retry only when prior `resolution_label = 'analysis_incomplete'`.
- If `starting_after_orphan_id` is provided:
  - locate that orphan in the same ordering,
  - start strictly after it,
  - then take next `limit` rows.
- Agent should report the queue-order key and first/last selected orphan IDs in the run summary.

## Loop Contract
Process queue items, where each item is either a single orphan or a grouped incident batch:
1. Find next unmatched orphan.
2. If `group_same_incident=true`, detect consecutive orphans that should be adjudicated together.
3. Build dossier for the queue item from DuckDB and supporting article context from SQLite.
4. Build recall-first candidate pool for the item.
5. Run dynamic-depth analysis:
   - Expand/narrow queries based on prior findings.
   - Stop gathering when evidence is sufficient for a final label.
6. Adjudicate final label per orphan in the item.
7. Persist outcomes.
8. Continue until stop condition.

## Same-Incident Grouping Rules
When `group_same_incident=true`, group consecutive unmatched orphans before deep analysis.

### Primary grouping key
- Same `article_id` (the most important rule).

### Secondary compatibility checks
- Same incident-date signal (exact date or same year/month when date is partial).
- Same city/area context.
- No hard contradiction in extracted victim/event fields.

### Group processing behavior
- Build a shared candidate pool and shared narrative anchor set once per group.
- Produce per-orphan decisions after reviewing shared and orphan-specific evidence.
- If one orphan clearly diverges, split it from the group and process separately.

## Decision Labels
- `likely_missed_match`
- `possible_but_weak`
- `unlikely`
- `analysis_incomplete` (non-terminal safeguard label; do not persist as final adjudication)

## Label Semantics
- Use `unlikely` when required stages were completed but evidence remains non-specific, contradictory, or insufficient for defensible person-level linkage.
- Use `analysis_incomplete` only for process failure conditions:
  - required stage not executed,
  - database/tooling failure,
  - missing required evidence due to execution failure (not due to inherently vague source narrative).
- For composite or high-level narrative articles with no unique incident anchors after fallback review, prefer `unlikely` with reason code `insufficient_incident_specificity_after_fallback`.

## Prohibited Shortcuts
- Do not assign `unlikely` solely because `orphan_entity_pairs` has zero rows.
- Do not end analysis after proving an orphan is unmatched; that is precondition context, not adjudication evidence.
- Do not use token-overlap-only, lexical-intersection-only, regex-only, or single-score heuristics as the sole basis for a terminal label.
- Do not generate one bulk script to auto-label all cases with terminal outcomes unless the user explicitly requests `batch_scoring`.
- In `interactive_casewise`, do not replace stage-by-stage SQL analysis with one precomputed script result.

## Evidence Principles
- Evaluate both structured and narrative signals.
- Consider MAR normalization/correction effects when location fields disagree.
- Weapon compatibility guidance:
  - Strongly incompatible by default: knife vs shotgun.
  - Potentially compatible with ambiguity: beating vs blunt object (for example, fists vs bat uncertainty).
- If Splink missed the case, document likely scope/feature blocking reasons where supported.

## Adaptive Retrieval Principles
- Stage B/C/C2 are goal-driven, not template-driven: maximize true-candidate recall while keeping search explainable.
- Hard filters should be minimal and defensible (for example, city scope and broad year bounds).
- Most incident attributes (circumstance, weapon family, partial dates, coarse location phrases) should be treated as soft ranking signals unless a true contradiction exists.
- Use compatibility families where appropriate (for example robbery/burglary, blunt object/beating) to avoid premature candidate exclusion.
- Agents should prefer iterative query refinement over fixed phrase lists when narrative wording is secondary, indirect, or historical.

## Multi-Victim Constraint
- Multiple victims from the same incident are distinct people by definition.
- Never match co-victims to each other.
- Never collapse multiple co-victims to the same target person/entity.
- If two orphans from one incident could both map to one candidate entity, treat that as a conflict and continue search or mark `analysis_incomplete` if unresolved.

## Multi-Victim Group Assignment Rule
When all conditions below are true, allow a strong group-level resolution:
1. Incident-level match is strong and unique.
2. Orphan subgroup size equals candidate entity subgroup size.
3. Orphans in the subgroup are indistinguishable from each other (no reliable differentiating attributes).

Then:
- Assign each orphan in the subgroup as `likely_missed_match`.
- Use a one-to-one arbitrary bijection across the matched entity subgroup (never many-to-one).
- Record explicit provenance in evidence output:
  - `assignment_mode = group_level_arbitrary_bijection`
  - `group_confidence = high`
  - `individual_identity_confidence = low`

## Required Analysis Stages
Before a terminal label (`likely_missed_match`, `possible_but_weak`, `unlikely`) is allowed, complete and report the following stages per orphan/group.

1. Stage A: baseline candidate check
- Query existing `orphan_entity_pairs`/`orphan_entity_pairs_top1` for context only.
- If zero rows, continue to Stage B (never finalize here).

2. Stage B: fallback candidate generation (mandatory when Stage A has zero rows)
- Query `entity_link_input` using recall-first constraints:
  - same city (or nearest compatible city context),
  - year/date proximity windows,
  - non-name overlap fields (sex, weapon family, circumstance family, coarse location/date compatibility).
- Stage B guidance:
  - Use soft scoring/ranking for most features by default.
  - Avoid hard `WHERE` exclusions on circumstance/weapon unless there is a true impossibility.
  - Candidate exclusion should be explainable with a specific contradiction, not just non-equality.
  - Stage B candidate cap policy:
    - minimum `LIMIT` is `75`.
    - if source narrative is secondary/reference style or orphan fields are sparse (`unknown`/`NULL` location, age, sex, weapon, or weak date precision), minimum `LIMIT` is `125`.
    - agent must record chosen Stage B `LIMIT` and rationale in the stage trace.

3. Stage C: relaxed expansion (mandatory when Stage B is sparse/empty)
- Widen date and location tolerances.
- Include MAR-corrected/normalized address compatibility.
- Preserve obvious hard contradictions (for example, extreme method mismatch unless source/extraction error evidence exists).

3b. Stage C2: text-led candidate expansion (mandatory trigger conditions)
- Run article-text retrieval before any terminal `unlikely` when one or more conditions hold:
  - source narrative is secondary/reference style (for example, commentary, roundup, retrospective),
  - structured orphan fields are sparse (`unknown`/`NULL` location, age, sex, or weak date precision),
  - Stage B/C produced weak or generic candidate sets with no strong anchor fit.
- Retrieval path guidance:
  - Use the configured FTS index/table (for example `articles_wp_m_fts`) as the default starting path.
  - Expand/refine phrase sets dynamically from source anchors and intermediate results.
  - If FTS coverage appears weak for the case wording, supplement with non-FTS retrieval and log rationale.
- Use source anchors to find likely underlying incident article(s), then map those articles back to candidate entities.
- Treat text-led hits as candidate-generation inputs (not terminal evidence by themselves); they must still pass Stage D narrative comparison.

4. Stage D: narrative anchor comparison (always required before terminal `unlikely`)
- Compare source-article anchors against top fallback candidates using article text and extracted context.
- Anchors include event-specific cues such as method details, location phrasing, offender cues, and rare incident descriptors.
- Record concrete anchor evidence:
  - at least 2 orphan-side anchors,
  - at least 2 candidate-side supporting or contradicting anchors.

5. Stage E: minimum candidate review
- Review at least top 3 candidates from fallback/retrieval stages (or all if fewer than 3).
- Provide a concise signal matrix per reviewed candidate.
- Candidate pool for Stage E must include any viable candidates discovered in Stage C2 text-led expansion.

## Terminal Decision Gate
- A terminal label is allowed only if all required stages are satisfied.
- If a required stage is skipped or fails unexpectedly:
  - assign `analysis_incomplete`,
  - report missing stage(s),
  - do not persist as final adjudication.
- A terminal `likely_missed_match` or `possible_but_weak` requires explicit fact-level correspondence, not score-only similarity.
- A terminal `unlikely` requires explicit conflict analysis (date/location/weapon/circumstance) and narrative-anchor review.
- If anchor evidence is missing because the source is inherently non-specific after required stages, assign `unlikely` with an explicit specificity-based reason code.
- If anchor evidence is missing because analysis steps failed to run, assign `analysis_incomplete`.
- Terminal decisions must satisfy the Multi-Victim Constraint.
- A terminal `unlikely` is invalid if Stage C2 trigger conditions were present but Stage C2 was not executed.
- A terminal `unlikely` is invalid if Stage C2 lacked an auditable retrieval rationale (queries used, why chosen, and why alternatives were rejected).

## Persistence Contract
Use DuckDB direct writes unless `dry_run=true`.

### Current Override Table
- Upsert to `orphan_adjudication_overrides` with:
  - `orphan_id` (stable key)
  - `resolution_label`
  - `resolved_entity_id` (nullable)
  - `confidence` (nullable/score)
  - `reason_summary`
  - `evidence_json`
  - `analyst_mode` (set to `interactive_agent`)
  - timestamps
- Do not upsert rows with terminal status `analysis_incomplete`.

### History Table
- Append to `orphan_adjudication_history` for each applied decision.
- Preserve prior and current values with run metadata for auditability.

## Stop Conditions
Stop the batch when any condition is met:
1. Processed `limit` orphans.
2. No more unmatched orphans are available.
3. DuckDB lock or hard database error occurs.

## Lock Prevention
- do *not* perform parallel database instructions because they will create locks
- strictly sequential database instructions

## Lock Handling
- On lock detection:
  - Stop immediately.
  - Report lock condition to the user.
  - Do not use snapshots or file-copy workarounds.
  - Resume only after user confirms lock is released.

## Output Contract
At end of run, return:
1. Batch summary:
  - requested limit
  - processed count
  - grouped count
  - counts by label
  - errors/skips
2. Per-orphan outcome list with:
  - orphan ID
  - group ID (nullable when processed singly)
  - final label
  - top candidates
  - concise reason summary
  - stage trace (stages executed, SQL summary, row counts, decision gate result)
  - execution audit:
    - `query_mode` (`interactive_sql` or `batch_script`)
    - `fts_used` (`true` or `false`)
    - `fts_query_list`
    - `sqlite_query_count`
    - `duckdb_query_count`
  - narrative evidence block:
    - orphan anchors (min 2)
    - candidate anchors (min 2)
    - conflict analysis (date/location/weapon/circumstance)
3. Persistence status:
  - rows upserted
  - rows appended to history
  - dry-run status

## Batch Quality Control
- For runs where `limit > 5`, complete deep narrative writeups for at least the first 3 adjudications before proceeding with the rest.
- If those first 3 fail the terminal decision gate, stop the batch and report remediation needed.

## Display Rule For IDs
- When rendering IDs such as for orphans, entities, clusters, or any other ID that uses colons to separate incident and victim numbers, insert a zero-width space after each colon to avoid editor line/column auto-linking issues.
- Example display: `100168529:\u200b0:\u200b0`
- Do not describe such suffixes as file line/column coordinates.
