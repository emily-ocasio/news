# TODO

## Orphan Adjudication

- Persist `adjudication_e2e` terminal decision cache incrementally during orphan adjudication instead of waiting for end-of-run batch finalization. Target the point where each group or orphan reaches a terminal `CaseDecision`, so partial progress survives mid-run failures.

- Handle user abort via `Ctrl-C` in the orphan adjudication controller similarly to the `[S]` and `[G]` controllers. On abort, return cleanly to the menu and show a summary of work completed so far, including queued groups/orphans processed, terminal decisions persisted, and any partial run state left in progress.

## Incident Deduplication

- In the incident dedupe comparison logic, when both sides have more than one offender and the selected primary offender names do not match, set that offender-name comparison level to `null` instead of a mismatch so there is no penalty for two articles or entities choosing different offenders as the main offender.

- Documented motivating case: entities `100721943:​0:​1` and `100721943:​0:​0` are the pair in question, and orphans `100861124:​0:​0` and `100861124:​0:​1` recently split off because the offender-name mismatch pushed the comparison score below threshold.
