# TODO

## Orphan Adjudication

- Persist `adjudication_e2e` terminal decision cache incrementally during orphan adjudication instead of waiting for end-of-run batch finalization. Target the point where each group or orphan reaches a terminal `CaseDecision`, so partial progress survives mid-run failures.

- Handle user abort via `Ctrl-C` in the orphan adjudication controller similarly to the `[S]` and `[G]` controllers. On abort, return cleanly to the menu and show a summary of work completed so far, including queued groups/orphans processed, terminal decisions persisted, and any partial run state left in progress.
