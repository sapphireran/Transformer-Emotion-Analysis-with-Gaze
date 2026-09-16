# Lab notes

Read in order if you are new to this clone. The HTML version
([`labbook.html`](labbook.html)) is generated from the same CSVs.

1. [Two tracks](01-two-tracks.md) — recorded 400 vs projected 11.8k.
2. [Data dictionary](02-data-dictionary.md) — `nFix` vs `nFixations`.
3. [Reader-3 forensics](03-reader3-forensics.md) — why id 150 has SentLen 9.5.
4. [Word-token census](04-word-token-census.md) — `emp11111ty` and friends.
5. [Predicted-gaze degeneracy](05-predicted-gaze-degeneracy.md) — rank-1 SST channels.
6. [Reader reliability](06-reader-reliability.md) — ICC on sentences 0–149 only.
7. [Pipeline and path traps](07-pipeline-and-path-traps.md) — `et_csv_data` vs git.
8. [Trainer errata](08-trainer-errata.md) — last 162 test rows; unused ZuCo split.
9. [CPU baselines](09-cpu-baselines.md) — how to read the ridge probes.
10. [Citations](10-citations.md).

Measured snapshots: [`sample-results.md`](sample-results.md) (filled after
`examples/run_all.py`) and `examples/output/`.
