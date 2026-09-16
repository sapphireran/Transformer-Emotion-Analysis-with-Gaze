# Gaze Sidecar Atlas (docs)

Personal notes for this checkout. They describe the code *as committed in
January 2024*, plus measurements taken from the committed CSVs.

| Note | Question it answers |
| --- | --- |
| [two-tracks.md](two-tracks.md) | Why are there two trainers and two almost-disjoint tables? |
| [sidecar-geometry.md](sidecar-geometry.md) | What does `EyeTrackingModel` actually concatenate? |
| [feature-dictionary.md](feature-dictionary.md) | nFix vs nFixations, FFD, GPT, TRT, GD, SFD, pupil |
| [subject-3-reindex.md](subject-3-reindex.md) | Why `3_SR.csv` has 299 rows numbered 0–298 |
| [splits.md](splits.md) | 80/10/10 vs StratifiedKFold, text leaks |
| [scripts-as-found.md](scripts-as-found.md) | File-by-file walk through the 2024 Python |
| [known-issues.md](known-issues.md) | Bugs that change numbers if you rerun training |
| [gaze-prediction-track.md](gaze-prediction-track.md) | Raw vs z-scored spaces, Provo, placeholders |
| [reproduction.md](reproduction.md) | What you can reproduce on CPU vs what needs a GPU |
| [findings.md](findings.md) | Numbers the examples recompute |

Run the examples after reading:

```bash
python3 examples/run_all.py
```
