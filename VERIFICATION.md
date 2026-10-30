# Release verification

Verified 30 October 2026, Europe/Copenhagen.

## Published release

- Code and research release: `b0d473d754665cc792da95a032f9988dc4f84aab`.
- All 29 release files matched local Git blob hashes after upload. Raw source archives and local feature caches were excluded.
- [Research checks](https://github.com/Ehsan-Ch/tick-calibration-lab/actions/runs/41058781393) succeeded: Python 3.11, Python 3.12 and Node 22 browser-logic job.
- [GitHub Pages deployment](https://github.com/Ehsan-Ch/tick-calibration-lab/actions/runs/41058822654) succeeded for that release, from `main` / `docs`.
- Local validation: 18 Python tests, 6 JavaScript tests and independent reconciliation of all 34,412 forecast records across 15 fold/model variants.
- Staged protocol, source-manifest and forecast archive hashes were checked before publishing; LF normalization is specified for consistent checkout behavior across platforms.

## Browser checks

The [public research console](https://ehsan-ch.github.io/tick-calibration-lab/) loaded the complete report. Model, evaluation-window and calibration controls changed the metrics and tables. The public F3 boosted/raw CSV export contained the correct 17,206 observations and its exact log loss matched the saved report. The complete public `predictions.csv.gz` download matched the report's SHA256 digest.

Replay on 14 January, after advancing to 00:06:10 UTC, displayed the first prediction's actual class while newer outcomes remained hidden. The initial hidden state and the subsequent reveal were checked locally on 7 January and in automated replay tests.

Local F2 linear/raw CSV export matched 8,603 observations and log loss 2.2079644152050775. Desktop chart layout was visually inspected. Mobile layout at 390 × 844 pixels had document client and scroll widths of 375 pixels, with no horizontal page overflow. These are smoke checks, not exhaustive device/browser certification.

## Evidence boundaries

CI reconciles stored forecasts and tests logic; it does not download all source archives or retrain the models. The full benchmark was run locally with the pinned dependencies. The research uses one crypto asset and four held-out days; these checks do not establish equities generalization, live data availability, execution quality or profitability. See [RESEARCH.md](RESEARCH.md) for the statistical protocol and negative findings.
