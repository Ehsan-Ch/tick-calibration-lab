# Research note: can probability calibration survive a temporal shift?

## Question and preregistered choices

The question is whether trade-flow and short return history improve three-class next-minute forecasts over a historical prior, and whether a simple post-hoc calibration step improves those probabilities. `config.json` fixes the dates, splits, horizon, threshold, candidates and bootstrap seed. The code and configuration were written before the first model evaluation. This is a locally recorded protocol, not an externally registered study.

The experiment uses ETHUSDT aggregate trades from Binance Vision on 1–14 January 2024. Each source archive is checked against its publisher SHA256 and a stored manifest. There are 11,075,805 aggregate-trade records, no detected aggregate-ID gaps and no empty 10-second bins in this sample. Each day yields 8,603 examples after warm-up and strict endpoint purging: 120,442 examples overall, of which 34,412 occur in the four evaluation days.

## Data timing and target

A prediction time is the exclusive end of a UTC 10-second bin. A trade exactly at a bin boundary belongs to the next bin. The feature vector uses that completed bin and previous bins only. All input timestamps use milliseconds, as appropriate for these 2024 archives; the loader rejects wrong units and dates rather than guessing.

Features comprise 10/60/300-second log returns, recent return standard deviations, intrabar and recent price range, log volume and count, signed taker-volume imbalance over three windows, last-trade age and five lagged returns. Buyer-maker trades contribute negative signed volume; the opposite contributes positive volume. These are trade-flow features, not order-book imbalance.

The label is the 60-second future last-trade-price log return in basis points: class 0 down (< −3), class 1 flat (between −3 and +3 inclusive), class 2 up (> +3). Neither current nor endpoint last-trade age may exceed 30 seconds. Features reset daily; there is no overnight carry. We exclude endpoints at or beyond the final UTC boundary. Partition checks additionally require the prior partition's maximum label-availability time to precede the next partition's first prediction.

These timestamps assume immediate access to completed exchange-time bins. The experiment does not model feed latency, out-of-order arrival at the consumer, execution delay or quote freshness. The replay visualizes label delay; it does not recreate exchange infrastructure.

## Models, selection and calibration

1. **Historical prior:** class frequencies from training only, with one pseudo-count per class. No test-period re-estimation.
2. **Incremental linear:** standardized features and an averaged SGD logistic classifier. The scaler is fitted on training chunks; the model receives six chronological passes of 4,096-row chunks. Alpha is selected from 0.0001 and 0.001 using the following selection day. There is no random validation split.
3. **Boosted trees:** histogram gradient boosting with 100 iterations, learning rate 0.07, minimum leaf size 100, L2 regularization 1 and either 7 or 15 maximum leaves. Automatic early stopping is disabled. The selection day chooses leaf count by log loss.

The selected candidates were alpha 0.001 and 7 leaves in every fold. Boosted trees beat the linear family on every selection day; this selection does not inspect test scores. Models are not refitted on selection or calibration days.

Temperature scaling applies softmax(log(p)/T). T is fitted by minimizing calibration-day log loss in [0.25, 4], with T=1 retained as an explicit candidate. It therefore cannot worsen the fitting objective on the calibration set; test improvement is not guaranteed. The linear model's F2/F3 temperatures reached the upper search bound, a diagnostic of severe overconfidence. This is preserved rather than retuned after seeing the tests. Temperature scaling preserves the most likely class, so F1/accuracy remain unchanged.

## Findings

| Variant | Log loss | ECE | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| Prior | 1.09481 | 0.02651 | 0.19338 | 0.40858 |
| Linear raw | 2.11130 | 0.47065 | 0.19423 | 0.40843 |
| Linear calibrated | 1.13683 | 0.14631 | 0.19423 | 0.40843 |
| Boosted raw | 1.04219 | 0.01556 | 0.38504 | 0.45159 |
| Boosted calibrated | 1.04045 | 0.01571 | 0.38504 | 0.45159 |

The calibrated boosted model improves pooled log loss relative to the prior by 4.97%, but accuracy remains about 45%. Much of the gain may concern predicting the flat class / volatility regime rather than profitable directional anticipation. No feature ablation was run to isolate the cause. The linear procedure is a weak baseline here: its training recipe and limited candidate grid should not be taken as evidence that all linear models fail.

Calibration improves boosted log loss slightly while slightly worsening top-label ECE. ECE uses ten fixed confidence bins; it is neither a proper scoring rule nor a complete classwise calibration diagnostic. Multiclass Brier is the mean sum of squared errors across all three classes (not divided by three). Proper scores combine calibration and discrimination, so improvement should not be attributed solely to calibration.

For calibrated boosted minus prior log loss, the paired difference is −0.05436 with a 95% percentile interval [−0.06212, −0.04744]. We use 300 seeded circular moving-block bootstrap replicates, with 60 consecutive rows (ten minutes) per block, sampled separately within each test day and preserving each day's sample count. This addresses some within-day dependence but **conditions on these four days**. It is not an interval over future market regimes, has no multiplicity correction and does not measure uncertainty from retraining.

The monitoring panel compares five feature distributions with their training distribution using quantile-bin PSI and half-count smoothing. The strongest selected-feature PSI is 0.421 on Jan 7 and 1.052 on Jan 10; Jan 13/14 are below 0.2. The 0.2 threshold is an illustrative review heuristic. It is not a significance test and does not identify causes. Monitoring is retrospective after the whole test day, not available at its start.

## Engineering and reproducibility

Raw archives are streamed row by row; they are never extracted or loaded as a full trade table. Aggregation keeps one day's arrays. The compact feature matrix is held in memory for training and selection. Only the linear trainer is incremental; the entire modeling pipeline is not claimed to be out-of-core. The local run took about 53 seconds on the available machine, including ingestion; this is an observation, not a controlled hardware benchmark. `reports/runtime.json` keeps timings separate from deterministic research outputs.

The compressed forecast export records fold, prediction time, label-availability time, actual class and probabilities for all five variants. `verify_report.py` independently recalculates six scoring metrics and confusion matrices for each fold and the pooled report from that export, checks its hash, and checks the protocol/source-manifest digests. Unit tests cover future-feature invariance, strict split separation, target timing, input validation, calibration and replay outcome masking. CI does not retrain or download all data.

## Why this is relevant to Alipes—and what it does not demonstrate

The advertised role emphasizes dataset preparation, calibration/benchmarking and time-series predictive modeling in Python. This project supplies concrete artifacts for those skills: source integrity, incremental processing, causal features, separated model-selection/calibration periods and transparent negative results.

It does not reproduce Alipes' strategy, news-reading system, proprietary data or production stack. For an equities follow-up, obtain appropriate licensed trade and quote records, implement exchange-session and corporate-action handling, test multiple instruments across months, include spread-aware labels and feed delays, and benchmark against stronger linear and sequence models. News features require publication/arrival timestamps and a separate leakage audit. Those experiments require fresh evaluation periods; the current tests should not become a tuning set.

No P&L backtest is presented. Last-trade-price forecasts alone cannot establish attainable fills, spread/fee coverage, queue position, market impact or capacity. The present result is a small reproducible research exercise, not evidence of a deployable strategy.

## Primary references

- [Alipes role](https://careers.alipes.dk/jobs/4881094-machine-learning-researcher), consulted 30 October 2026.
- [Binance public archive schema](https://github.com/binance/binance-public-data) and [aggregate-trade endpoint semantics](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market).
- [Guo et al., 2017: On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), temperature-scaling method.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) and [histogram boosting, version 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
