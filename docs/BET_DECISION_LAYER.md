# The bet-decision layer

## What this is, and what it is not

The forecast engine answers *"what will the high be?"* This layer answers a
different, narrower question: *"is that answer good enough to bet on this
exact 2°F bucket?"* — and it is expected to answer **no** most of the time.

The two must stay separate. A filter that can edit the forecast will
eventually be tuned to make the forecast look good, which is the failure
mode the whole layer exists to prevent. `polyweather.betfilter` imports
nothing from the forecasting pipeline; `dashboard_payload` attaches its
verdict under a separate `betDecision` key, leaving every pre-existing
forecast field byte-identical.

```
raw guidance → forecast engine → temperature distribution
  → bucket probabilities → calibration → reliability + stability + agreement
  → bet quality score → hard rejection gates → decision
```

## The headline finding

Measured on 10,279 held-out station-days, **the model's exact-bucket hit rate
is 36.5%.** A 1.75°F MAE is a genuinely good daily-high forecast and is still
not enough to reliably hit a 3°F-wide settlement window: hitting that window
80% of the time needs σ ≈ 1.17°F, and the current model runs about σ ≈ 2.2°F.

That gap is not closable by tuning thresholds. What the filter *can* do is
find the subset of station-days where the forecast happens to be both sharp
and well-centred. On strictly held-out data (fitted on the earlier 60%,
evaluated on the later 40%) that subset behaves like this:

| Gate | Bets | Coverage | Win rate | 95% CI |
|---|---:|---:|---:|---|
| none (bet every favourite) | 3,919 | 100% | 36.5% | [35.0, 38.0] |
| calibrated p ≥ 0.50 | 120 | 3.1% | 55.8% | [46.9, 64.4] |
| calibrated p ≥ 0.55 + boundary + spread | 58 | 1.5% | **67.2%** | [54.4, 77.9] |
| calibrated p ≥ 0.55, tightest | 31 | 0.8% | 67.7% | [50.1, 81.4] |

A ~30 percentage-point lift is real and large. An 80% win rate is not
available at any threshold, and the honest operating range is roughly
**60–70% at 1–3% coverage** — a handful of bets per month across 20 cities,
with confidence intervals wide enough that a good month and a bad month look
identical.

## Probability calibration

Raw bucket probabilities are **overconfident**: markets this system labelled
64% settled 51% of the time. Every gate is a probability threshold, so an
uncorrected probability silently moves every gate — asking for 63% while
receiving 51% is not a conservative filter, it is a broken one.

`betfilter/calibration.py` fits an isotonic correction on resolved history
only. Two guards matter:

- **Sparse-tail shrinkage.** The top of the probability range is always the
  thinnest part of the sample, and unshrunk isotonic maps a stated 0.75 onto
  a claimed certainty of 1.0. Each grid point is pulled back toward the
  uncorrected value in proportion to how little evidence sits nearby.
- **No extrapolated confidence.** `np.interp` clamps outside the fitted
  range, which can *raise* a probability more confident than anything the
  calibrator ever saw. Beyond the fitted range the correction may only ever
  be conservative.

Applying it cut expected calibration error by 33% on held-out data.

## Hard gates

Gates run **before** scoring and cannot be outvoted by it, because a weighted
average will always let unrelated strengths drag a fatal weakness back over
the line. A beautifully stable forecast from agreeing models is still
unbettable if it sits 0.1°F from a bucket edge.

| Gate | Conservative default | Why |
|---|---|---|
| `minimum_range_probability` | 0.63 | A coin flip is not evidence |
| `minimum_normalized_boundary_safety` | 0.45σ | Raw degrees are not comparable across stations |
| `maximum_ensemble_spread_f` | 3.0 | Sources that disagree do not know |
| `maximum_forecast_revision_6h_f` | 1.75 | A forecast still moving has not landed |
| `maximum_bucket_flips_12h` | 1 | The recommendation itself already changed |
| `minimum_probability_gap` | 0.08 | 45%/43% is a coin flip wearing a favourite's label |
| `minimum_station_adjusted_accuracy` | 0.45 | Shrunk, never raw |

Data-availability failures return `DATA_INSUFFICIENT`, not `PASS`. *"We
cannot tell"* and *"we looked and it is not good enough"* are different
answers and collapsing them hides which one happened.

## Bet Quality Score

Weighted 0–100, every component derived from a measured quantity. Weights
live in `betfilter/config.py` and are the thing to backtest, not to argue
about.

| Component | Weight | Notes |
|---|---:|---|
| Range probability | 30% | Scores 0 at p ≤ 0.45; a 50% bucket earns nothing |
| Ensemble agreement | 15% | |
| Forecast stability | 15% | A single snapshot scores 0, not neutral |
| Boundary safety | 10% | 1σ of headroom is full marks |
| Station reliability | 10% | Asymmetric — see below |
| Observation alignment | 10% | Same-day only |
| Weather uncertainty | 5% | 1 − normalized distribution entropy |
| Forecast horizon | 5% | |

Tiers: ELITE ≥ 90, STRONG ≥ 82, PLAYABLE ≥ 75, MARGINAL ≥ 68, else PASS.
**Only ELITE and STRONG count as recommendations.** MARGINAL exists to be
shown and skipped; if it ever counts as a bet the selectivity premise
collapses.

## Station reliability

Three wins from three bets is not a 100% station, and stopping that number
from reaching a score is most of what `reliability.py` is for. Every rate is
a Beta-Binomial posterior shrunk toward the global prior with strength set by
sample size, and every rate carries a Wilson interval.

The score component is deliberately **asymmetric**: a thin record cannot lift
the score above neutral, but a genuinely poor record still drags it down.
Absence of evidence is not evidence of quality; evidence of failure is
evidence of failure regardless of sample size.

Real exact-bucket accuracy varies enormously by station — KLAS 52.8%, KSFO
25.8%. Not every city is worth betting, and the reliability gate encodes
that rather than hiding it behind a board-wide average.

## Financial results

**A bet is a win only when more money came back than went in.** A $3.60 entry
returning $2.56 is displayed by the platform as a win and is a $1.04 loss.
`results.financial_result` is what every reliability and backtest number
consumes; the platform's own label is stored but never scored.

Break-even counts as a loss: capital was risked and nothing was gained.

## Measuring whether the filter helps

`betfilter/backtest.py` requires that **rejected** markets were logged too.
If only placed bets are stored, selection bias makes all of it meaningless.

- `effectiveness()` compares filtered picks against betting every model
  favourite — the honest control, and the only thing that justifies the
  filter existing.
- `sweep_thresholds()` grid-searches the gates against resolved history.
  Rows below the minimum sample are returned but flagged: a 100% win rate
  from four bets is the most misleading number the table can produce.
- `calibration_table()` / `expected_calibration_error()` check whether stated
  probabilities happen at their stated rate.

## Configuration

Everything is in `betfilter/config.py`. Modes (`standard`, `conservative`,
`very_conservative`) move **only decision thresholds** — the forecast, its
distribution, and every measured component are identical across modes.

```
WEATHERPICKS_BET_FILTER=0            # disable the layer entirely
WEATHERPICKS_BET_FILTER_MODE=standard|conservative|very_conservative
```

## The realistic path to a higher win rate

Not threshold tuning. In order of expected value:

1. **Bet later in the day.** Once the observed high is within 1–2°F of the
   final high, the conditional distribution collapses and the bucket
   probability becomes genuinely high. `TemperatureDistribution` already
   supports this through `observed_high_floor_f`; what is missing is
   per-station hourly heating curves built from real observation history
   (`observation.build_heating_curve` is the hook, currently unfed).
2. **Sharpen the forecast.** σ ≈ 2.2°F is the binding constraint. This needs
   the fixed-issuance NBM GRIB archive the model card already identifies as
   the release blocker — not a better regressor on the current data.
3. **Log every decision, including passes,** and re-run the threshold sweep
   against real outcomes rather than the held-out proxy used here.
