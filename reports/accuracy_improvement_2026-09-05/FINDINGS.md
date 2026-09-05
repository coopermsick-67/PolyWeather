# Can WeatherPicks hit an 85% win rate?

**Measured answer: yes on one contract type, no on the other two.**

Everything below was frozen on the validation window (2026-05-05 → 2026-07-03,
1,200 rows) and then measured exactly once on the held-out test window
(2026-07-04 → 2026-09-01, 1,199 rows). Thresholds were never tuned on test.
Confidence intervals are weekly-block bootstraps, because forecasts made on
the same day across cities share one weather pattern and an i.i.d. interval
would be far too narrow.

Reproduce with:

```bash
python scripts/evaluate_selective_win_rate.py
```

## Result

| Contract | Test win rate | 95% CI | Coverage | ≥85%? |
|---|---|---|---|---|
| 2°F bracket (exact bucket) | 38.0% | 34.8–41.2% | 100% | **No** |
| Point forecast within 2°F | 68.3% | 66.3–70.4% | 100% | **No** |
| Threshold `≥ forecast − 3°F` | **91.8%** | 90.4–93.1% | 100% | **Yes** |
| Threshold `≤ forecast + 3°F` | **93.2%** | 91.8–94.7% | 100% | **Yes** |

## Why selectivity could not rescue the first two

The obvious lever is to bet only when the forecast looks confident. It was
tried properly — a confidence model fit on validation over ensemble spread,
inter-model disagreement, NBM position, and lead time — and it is not enough:

- **Within 2°F**: the best achievable validation win rate at ≥5% coverage was
  **75.7%**. No cutoff reached 85%, so nothing was carried to test.
- **Exact 2°F bucket**: best achievable was **45.5%**. Not close.

This is a skill ceiling, not a tuning failure. Held-out MAE is 1.71°F, so the
predictive distribution has σ ≈ 2.3°F. Landing inside a *specific* 2°F window
85% of the time would require σ ≈ 0.5°F — roughly four times the skill any
operational day-ahead guidance has. Raising a threshold cannot manufacture it.

A raw decile sweep on ensemble spread says the same thing from the other
direction: even the most-agreeing 10% of city-days came in at 71.9% within 2°F
on test, and 40.0% on the bucket.

## Why threshold contracts do reach it

A bracket asks the forecast to be nearly exact. A threshold contract only asks
it to land on the right side of a line, and pushing that line away from the
point forecast buys win rate directly:

| Margin | val ≥ | val ≤ | test ≥ | test ≤ |
|---|---|---|---|---|
| 1°F | 66.2% | 72.7% | 72.0% | 66.9% |
| 2°F | 79.6% | 86.0% | 84.9% | 83.4% |
| **3°F** | **88.0%** | **93.6%** | **91.8%** | **93.2%** |
| 4°F | 92.6% | 96.6% | 95.2% | 97.1% |
| 5°F | 95.8% | 98.6% | 97.2% | 99.0% |

3°F is the smallest margin clearing 85% on **both** sides in validation, and it
held up on test with room to spare. 2°F is borderline and should not be relied
on: it failed the ≥ side in validation (79.6%).

## The caveat that matters more than the number

**A 92% win rate here is not a 92% edge, and it is not profit.**

Anyone competent prices a contract that settles 92% of the time at roughly 92
cents. Winning 92% of the time at those odds is break-even before fees and
losing after them. The measurement above establishes that the *forecast* is
well enough calibrated to know which side of a 3°F line the high will fall
on — it says nothing about whether the market is offering that at a price
worth taking. Turning this into profit requires the other half of the
analysis: contract pricing, which is not modelled anywhere in this repository.

Two further limits on the evidence itself:

- **Retrospective, not prospective.** This archive was inspected during
  development and the issuance times do not match live operation. It is
  research evidence, not a betting track record.
- **No live ledger exists.** `verified_operational_win_rate` is still `null`.
  Nothing here has been confirmed against money at risk.

## What was changed in code

`src/polyweather/betfilter/threshold.py` implements threshold contracts
against the existing calibrated distribution: `win_probability` for a given
line, and `contract_for_target` to solve for the nearest line meeting a
target confidence. It applies the same half-degree settlement rounding the
bracket path already gets right (a "≥95F" contract is won by a true high of
94.5F), and it collapses correctly when today's observed high has already
settled one side. Solved margins for 85–95% confidence land at 2.5–3.5°F,
which independently agrees with the empirical table above.

The model itself was **not** changed and **not** promoted. The GFS-ablation
and alternative-regressor candidates both scored worse than the incumbent on
validation (61.4% and 61.8% within 2°F, against 65.6%), so the incumbent
stands.
