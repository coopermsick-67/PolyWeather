# Training-data quality report

- Grain: one station x local target date x fixed 24-hour archived forecast lead
- Coverage: 672 rows, 2024-10-08 to 2026-08-10
- Duplicate station/date rows: 0
- TMAX range: 30.0 to 100.0 °F
- NBM baseline range: 32.1 to 96.0 °F
- Implausible labels/baselines: 0 / 0
- NBM profiles below 90% hourly availability: 1

## Station coverage

| station | rows | first date | last date | missing Tmax | missing NBM |
|---|---:|---|---|---:|---:|
| KATL | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |

## Caveat

A valid row means the public archived forecast fields and NCEI daily label joined successfully. It does not prove a single frozen daily forecast issuance; that is tracked separately in shadow operation.
