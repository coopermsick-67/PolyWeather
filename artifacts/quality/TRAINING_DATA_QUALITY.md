# Training-data quality report

- Grain: one station x local target date x fixed 24-hour archived forecast lead
- Coverage: 3,359 rows, 2024-10-08 to 2026-08-10
- Duplicate station/date rows: 0
- TMAX range: 2.0 to 100.0 °F
- NBM baseline range: 3.6 to 101.5 °F
- Implausible labels/baselines: 0 / 0
- NBM profiles below 90% hourly availability: 5

## Station coverage

| station | rows | first date | last date | missing Tmax | missing NBM |
|---|---:|---|---|---:|---:|
| KLAX | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMDW | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMIA | 671 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KNYC | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KSFO | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |

## Caveat

A valid row means the public archived forecast fields and NCEI daily label joined successfully. It does not prove a single frozen daily forecast issuance; that is tracked separately in shadow operation.
