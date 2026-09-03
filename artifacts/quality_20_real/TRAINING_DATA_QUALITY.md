# Training-data quality report

- Grain: one station x local target date x fixed 24-hour archived forecast lead
- Coverage: 13,439 rows, 2024-10-08 to 2026-08-10
- Duplicate station/date rows: 0
- TMAX range: -9.0 to 118.0 °F
- NBM baseline range: -9.0 to 116.2 °F
- Implausible labels/baselines: 0 / 0
- NBM profiles below 90% hourly availability: 20

## Station coverage

| station | rows | first date | last date | missing Tmax | missing NBM |
|---|---:|---|---|---:|---:|
| KATL | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KAUS | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KBOS | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KDCA | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KDEN | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KDFW | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KHOU | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KLAS | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KLAX | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMDW | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMIA | 671 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMSP | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KMSY | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KNYC | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KOKC | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KPHL | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KPHX | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KSAT | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KSEA | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |
| KSFO | 672 | 2024-10-08 | 2026-08-10 | 0 | 0 |

## Caveat

A valid row means the public archived forecast fields and NCEI daily label joined successfully. It does not prove a single frozen daily forecast issuance; that is tracked separately in shadow operation.
