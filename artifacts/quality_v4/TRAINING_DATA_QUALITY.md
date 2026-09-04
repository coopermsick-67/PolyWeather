# Training-data quality report

- Grain: one station x local target date x fixed 24-hour archived forecast lead
- Coverage: 13,879 rows, 2024-10-08 to 2026-09-01
- Duplicate station/date rows: 0
- TMAX range: -9.0 to 118.0 °F
- NBM baseline range: -9.0 to 116.2 °F
- Implausible labels/baselines: 0 / 0
- Rows missing any core-model Tmax: 0
- Forecast lead days present: [1] (unsupported: [])
- All-missing numeric columns excluded by training: 6
- Profiles below 90% hourly availability: {'ncep_nbm_conus': 20, 'ncep_hrrr_conus': 40, 'ncep_gfs_seamless': 0}

## Station coverage

| station | rows | first date | last date | missing Tmax | missing NBM |
|---|---:|---|---|---:|---:|
| KATL | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KAUS | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KBOS | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KDCA | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KDEN | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KDFW | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KHOU | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KLAS | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KLAX | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KMDW | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KMIA | 693 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KMSP | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KMSY | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KNYC | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KOKC | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KPHL | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KPHX | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KSAT | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KSEA | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |
| KSFO | 694 | 2024-10-08 | 2026-09-01 | 0 | 0 |

## Caveat

A valid row means the public archived forecast fields and NCEI daily label joined successfully. It does not prove a single frozen daily forecast issuance; that is tracked separately in shadow operation.
