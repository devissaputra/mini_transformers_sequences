# Data directory

The WDC-SILSO source CSV is intentionally not versioned in this repository.

The runner downloads `SN_m_tot_V2.0.csv` from the official SILSO service and caches it under `data/cache/`, which is gitignored. A local file can also be supplied with `--data-path`.

The full live file may change as new months are released. Research evidence is therefore frozen to definitive observations through `2026-03` and validated against model-input SHA-256 `02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b`. The full downloaded-file SHA-256 is recorded separately for provenance.

SILSO data are CC BY-NC 4.0 and require explicit attribution.
