# Data directory

The WDC-SILSO source CSV is intentionally not versioned in this repository.

The runner downloads `SN_m_tot_V2.0.csv` from the official SILSO service and caches it under `data/cache/`, which is gitignored. A local file can also be supplied with `--data-path`.

Each run records the source SHA-256, the number of provisional rows excluded, and the last definitive month. SILSO data are CC BY-NC 4.0 and require attribution.
