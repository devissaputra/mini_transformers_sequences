# Dataset Card — WDC-SILSO Monthly Sunspot Number

## Source

**World Data Center SILSO, Royal Observatory of Belgium**  
Dataset: Monthly mean total Sunspot Number, Version 2.0  
Official file: `SN_m_tot_V2.0.csv`  
DOI: `10.24414/qnza-ac80`  
License: **CC BY-NC 4.0**

The official CSV uses semicolon delimiters. Its final field is the definitive/provisional marker: `1` denotes a definitive monthly value and `0` denotes a provisional value.

## Frozen study era

The upstream source is continuously extended, so the full downloaded file itself is not a stable scientific identity.

This bundle therefore freezes the **model input** to:

- first study month: `1749-01`;
- final study month: `2026-03`;
- study rows: `3327`;
- expected study-input SHA-256: `02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b`.

The fingerprint is computed from the exact ordered `YYYY-MM;sunspot-value` sequence used by the models. Any change inside this frozen history fails the experiment. Rows after March 2026 are excluded even if SILSO later marks them definitive.

The runner additionally records the complete downloaded-file SHA-256, latest available month, latest definitive month, number of provisional rows and number of post-cutoff rows. Those fields document the live source snapshot but do not redefine the study.

## Missing values and integrity

A sunspot value below zero is treated as unavailable and removed before the study freeze. The runner requires explicit numeric definitive/provisional flags and verifies that the frozen study forms one uninterrupted monthly sequence.

## Forecast chronology

Target-date boundaries are defined on the frozen raw monthly timeline, not separately from each supervised-window matrix. Consequently:

- every forecast horizon uses the same validation-start and test-start target months;
- every context-length sensitivity condition uses the same target months;
- the scaler uses only observations through the final training target.

No random forecasting split is used.

## Licensing and attribution

The source data are not redistributed in this repository. Downstream users should respect SILSO's CC BY-NC 4.0 terms and explicitly credit WDC-SILSO, Royal Observatory of Belgium, Brussels, DOI `10.24414/qnza-ac80`.

## Limitations

The Sunspot Number is a scientific index with long-term revisions and physical structure. This benchmark uses one frozen historical revision and does not constitute an operational solar-activity forecasting product.
