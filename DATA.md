# Dataset Card — WDC-SILSO Monthly Sunspot Number

## Source

World Data Center SILSO, Royal Observatory of Belgium  
Dataset: Monthly mean total sunspot number, Version 2.0  
Official data service: https://www.sidc.be/SILSO/datafiles  
Direct file: https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv  
DOI: https://doi.org/10.24414/qnza-ac80

The official monthly series starts in January 1749 and is updated through the latest elapsed month.

## License and credit

SILSO states that the Sunspot Number data are licensed **CC BY-NC 4.0**. Publications using the data should explicitly credit WDC-SILSO, Royal Observatory of Belgium, Brussels, and the Version 2 DOI.

This repository contains code and generated metrics, not a redistributed copy of the SILSO source file.

## Fields

The semicolon-delimited CSV provides year, month, decimal date, monthly mean sunspot number, standard deviation, number of observations and a definitive/provisional indicator.

## Data integrity

The runner records SHA-256 of the exact downloaded source bytes. It also records the first and last definitive months and the number of provisional rows excluded.

## Definitive-only primary analysis

SILSO marks the most recent values as provisional and subject to revision. To reduce evaluation drift, the primary research protocol uses only rows whose official indicator marks them definitive. This choice is recorded in the generated manifest.

## Missing values

A sunspot value of -1 indicates unavailable data and is excluded before modeling.

## Chronology

No random forecasting split is used. All test targets occur after training and validation targets, and normalization is estimated only from the training era.
