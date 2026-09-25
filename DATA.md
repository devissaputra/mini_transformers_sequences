# Dataset Card — WDC-SILSO Monthly Sunspot Number

## Source
World Data Center SILSO, Royal Observatory of Belgium.  
Dataset: Monthly mean total sunspot number, Version 2.0.  
Official data service: https://www.sidc.be/SILSO/datafiles  
Direct file used by the runner: https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv

SILSO asks users to credit the source and provides DOI-based citation guidance. See https://www.sidc.be/SILSO/infosnmtot and DOI https://doi.org/10.24414/qnza-ac80.

## License/use boundary
The SILSO data service specifies non-commercial attribution terms. This repository contains code, not a redistributed copy of the dataset. Users should verify current SILSO terms before redistribution or commercial use.

## Fields used
The semicolon-delimited source contains calendar year, month, decimal date and monthly mean sunspot number among additional metadata. The runner uses decimal date and the monthly mean total sunspot number.

## Data integrity
The raw response bytes are hashed with SHA-256 and the hash is written into the results. This allows a paper or later replication to state exactly which changing upstream snapshot was used.

## Missing/sentinel handling
Rows with negative monthly activity values are treated as unavailable sentinels and excluded.

## Evaluation chronology
No random train/test split is used. Test windows come strictly after all model-development windows.
