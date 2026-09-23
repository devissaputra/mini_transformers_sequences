# Data

This project uses the yearly sunspot activity dataset distributed with statsmodels.

Source documentation: https://www.statsmodels.org/stable/datasets/generated/sunspots.html

The forecasting setup uses:

- 24 years of history per input window;
- the following year as the target;
- 285 total windows;
- 228 training windows;
- 57 chronological hold-out windows.

Normalization statistics are calculated from the training period only.
