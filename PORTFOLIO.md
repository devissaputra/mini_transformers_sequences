# Transformer Forecasting on WDC-SILSO

This multi-horizon forecasting study asks whether a compact Transformer actually earns its complexity over persistence, seasonal-naive, ridge, and gradient-boosted baselines on a long scientific time series. It evaluates several forecast horizons, seed stability, context-length sensitivity, early-versus-late test behavior, and performance during high solar activity.

The Transformer has lower mean error than the simpler baselines at the 1- and 6-month horizons under the recorded protocol. The 12-month result is less stable: its three-seed mean MAE is 24.206, versus 23.973 for histogram gradient boosting, despite seed 42 favoring the Transformer. The seed-42 intervals against that baseline also cross zero at 6 and 12 months. This qualifies any broad claim that the Transformer is consistently superior.

See [CALCULATIONS.md](CALCULATIONS.md) for evidence and verification scope.
