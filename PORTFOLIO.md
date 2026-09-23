# Web Portfolio Entry — Transformer for Real Temporal Sequences

**Track:** AI Engineering  
**Difficulty:** ★★★★★  
**Dataset:** Sunspots dataset via statsmodels

Train a compact Transformer encoder to forecast real yearly sunspot activity from 24-year historical windows.

## Four-image gallery

![Cover](assets/01_cover.svg)

![Transformer forecasting pipeline](assets/02_data_pipeline.svg)

![Mini Transformer architecture](assets/03_data_or_model.svg)

![Chronological forecast results](assets/04_evaluation_or_results.svg)

**Skills:** Transformers, self-attention, temporal forecasting, PyTorch, positional embeddings

### Portfolio copy
This project applies a compact Transformer encoder to real yearly sunspot data. Each 24-year sequence is projected into 24-dimensional tokens, augmented with learned positional embeddings, processed by a four-head self-attention layer, and used to forecast the next annual observation. The current implementation reaches RMSE 33.07 and MAE 23.19 on a chronological hold-out split and explicitly documents the need to move normalization statistics to the training period only in a stricter version.
