# Inference

These commands perform model loading and prediction only; dataset-specific metrics
belong under `benchmarks/`.

```bash
python inference/predict_te.py --model final_tdc --input INPUT.parquet --output te.csv
python inference/predict_half_life.py --input INPUT.parquet --output half_life.csv
python inference/extract_embeddings.py --input INPUT.parquet --output embeddings.pkl
```
