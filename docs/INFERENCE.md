# InvaRNA inference

Only InvaRNA-native Stage1, Stage2, and half-life checkpoints are registered. External
baseline implementations, dependencies, and weights are excluded.

Input CSV/parquet files require `mrna` and `utr5_size`. Sequences are aligned with CDS at
position 1,000 and padded or truncated to 10,000 nt.

```bash
python inference/predict_te.py \
  --model w0 --input input.parquet --output predictions.csv

torchrun --standalone --nproc_per_node=4 inference/predict_te.py \
  --model w0 --input input.parquet --output predictions.csv
```

Python API:

```python
from invarna.evaluation import InvaRNAPredictor

predictor = InvaRNAPredictor("w0", device="cuda:0")
scores = predictor.predict(["ACGT..."], utr5_sizes=[100])
```
