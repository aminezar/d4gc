# Retrieval evaluation

The evaluation contains 43 questions in English, French, Spanish and Arabic: 22 for threshold tuning and 21 held out for testing. It uses the full challenge collection; the sample documents demonstrate functionality and do not reproduce these metrics.

## Held-out results

| Measure | Hybrid search | Keyword baseline |
|---|---:|---:|
| Expected document retrieved in the top five, without refusal | 9/9 | 8/9 |
| Answerable questions receiving results | 9/9 | 9/9 |
| Unanswerable questions correctly refused | 9/12 | 4/12 |
| Unanswerable questions receiving results | 3/12 | 8/12 |

Hybrid search uses the bundle's stored thresholds: `T_vec=0.645` and `T_kw=1.01`. The keyword baseline uses `T_kw=0.4`, selected on the tuning split. Environment versions, bundle metadata and aggregate counts are recorded in [results.json](results.json).

## Run the evaluation

With a full collection bundle available:

```sh
# Evaluate stored thresholds without changing the bundle
HF_HUB_OFFLINE=1 uv run python eval/run.py --bundle bundles/full.sqlite

# Compare thresholds tuned on the tuning split, without saving
HF_HUB_OFFLINE=1 uv run python eval/run.py --bundle bundles/full.sqlite --tune

# Tune and save thresholds
HF_HUB_OFFLINE=1 uv run python eval/run.py --bundle bundles/full.sqlite --write
```

For `full.sqlite`, `--write` also updates `eval/thresholds.json`. Thresholds depend on the collection and should be evaluated before use.

## Limits

Nine answerable and twelve unanswerable held-out questions are a small test. The results do not establish clinical safety or general cross-language reliability. Returned passages can be irrelevant or outdated, and relevant passages can be refused.

The challenge dataset is not included. The evaluated bundle does not record the original passage-embedding runtime, so its exact build environment cannot be reconstructed. New bundles record the checkpoint revision and encoding-runtime signature. These results measure retrieval quality, not latency or memory use.
