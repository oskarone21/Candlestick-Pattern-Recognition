# Explainability Pipeline

`scripts/render_explainability.py` generates static SHAP artifacts for the champion models selected by `scripts/run_experiment_suite.py`.

## Inputs

The script expects a completed run with these saved artifacts:

- `outputs/metrics/<run_name>/champions.csv`
- `outputs/metrics/<run_name>/<pattern>/split_metadata.csv`
- `outputs/checkpoints/<run_name>/<pattern>/<model>.bin`
- `outputs/datasets/<run_name>/<pattern>/dataset.npz`
- `outputs/datasets/<run_name>/<pattern>/metadata.csv`

## Command

```bash
python scripts/render_explainability.py \
  --config configs/config.yaml \
  --config-override configs/overrides/production_repro.yaml \
  --set project.run_name=my_run
```

Useful optional flags:

- `--pattern head_shoulders`
- `--explain-split test`
- `--background-split train`
- `--explain-size 32`
- `--background-size 64`
- `--top-features 15`
- `--max-waterfalls 3`

## Outputs

Run-level output root:

- `outputs/explainability/<run_name>/explainability_summary.json`

Per-pattern outputs:

- `summary.json`
- `feature_group_importance.csv`
- `flat_feature_importance.csv`
- `time_profile.csv`
- `top_window_features.csv`
- `sample_scores.csv`
- `shap_values.npz`
- `feature_group_importance.png`
- `temporal_heatmap.png`
- `top_window_features.png`
- `sample_*_waterfall.png`

## Notes

- Linear and tree champions use SHAP explainers matched to the saved model family.
- Sequence champions use gradient-based SHAP on the saved PyTorch checkpoint.
- Background and explained samples are subsampled from the saved split metadata to keep artifact generation manageable.
