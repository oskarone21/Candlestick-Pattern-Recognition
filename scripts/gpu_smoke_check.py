from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from scripts._bootstrap import ensure_repo_root
except ImportError:
    from _bootstrap import ensure_repo_root

ensure_repo_root()

from candlestick.config import load_config
from candlestick.domain import MODEL_TCN
from candlestick.models.registry import predict_model_proba, train_model
from candlestick.models.torch_common import runtime_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny GPU/CUDA smoke check for the sequence models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    parser.add_argument("--model", default=MODEL_TCN, choices=["lstm", MODEL_TCN])
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.config_override, args.set_overrides)
    summary = runtime_summary(cfg)

    if args.require_cuda and summary["selected_device"] != "cuda":
        raise RuntimeError(
            f"CUDA smoke check failed: requested CUDA validation but selected device was {summary['selected_device']}."
        )

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(48, 80, 5)).astype(np.float32)
    y_train = np.asarray([0] * 24 + [1] * 24, dtype=np.int64)
    X_val = rng.normal(size=(16, 80, 5)).astype(np.float32)
    y_val = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)

    params = {"epochs": 1, "batch_size": 8, "early_stopping_patience": 1}
    model = train_model(
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        seed=42,
        params=params,
    )
    probs = predict_model_proba(args.model, model, X_val)

    payload = {
        "runtime_summary": summary,
        "model": args.model,
        "validation_probability_mean": float(probs.mean()),
        "validation_probability_std": float(probs.std()),
        "validation_probability_count": int(len(probs)),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
