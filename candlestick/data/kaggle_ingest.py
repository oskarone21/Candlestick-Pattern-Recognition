from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from candlestick.config import ensure_dir
from candlestick.domain import (
    COLUMN_SYMBOL,
    COLUMN_TS_EVENT,
    DEFAULT_INSTRUMENT,
)

DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "ts_event": ["ts_event", "timestamp", "date", "datetime", "time"],
    "symbol": ["symbol", "ticker", "instrument"],
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "close": ["close", "Close"],
    "volume": ["volume", "Volume"],
}


class IngestError(RuntimeError):
    """Raised when dataset ingestion or schema normalization fails."""


def _resolve_csv_from_download_root(download_root: Path, requested_path: str | None) -> Path:
    """Resolve CSV path from Kaggle download directory."""
    if requested_path:
        candidate = download_root / requested_path
        if candidate.exists() and candidate.is_file():
            return candidate

    csv_files = sorted(download_root.rglob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    if requested_path and len(csv_files) > 1:
        options = ", ".join(str(p.relative_to(download_root)) for p in csv_files[:10])
        raise IngestError(
            "Configured data_source.kaggle_file_path does not exist inside downloaded dataset. "
            f"Requested '{requested_path}'. Example available CSVs: {options}"
        )

    if not csv_files:
        raise IngestError(
            "No CSV files were found in the downloaded Kaggle dataset directory. "
            "Check data_source.kaggle_dataset and dataset contents."
        )

    options = ", ".join(str(p.relative_to(download_root)) for p in csv_files[:10])
    raise IngestError(
        "Multiple CSV files found in the Kaggle dataset; set data_source.kaggle_file_path explicitly. "
        f"Example available CSVs: {options}"
    )


def _column_lookup(df: pd.DataFrame, preferred_name: str | None, aliases: list[str]) -> str | None:
    if preferred_name and preferred_name in df.columns:
        return preferred_name

    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        candidate = lowered.get(alias.lower())
        if candidate is not None:
            return candidate
    return None


def normalize_intraday_schema(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Normalize raw dataset columns to project-standard schema."""
    timestamp_cfg = cfg["data_source"].get("timestamp", {})
    col_cfg = cfg["data_source"].get("columns", {})

    rename_map: dict[str, str] = {}
    for canonical, aliases in DEFAULT_COLUMN_ALIASES.items():
        preferred = None
        if canonical == "ts_event":
            preferred = timestamp_cfg.get("column")
        else:
            preferred = col_cfg.get(canonical)

        raw_col = _column_lookup(df, preferred, aliases)
        if raw_col is None and canonical not in {"symbol"}:
            raise IngestError(
                f"Required column '{canonical}' not found in input columns: {list(df.columns)}"
            )
        if raw_col is not None:
            rename_map[raw_col] = canonical

    normalized = df.rename(columns=rename_map).copy()

    if COLUMN_SYMBOL not in normalized.columns:
        normalized[COLUMN_SYMBOL] = cfg["data_source"].get("instrument", DEFAULT_INSTRUMENT)

    keep_cols = ["symbol", "ts_event", "open", "high", "low", "close", "volume"]
    extra_cols = [c for c in ["barCount", "barcount", "average", "avg"] if c in normalized.columns]
    normalized = normalized[keep_cols + extra_cols]

    if "Unnamed: 0" in normalized.columns:
        normalized = normalized.drop(columns=["Unnamed: 0"])

    normalized[COLUMN_TS_EVENT] = pd.to_datetime(normalized[COLUMN_TS_EVENT], errors="coerce")
    if normalized[COLUMN_TS_EVENT].isna().all():
        raise IngestError("All timestamps failed to parse. Check data_source.timestamp.column mapping.")

    normalized = normalized.sort_values([COLUMN_SYMBOL, COLUMN_TS_EVENT]).reset_index(drop=True)
    return normalized


def load_kaggle_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load dataset via kagglehub and return a normalized DataFrame."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise IngestError(
            "kagglehub is not installed. Install it with `pip install kagglehub[pandas-datasets]`."
        ) from exc

    data_cfg = cfg["data_source"]
    dataset_handle = data_cfg.get("kaggle_dataset")
    if not dataset_handle:
        raise IngestError("Missing data_source.kaggle_dataset in config.")

    file_path = data_cfg.get("kaggle_file_path")

    pandas_kwargs = data_cfg.get("kaggle_pandas_kwargs", {})

    load_fn = getattr(kagglehub, "load_dataset", None)
    if load_fn is None:
        load_fn = getattr(kagglehub, "dataset_load", None)

    if load_fn is None:
        raise IngestError("kagglehub does not expose load_dataset/dataset_load in this environment.")

    df = None
    last_error: Exception | None = None

    if file_path:
        try:
            try:
                df = load_fn(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_handle,
                    path=file_path,
                    pandas_kwargs=pandas_kwargs,
                )
            except TypeError:
                df = load_fn(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_handle,
                    file_path=file_path,
                    pandas_kwargs=pandas_kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if df is None:
        try:
            download_root = Path(kagglehub.dataset_download(dataset_handle))
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            if "401" in detail or "403" in detail:
                raise IngestError(
                    "Kaggle authentication/authorization failed. "
                    "Please ensure ~/.kaggle/kaggle.json is present and has access to this dataset."
                ) from exc
            if "404" in detail:
                raise IngestError(
                    "Kaggle dataset was not found. Verify data_source.kaggle_dataset "
                    f"(current: '{dataset_handle}')."
                ) from exc
            raise IngestError(
                "Failed to download Kaggle dataset. "
                f"Original error: {detail}"
            ) from exc

        try:
            csv_path = _resolve_csv_from_download_root(download_root, file_path)
        except IngestError:
            try:
                download_root = Path(kagglehub.dataset_download(dataset_handle, force_download=True))
                csv_path = _resolve_csv_from_download_root(download_root, file_path)
            except Exception as exc:  # noqa: BLE001
                raise IngestError(
                    "Kaggle dataset cache appears inconsistent and force-download recovery failed. "
                    f"Original error: {exc}"
                ) from exc

        try:
            df = pd.read_csv(csv_path, **(pandas_kwargs or {}))
        except Exception as exc:  # noqa: BLE001
            raise IngestError(f"Failed reading CSV '{csv_path}': {exc}") from exc

    if last_error is not None and (not isinstance(df, pd.DataFrame) or df.empty):
        raise IngestError(
            "Kaggle load failed and fallback did not return a usable DataFrame. "
            f"Original load error: {last_error}"
        ) from last_error

    if isinstance(df, dict):
        raise IngestError(
            "Dataset loader returned multiple tables. Provide a single CSV file in data_source.kaggle_file_path."
        )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise IngestError("Kaggle loader returned an empty or invalid DataFrame.")

    return normalize_intraday_schema(df, cfg)


def save_raw_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist normalized raw data to CSV and return output path."""
    out = Path(output_path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    return out
