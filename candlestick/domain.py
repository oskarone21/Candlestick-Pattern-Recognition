from __future__ import annotations

from enum import Enum


class PatternName(str, Enum):
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class PatternEventReason(str, Enum):
    NEAR_MISS = "near_miss"
    CONFIRMED_BREAKOUT = "confirmed_breakout"
    FAILED_BREAKOUT = "failed_breakout"
    PARTIAL = "partial"


class ExitReason(str, Enum):
    TIME_STOP = "time_stop"
    STOP = "stop"
    TAKE_PROFIT = "tp"


DEFAULT_INSTRUMENT = "SPY"
DEFAULT_TIMEZONE_IN_DATA = "America/Denver"
DEFAULT_WORKING_TIMEZONE = "America/New_York"
DEFAULT_MARKET_CALENDAR = "XNYS"
DEFAULT_SESSION_START = "09:30"
DEFAULT_SESSION_END = "16:00"
DEFAULT_EARLY_CLOSE_END = "13:00"
DEFAULT_RUN_NAME = "pattern_suite"
DEFAULT_BASE_TIMEFRAME = "1min"
DEFAULT_TARGET_TIMEFRAME = "15min"
DEFAULT_PROCESSED_15M_PATH = "data/processed/spy_15m.csv"

MODEL_LOGREG = "logreg"
MODEL_HGB = "hgb"
MODEL_LSTM = "lstm"
MODEL_TCN = "tcn"
MODEL_TRANSFORMER = "transformer"

DEFAULT_ALLOWED_PATTERNS = tuple(pattern.value for pattern in PatternName)
DEFAULT_CANDIDATE_MODELS = (MODEL_LOGREG, MODEL_LSTM, MODEL_TCN)
SEQUENCE_MODEL_NAMES = (MODEL_LSTM, MODEL_TCN, MODEL_TRANSFORMER)
CALIBRATED_MODEL_NAMES = (MODEL_LSTM, MODEL_TCN)

METRIC_ACCURACY = "accuracy"
METRIC_PRECISION = "precision"
METRIC_RECALL = "recall"
METRIC_F1 = "f1"
METRIC_F2 = "f2"
METRIC_PR_AUC = "pr_auc"
SUPPORTED_SELECTION_METRICS = (
    METRIC_ACCURACY,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_F1,
    METRIC_F2,
    METRIC_PR_AUC,
)

SELECTION_SPLIT_VAL = "val"
UNKNOWN_PATTERN = "unknown"

GALLERY_BUCKET_TP = "tp"
GALLERY_BUCKET_FP = "fp"
GALLERY_BUCKET_FN = "fn"
GALLERY_BUCKETS = (GALLERY_BUCKET_TP, GALLERY_BUCKET_FP, GALLERY_BUCKET_FN)

COLUMN_SYMBOL = "symbol"
COLUMN_TS_EVENT = "ts_event"
COLUMN_OPEN = "open"
COLUMN_HIGH = "high"
COLUMN_LOW = "low"
COLUMN_CLOSE = "close"
COLUMN_VOLUME = "volume"
OHLCV_COLUMNS = (
    COLUMN_OPEN,
    COLUMN_HIGH,
    COLUMN_LOW,
    COLUMN_CLOSE,
    COLUMN_VOLUME,
)
