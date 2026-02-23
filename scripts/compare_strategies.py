"""
多策略回測比較腳本

用現有的 walk-forward 模型（2023W01~2025W52）跑多種交易策略，
加入真實手續費和證交稅，比較各策略扣除成本後的超額收益。

策略類型：
  1. TopK — 每天完全重建持倉（基準）
  2. TopK-Drop — 每天只替換掉出 Top-K 的 D 支（降低 turnover）
  3. Hold Period — 買入後持有至少 H 天
  4. Threshold Exit — 排名掉出 Top-M 才賣
  5. Score Weighted — 按模型分數加權

投資模式：
  A. 複利模式（Compound）— 100 萬初始資金，收益再投入
  B. 定額模式（Fixed）— 每支 5 萬固定投入
"""

import json
import pickle
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# 確保能 import src 模組
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ──
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"

# ── 交易成本（玉山證券無 VIP，電子下單 6 折） ──
COMMISSION_RATE = 0.001425       # 0.1425%
COMMISSION_DISCOUNT = 0.6        # 6 折
EFFECTIVE_COMMISSION = COMMISSION_RATE * COMMISSION_DISCOUNT  # 0.0855%
MIN_COMMISSION = 20              # 最低手續費 20 元
TRANSACTION_TAX = 0.003          # 證交稅 0.3%（賣出才收）

# ── 投資參數 ──
INITIAL_CAPITAL = 1_000_000      # 複利模式初始資金
FIXED_AMOUNT_PER_STOCK = 50_000  # 定額模式每支金額

# ── 回測範圍 ──
START_MODEL_WEEK = "2023W01"
END_MODEL_WEEK = "2025W51"       # predict_week = 2023W02 ~ 2025W52


# ══════════════════════════════════════════════════════════════
# 資料準備
# ══════════════════════════════════════════════════════════════

def export_qlib_data(start_date: date, end_date: date) -> None:
    """用 QlibExporter 導出 qlib 資料"""
    from src.repositories.database import get_session
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    session = get_session()
    try:
        exporter = QlibExporter(session)
        config = ExportConfig(
            start_date=start_date,
            end_date=end_date,
            output_dir=QLIB_DATA_DIR,
        )
        result = exporter.export(config)
        print(f"  Exported: {result.stocks_exported} stocks, {result.fields_per_stock} fields, "
              f"{result.calendar_days} days")
        if result.errors:
            print(f"  Warnings: {len(result.errors)} errors")
    finally:
        session.close()


def init_qlib() -> None:
    """初始化 qlib"""
    import qlib
    from qlib.config import REG_CN

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)


def get_instruments() -> list[str]:
    """從 instruments 檔案讀取股票清單"""
    instruments_file = QLIB_DATA_DIR / "instruments" / "all.txt"
    with open(instruments_file) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def load_model(model_name: str) -> tuple:
    """載入模型 (model, factors, config)"""
    model_dir = MODELS_DIR / model_name
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "factors.json") as f:
        factors = json.load(f)
    config = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    return model, factors, config


def process_inf(df: pd.DataFrame) -> pd.DataFrame:
    """處理無窮大值"""
    df = df.copy()
    for col in df.columns:
        mask = np.isinf(df[col])
        if mask.any():
            col_mean = df.loc[~mask, col].mean()
            df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
    return df


def zscore_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """每日截面標準化"""
    return df.groupby(level="datetime", group_keys=False).apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


# ══════════════════════════════════════════════════════════════
# 模型收集 & 預測
# ══════════════════════════════════════════════════════════════

@dataclass
class WeekModelInfo:
    predict_week: str
    model_week: str
    model_name: str
    is_fallback: bool


def parse_week_id(week_id: str) -> tuple[int, int]:
    return int(week_id[:4]), int(week_id[5:])


def get_next_week_id(week_id: str) -> str:
    year, week = parse_week_id(week_id)
    friday = date.fromisocalendar(year, week, 5)
    next_monday = friday + timedelta(days=3)
    iso_year, iso_week, _ = next_monday.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def get_previous_week_id(week_id: str) -> str:
    year, week = parse_week_id(week_id)
    monday = date.fromisocalendar(year, week, 1)
    prev_friday = monday - timedelta(days=3)
    iso_year, iso_week, _ = prev_friday.isocalendar()
    return f"{iso_year}W{iso_week:02d}"


def compare_week_ids(a: str, b: str) -> int:
    ya, wa = parse_week_id(a)
    yb, wb = parse_week_id(b)
    if ya != yb:
        return -1 if ya < yb else 1
    if wa != wb:
        return -1 if wa < wb else 1
    return 0


def get_weeks_in_range(start: str, end: str) -> list[str]:
    weeks = []
    current = start
    while compare_week_ids(current, end) <= 0:
        weeks.append(current)
        current = get_next_week_id(current)
    return weeks


def get_week_date_range(week_id: str) -> tuple[date, date]:
    year, week = parse_week_id(week_id)
    monday = date.fromisocalendar(year, week, 1)
    friday = date.fromisocalendar(year, week, 5)
    return monday, friday


def collect_models(start_week: str, end_week: str) -> list[WeekModelInfo]:
    """收集回測範圍內的模型（含 fallback）"""
    all_weeks = get_weeks_in_range(start_week, end_week)

    # 找出所有存在的模型目錄
    existing_models: dict[str, str] = {}
    for d in MODELS_DIR.iterdir():
        if d.is_dir() and (d / "model.pkl").exists():
            # 目錄名格式: 2024W01-8d9fdb
            name = d.name
            parts = name.split("-")
            if len(parts) >= 1:
                week_id = parts[0]
                existing_models[week_id] = name

    result = []
    for week_id in all_weeks:
        predict_week = get_next_week_id(week_id)

        if week_id in existing_models:
            result.append(WeekModelInfo(
                predict_week=predict_week,
                model_week=week_id,
                model_name=existing_models[week_id],
                is_fallback=False,
            ))
        else:
            # fallback: 往前找最近的模型
            current = get_previous_week_id(week_id)
            found = False
            for _ in range(10):
                if current in existing_models:
                    result.append(WeekModelInfo(
                        predict_week=predict_week,
                        model_week=current,
                        model_name=existing_models[current],
                        is_fallback=True,
                    ))
                    found = True
                    break
                current = get_previous_week_id(current)
            if not found:
                print(f"  Warning: no model for {week_id}, skipping")

    return result


def preload_data(
    factors: list[dict],
    instruments: list[str],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """預載入所有特徵和價格資料"""
    from qlib.data import D

    print("  Loading features...")
    fields = [f["expression"] for f in factors]
    names = [f["name"] for f in factors]

    features = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d"),
    )
    if not features.empty:
        features.columns = names
    print(f"  Features: {len(features):,} rows")

    # 價格（往後延伸 10 天確保最後一週能計算收益）
    extended_end = end_date + timedelta(days=10)
    print("  Loading prices...")
    prices = D.features(
        instruments=instruments,
        fields=["$close"],
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=extended_end.strftime("%Y-%m-%d"),
    )
    if not prices.empty:
        prices.columns = ["close"]
    print(f"  Prices: {len(prices):,} rows")

    return features, prices


def predict_week(
    model,
    factors: list[dict],
    features_cache: pd.DataFrame,
    predict_start: date,
    predict_end: date,
) -> pd.DataFrame:
    """對指定週期進行預測，返回 DataFrame (index=date, columns=stock_id, values=score)"""
    start_str = predict_start.strftime("%Y-%m-%d")
    end_str = predict_end.strftime("%Y-%m-%d")

    df = features_cache.loc[
        (features_cache.index.get_level_values("datetime") >= start_str) &
        (features_cache.index.get_level_values("datetime") <= end_str)
    ].copy()

    names = [f["name"] for f in factors]
    available = [n for n in names if n in df.columns]
    if available:
        df = df[available]

    if df.empty:
        return pd.DataFrame()

    df = process_inf(df)
    df = zscore_by_date(df)
    df = df.fillna(0)

    predictions = model.predict(df.values)
    pred_series = pd.Series(predictions, index=df.index, name="score")
    pred_df = pred_series.unstack(level="instrument")
    pred_df.index = pd.to_datetime(pred_df.index.date)

    return pred_df


def generate_daily_scores(
    model_infos: list[WeekModelInfo],
    features_cache: pd.DataFrame,
) -> dict[date, pd.Series]:
    """
    對所有模型週，產生每日全股票分數。

    Returns:
        {date: pd.Series(index=stock_id, values=score)}
    """
    daily_scores: dict[date, pd.Series] = {}
    model_cache: dict[str, tuple] = {}

    for i, info in enumerate(model_infos):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(model_infos)}] Predicting {info.predict_week}...")

        # 載入模型（快取避免重複）
        if info.model_name not in model_cache:
            model_cache[info.model_name] = load_model(info.model_name)
        model, factors, _ = model_cache[info.model_name]

        predict_start, predict_end = get_week_date_range(info.predict_week)
        pred_df = predict_week(model, factors, features_cache, predict_start, predict_end)

        if pred_df.empty:
            continue

        for dt in pred_df.index:
            scores = pred_df.loc[dt].dropna()
            if not scores.empty:
                daily_scores[dt.date() if hasattr(dt, 'date') else dt] = scores

    return daily_scores


# ══════════════════════════════════════════════════════════════
# 交易成本
# ══════════════════════════════════════════════════════════════

def calc_trade_cost(amount: float, is_sell: bool) -> float:
    """計算單筆交易成本"""
    commission = max(abs(amount) * EFFECTIVE_COMMISSION, MIN_COMMISSION)
    tax = abs(amount) * TRANSACTION_TAX if is_sell else 0.0
    return commission + tax


# ══════════════════════════════════════════════════════════════
# 策略
# ══════════════════════════════════════════════════════════════

@dataclass
class DailyRecord:
    """單日記錄"""
    dt: date
    gross_return: float   # 扣費前日收益率
    net_return: float     # 扣費後日收益率
    market_return: float  # 大盤等權重收益率
    cost: float           # 當日交易成本（元）
    turnover: float       # 換手率 = 交易股數 / 持倉股數
    n_holdings: int
    n_buy: int
    n_sell: int


@dataclass
class StrategyResult:
    """策略回測結果"""
    name: str
    mode: str  # "compound" or "fixed"
    records: list[DailyRecord]
    total_cost: float


class Strategy(ABC):
    """策略基類"""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def decide(
        self,
        scores: pd.Series,
        prev_holdings: dict[str, int],  # stock -> holding_days
    ) -> tuple[list[str], list[str]]:
        """
        決定買賣。

        Args:
            scores: 當日全股票分數 (index=stock_id, values=score)
            prev_holdings: 前一日持倉 {stock_id: days_held}

        Returns:
            (to_buy, to_sell) 股票代碼列表
        """
        ...


class TopKStrategy(Strategy):
    """每天完全重建 Top-K"""

    def __init__(self, k: int):
        self.k = k

    def name(self) -> str:
        return f"TopK(K={self.k})"

    def decide(self, scores, prev_holdings):
        ranked = scores.sort_values(ascending=False)
        # tie-breaking by symbol
        ranked_df = ranked.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        ).head(self.k)
        new_set = set(ranked_df["symbol"].tolist())
        old_set = set(prev_holdings.keys())

        to_sell = [s for s in old_set if s not in new_set]
        to_buy = [s for s in ranked_df["symbol"] if s not in old_set]
        return to_buy, to_sell


class TopKDropStrategy(Strategy):
    """TopK-Drop: 每天只替換 D 支"""

    def __init__(self, k: int, d: int):
        self.k = k
        self.d = d

    def name(self) -> str:
        return f"TopKDrop(K={self.k},D={self.d})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        )
        topk_set = set(ranked_df["symbol"].head(self.k).tolist())
        old_set = set(prev_holdings.keys())

        # 掉出 Top-K 的持倉，按分數升序取最差的 D 支
        sell_candidates = [s for s in old_set if s not in topk_set]
        sell_candidates.sort(key=lambda s: scores.get(s, -999))
        to_sell = sell_candidates[:self.d]

        # 未持有但在排名內的，按分數降序取同數量
        buy_candidates = [s for s in ranked_df["symbol"] if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        # 如果持倉不足 K，補到 K
        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            extra_needed = self.k - current_count
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:extra_needed]

        return to_buy, to_sell


class HoldPeriodStrategy(Strategy):
    """TopK + 最少持有 H 天"""

    def __init__(self, k: int, h: int):
        self.k = k
        self.h = h

    def name(self) -> str:
        return f"HoldPeriod(K={self.k},H={self.h})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        )
        topk = ranked_df["symbol"].head(self.k).tolist()
        topk_set = set(topk)
        old_set = set(prev_holdings.keys())

        # 只賣出已持有 >= H 天且掉出 Top-K 的
        to_sell = [
            s for s in old_set
            if s not in topk_set and prev_holdings[s] >= self.h
        ]

        # 買入：未持有但在 Top-K 內
        buy_candidates = [s for s in topk if s not in old_set]
        slots = len(to_sell)
        to_buy = buy_candidates[:slots]

        # 如果持倉不足 K，補到 K
        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            extra_needed = self.k - current_count
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:extra_needed]

        return to_buy, to_sell


class HoldDropStrategy(Strategy):
    """HoldPeriod + TopKDrop 混合：持有 >= H 天才能賣，且每天最多賣 D 支"""

    def __init__(self, k: int, h: int, d: int):
        self.k = k
        self.h = h
        self.d = d

    def name(self) -> str:
        return f"HoldDrop(K={self.k},H={self.h},D={self.d})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        )
        topk_set = set(ranked_df["symbol"].head(self.k).tolist())
        old_set = set(prev_holdings.keys())

        # 賣出條件：持有 >= H 天 AND 掉出 Top-K
        sell_candidates = [
            s for s in old_set
            if s not in topk_set and prev_holdings[s] >= self.h
        ]
        # 按分數升序，只賣最差的 D 支
        sell_candidates.sort(key=lambda s: scores.get(s, -999))
        to_sell = sell_candidates[:self.d]

        # 買入：未持有但在排名內的，按分數降序取同數量
        buy_candidates = [s for s in ranked_df["symbol"] if s not in old_set]
        to_buy = buy_candidates[:len(to_sell)]

        # 補到 K
        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            extra_needed = self.k - current_count
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:extra_needed]

        return to_buy, to_sell


class HoldDropBottomStrategy(Strategy):
    """HoldDrop + Bottom 強制賣出：跌到全市場最差 B 名時無視持有天數強制出場"""

    def __init__(self, k: int, h: int, d: int, b: int):
        self.k = k
        self.h = h
        self.d = d
        self.b = b  # 排名倒數 B 名以內強制賣

    def name(self) -> str:
        return f"HoldDropBot(K={self.k},H={self.h},D={self.d},B={self.b})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        )
        all_symbols = ranked_df["symbol"].tolist()
        topk_set = set(all_symbols[:self.k])
        bottom_b_set = set(all_symbols[-self.b:]) if len(all_symbols) >= self.b else set()
        old_set = set(prev_holdings.keys())

        # 1. 強制賣出：持有中排名跌到倒數 B 名（無視 H 和 D）
        force_sell = [s for s in old_set if s in bottom_b_set]

        # 2. 正常 HoldDrop：掉出 TopK + 持有 >= H 天，最多賣 D 支
        regular_candidates = [
            s for s in old_set
            if s not in topk_set and s not in force_sell and prev_holdings[s] >= self.h
        ]
        regular_candidates.sort(key=lambda s: scores.get(s, -999))
        regular_sell = regular_candidates[:self.d]

        to_sell = force_sell + regular_sell

        # 買入補到 K
        new_holding_set = old_set - set(to_sell)
        buy_candidates = [s for s in all_symbols if s not in new_holding_set]
        slots = max(0, self.k - len(new_holding_set))
        to_buy = buy_candidates[:slots]

        return to_buy, to_sell


class ThresholdExitStrategy(Strategy):
    """排名掉出 Top-M 才賣"""

    def __init__(self, k: int, m: int):
        self.k = k
        self.m = m

    def name(self) -> str:
        return f"ThresholdExit(K={self.k},M={self.m})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        )
        topk = ranked_df["symbol"].head(self.k).tolist()
        topm_set = set(ranked_df["symbol"].head(self.m).tolist())
        old_set = set(prev_holdings.keys())

        # 只賣出掉出 Top-M 的（比 Top-K 更寬鬆）
        to_sell = [s for s in old_set if s not in topm_set]

        # 買入：未持有但在 Top-K 內
        buy_candidates = [s for s in topk if s not in old_set]
        slots = len(to_sell)
        to_buy = buy_candidates[:slots]

        # 補到 K
        current_count = len(old_set) - len(to_sell) + len(to_buy)
        if current_count < self.k:
            extra_needed = self.k - current_count
            remaining = [s for s in buy_candidates if s not in to_buy]
            to_buy = to_buy + remaining[:extra_needed]

        return to_buy, to_sell


class ScoreWeightedStrategy(Strategy):
    """分數加權（decide 回傳同 TopK，權重在 simulate 中處理）"""

    def __init__(self, k: int):
        self.k = k

    def name(self) -> str:
        return f"ScoreWeighted(K={self.k})"

    def decide(self, scores, prev_holdings):
        ranked_df = scores.reset_index()
        ranked_df.columns = ["symbol", "score"]
        ranked_df = ranked_df.sort_values(
            by=["score", "symbol"], ascending=[False, True]
        ).head(self.k)
        new_set = set(ranked_df["symbol"].tolist())
        old_set = set(prev_holdings.keys())

        to_sell = [s for s in old_set if s not in new_set]
        to_buy = [s for s in ranked_df["symbol"] if s not in old_set]
        return to_buy, to_sell


# ══════════════════════════════════════════════════════════════
# 回測模擬引擎
# ══════════════════════════════════════════════════════════════

def simulate(
    strategy: Strategy,
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list,
    mode: str = "compound",
) -> StrategyResult:
    """
    模擬策略執行。

    核心邏輯：
    - 追蹤持倉 {stock_id: days_held}
    - 每日計算持倉的等權重（或分數加權）收益率 = gross_return
    - 計算交易成本佔總持倉的比例，扣掉 = net_return
    - compound 模式：equity *= (1 + net_return)
    - fixed 模式：累積淨收益率，總損益 = 累積率 × 持倉市值

    Args:
        strategy: 策略物件
        daily_scores: {date: scores_series}
        close_wide: DataFrame (index=datetime, columns=stock_id, values=close)
        trading_days: 排序好的交易日列表
        mode: "compound" or "fixed"
    """
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    holdings: dict[str, int] = {}  # stock_id -> days_held
    equity = float(INITIAL_CAPITAL)  # 追蹤複利模式的資產規模
    total_cost = 0.0
    records: list[DailyRecord] = []

    score_dates = sorted(daily_scores.keys())
    is_score_weighted = isinstance(strategy, ScoreWeightedStrategy)
    k = getattr(strategy, 'k', 10)

    for pred_date in score_dates:
        # pred_date = T（分數產生日）
        # 找到 T 在交易日中的位置
        if pred_date not in day_to_idx:
            candidates = [d for d in trading_days if d <= pred_date]
            if not candidates:
                continue
            idx = day_to_idx[candidates[-1]]
        else:
            idx = day_to_idx[pred_date]

        # T+1 close 買入, T+2 close 結算
        if idx + 2 >= len(trading_days):
            continue
        t1 = trading_days[idx + 1]
        t2 = trading_days[idx + 2]

        if t1 not in close_wide.index or t2 not in close_wide.index:
            continue
        prices_t1 = close_wide.loc[t1].dropna()
        prices_t2 = close_wide.loc[t2].dropna()

        scores = daily_scores[pred_date]

        # 大盤等權重收益
        common_all = prices_t1.index.intersection(prices_t2.index)
        if len(common_all) < 2:
            continue
        all_returns = (prices_t2[common_all] - prices_t1[common_all]) / prices_t1[common_all]
        market_return = float(all_returns.mean())

        # 策略決定買賣
        to_buy, to_sell = strategy.decide(scores, holdings)
        to_sell = [s for s in to_sell if s in prices_t1.index]
        to_buy = [s for s in to_buy if s in prices_t1.index and s in prices_t2.index]

        # 執行交易：更新持倉
        for s in to_sell:
            if s in holdings:
                del holdings[s]
        for s in to_buy:
            holdings[s] = 0

        # 更新持倉天數
        for s in list(holdings.keys()):
            holdings[s] += 1

        # 計算持倉收益（T+1 close → T+2 close）
        held_stocks = [s for s in holdings if s in common_all]
        if held_stocks:
            if is_score_weighted and len(held_stocks) > 1:
                weights = {}
                total_score = 0.0
                for s in held_stocks:
                    sc = max(scores.get(s, 0), 0)
                    weights[s] = sc
                    total_score += sc
                if total_score > 0:
                    gross_return = sum(
                        (weights[s] / total_score) * float(all_returns.get(s, 0))
                        for s in held_stocks
                    )
                else:
                    gross_return = float(all_returns[held_stocks].mean())
            else:
                gross_return = float(all_returns[held_stocks].mean())
        else:
            gross_return = 0.0

        # 計算交易成本
        n_sell = len(to_sell)
        n_buy = len(to_buy)
        n_held = len(holdings)

        if mode == "compound":
            # 複利：每支持倉 = equity / K，成本按實際金額計算
            per_stock_amount = equity / k if k > 0 else 0
            day_cost = 0.0
            for _ in range(n_sell):
                day_cost += calc_trade_cost(per_stock_amount, is_sell=True)
            for _ in range(n_buy):
                day_cost += calc_trade_cost(per_stock_amount, is_sell=False)
            cost_rate = day_cost / equity if equity > 0 else 0
            net_return = gross_return - cost_rate
            equity *= (1 + net_return)
        else:
            # 定額：每支固定金額
            day_cost = 0.0
            for _ in range(n_sell):
                day_cost += calc_trade_cost(FIXED_AMOUNT_PER_STOCK, is_sell=True)
            for _ in range(n_buy):
                day_cost += calc_trade_cost(FIXED_AMOUNT_PER_STOCK, is_sell=False)
            total_position = n_held * FIXED_AMOUNT_PER_STOCK if n_held > 0 else k * FIXED_AMOUNT_PER_STOCK
            cost_rate = day_cost / total_position if total_position > 0 else 0
            net_return = gross_return - cost_rate

        total_cost += day_cost

        # Turnover = 交易股數 / 持倉股數
        turnover = (n_sell + n_buy) / (2 * n_held) if n_held > 0 else 0

        records.append(DailyRecord(
            dt=pred_date,
            gross_return=gross_return,
            net_return=net_return,
            market_return=market_return,
            cost=day_cost,
            turnover=turnover,
            n_holdings=n_held,
            n_buy=n_buy,
            n_sell=n_sell,
        ))

    return StrategyResult(
        name=strategy.name(),
        mode=mode,
        records=records,
        total_cost=total_cost,
    )


# ══════════════════════════════════════════════════════════════
# 統計指標計算
# ══════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    strategy: str
    mode: str
    ann_return: float        # 年化收益 %
    ann_excess: float        # 年化超額收益 %
    sharpe: float | None
    max_drawdown: float      # %
    calmar: float | None
    win_rate: float          # 跑贏大盤天數 %
    avg_turnover: float      # 平均日換手率 %
    total_cost: float        # 總交易成本（元）
    info_ratio: float | None # 資訊比率
    t_stat: float | None     # 超額收益 t 統計量
    n_days: int


def compute_metrics(result: StrategyResult) -> Metrics:
    """從回測結果計算統計指標"""
    records = result.records
    if not records:
        return Metrics(
            strategy=result.name, mode=result.mode,
            ann_return=0, ann_excess=0, sharpe=None, max_drawdown=0,
            calmar=None, win_rate=0, avg_turnover=0, total_cost=0,
            info_ratio=None, t_stat=None, n_days=0,
        )

    n_days = len(records)
    ann_factor = 250 / n_days if n_days > 0 else 1

    # 直接使用 daily net_return
    daily_returns = [r.net_return for r in records]
    market_returns = [r.market_return for r in records]
    turnovers = [r.turnover for r in records]

    # 累積收益
    cum_return = 1.0
    for r in daily_returns:
        cum_return *= (1 + r)

    cum_market = 1.0
    for r in market_returns:
        cum_market *= (1 + r)

    # 年化
    ann_return = ((cum_return ** ann_factor) - 1) * 100
    ann_market = ((cum_market ** ann_factor) - 1) * 100
    ann_excess = ann_return - ann_market

    # Sharpe ratio
    sharpe = None
    if len(daily_returns) >= 2:
        arr = np.array(daily_returns)
        if arr.std() > 0:
            sharpe = float(arr.mean() / arr.std() * np.sqrt(250))

    # Max Drawdown
    equity = 1.0
    peak = equity
    max_dd = 0.0
    for r in daily_returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    max_drawdown = max_dd * 100

    # Calmar
    calmar = None
    if max_drawdown > 0:
        calmar = ann_return / max_drawdown

    # Win rate（超額收益為正的天數）
    excess_daily = [r.net_return - r.market_return for r in records]
    win_days = sum(1 for e in excess_daily if e > 0)
    win_rate = (win_days / len(excess_daily) * 100) if excess_daily else 0

    # Avg turnover
    avg_turnover = (np.mean(turnovers) * 100) if turnovers else 0

    # Information ratio
    info_ratio = None
    if len(excess_daily) >= 2:
        arr = np.array(excess_daily)
        if arr.std() > 0:
            info_ratio = float(arr.mean() / arr.std() * np.sqrt(250))

    # t-statistic of excess return
    t_stat = None
    if len(excess_daily) >= 2:
        t_val, _ = stats.ttest_1samp(excess_daily, 0)
        if not np.isnan(t_val):
            t_stat = float(t_val)

    return Metrics(
        strategy=result.name,
        mode=result.mode,
        ann_return=round(ann_return, 2),
        ann_excess=round(ann_excess, 2),
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        max_drawdown=round(max_drawdown, 2),
        calmar=round(calmar, 3) if calmar is not None else None,
        win_rate=round(win_rate, 1),
        avg_turnover=round(avg_turnover, 2),
        total_cost=round(result.total_cost, 0),
        info_ratio=round(info_ratio, 3) if info_ratio is not None else None,
        t_stat=round(t_stat, 3) if t_stat is not None else None,
        n_days=n_days,
    )


# ══════════════════════════════════════════════════════════════
# 輸出 & 圖表
# ══════════════════════════════════════════════════════════════

def print_metrics_table(all_metrics: list[Metrics]) -> None:
    """印出策略比較表"""
    # 按 Sharpe 降序排列
    sorted_m = sorted(
        all_metrics,
        key=lambda m: m.sharpe if m.sharpe is not None else -999,
        reverse=True,
    )

    header = (
        f"{'Strategy':<30} {'Mode':<8} {'Ann.Ret%':>8} {'Excess%':>8} "
        f"{'Sharpe':>7} {'MaxDD%':>7} {'Calmar':>7} {'WinR%':>6} "
        f"{'Turn%':>6} {'Cost':>10} {'IR':>7} {'t-stat':>7}"
    )
    print("\n" + "=" * len(header))
    print("策略比較表（按 Sharpe Ratio 降序）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in sorted_m:
        sharpe_str = f"{m.sharpe:>7.3f}" if m.sharpe is not None else "   N/A "
        calmar_str = f"{m.calmar:>7.3f}" if m.calmar is not None else "   N/A "
        ir_str = f"{m.info_ratio:>7.3f}" if m.info_ratio is not None else "   N/A "
        t_str = f"{m.t_stat:>7.3f}" if m.t_stat is not None else "   N/A "

        print(
            f"{m.strategy:<30} {m.mode:<8} {m.ann_return:>8.2f} {m.ann_excess:>8.2f} "
            f"{sharpe_str} {m.max_drawdown:>7.2f} {calmar_str} {m.win_rate:>6.1f} "
            f"{m.avg_turnover:>6.2f} {m.total_cost:>10,.0f} {ir_str} {t_str}"
        )

    print("=" * len(header))


def save_metrics_csv(all_metrics: list[Metrics]) -> None:
    """存 CSV"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in all_metrics:
        rows.append({
            "strategy": m.strategy,
            "mode": m.mode,
            "ann_return": m.ann_return,
            "ann_excess": m.ann_excess,
            "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown,
            "calmar": m.calmar,
            "win_rate": m.win_rate,
            "avg_turnover": m.avg_turnover,
            "total_cost": m.total_cost,
            "info_ratio": m.info_ratio,
            "t_stat": m.t_stat,
            "n_days": m.n_days,
        })
    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "strategy_metrics.csv"
    df.to_csv(path, index=False)
    print(f"\nMetrics saved to {path}")


def plot_equity_curves(results: list[StrategyResult], mode: str) -> None:
    """繪製權益曲線"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Strategy Comparison — {mode.title()} Mode", fontsize=14)

    for result in results:
        if result.mode != mode or not result.records:
            continue

        dates = [r.dt for r in result.records]

        # 用 net_return 累積
        cum = [1.0]
        for r in result.records:
            cum.append(cum[-1] * (1 + r.net_return))

        ax1.plot(dates, cum[1:], label=result.name, alpha=0.8)

    # 大盤基準
    if results:
        bench_result = next((r for r in results if r.mode == mode and r.records), None)
        if bench_result:
            dates = [r.dt for r in bench_result.records]
            bench_cum = [1.0]
            for r in bench_result.records:
                bench_cum.append(bench_cum[-1] * (1 + r.market_return))
            ax1.plot(dates, bench_cum[1:], label="Market (Equal Weight)", color="black",
                     linestyle="--", linewidth=2, alpha=0.5)

    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=7, ncol=3, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="gray", linestyle="-", alpha=0.3)

    # Drawdown
    for result in results:
        if result.mode != mode or not result.records:
            continue

        dates = [r.dt for r in result.records]
        equity = 1.0
        peak = 1.0
        dd_list = []
        for r in result.records:
            equity *= (1 + r.net_return)
            peak = max(peak, equity)
            dd_list.append(-(peak - equity) / peak * 100)

        ax2.plot(dates, dd_list, label=result.name, alpha=0.8)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=7, ncol=3, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / f"equity_curves_{mode}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Equity curves saved to {path}")


def plot_turnover_vs_return(all_metrics: list[Metrics]) -> None:
    """Turnover vs Return 散佈圖"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for mode, marker in [("compound", "o"), ("fixed", "s")]:
        metrics = [m for m in all_metrics if m.mode == mode]
        if not metrics:
            continue

        x = [m.avg_turnover for m in metrics]
        y = [m.ann_excess for m in metrics]
        labels = [m.strategy for m in metrics]

        ax.scatter(x, y, marker=marker, s=80, alpha=0.7, label=f"{mode.title()} mode")

        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), fontsize=6, alpha=0.7,
                       xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Avg Daily Turnover (%)")
    ax.set_ylabel("Annualized Excess Return (%)")
    ax.set_title("Turnover vs Excess Return")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "turnover_vs_return.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Turnover vs Return plot saved to {path}")


# ══════════════════════════════════════════════════════════════
# 策略配置
# ══════════════════════════════════════════════════════════════

def build_strategies() -> list[Strategy]:
    """建立所有要比較的策略"""
    strategies = []

    # 策略 1: TopK
    for k in [5, 10, 15, 20, 30]:
        strategies.append(TopKStrategy(k))

    # 策略 2: TopK-Drop
    for k, d in [(10, 1), (10, 2), (10, 3), (20, 2), (20, 5)]:
        strategies.append(TopKDropStrategy(k, d))

    # 策略 3: Hold Period
    for k, h in [(10, 3), (10, 5), (20, 3), (20, 5)]:
        strategies.append(HoldPeriodStrategy(k, h))

    # 策略 3b: HoldDrop 混合（HoldPeriod + Drop 限制）
    for k, h, d in [
        (10, 2, 1), (10, 3, 1), (10, 4, 1), (10, 5, 1),
        (10, 7, 1), (10, 10, 1),
        (10, 5, 2), (10, 3, 2),
    ]:
        strategies.append(HoldDropStrategy(k, h, d))

    # 策略 3c: HoldDrop + Bottom 強制賣出
    for k, h, d, b in [
        # 基於最佳 H=3
        (10, 3, 1, 10), (10, 3, 1, 20), (10, 3, 1, 30),
        # 基於 H=4
        (10, 4, 1, 10), (10, 4, 1, 20), (10, 4, 1, 30),
    ]:
        strategies.append(HoldDropBottomStrategy(k, h, d, b))

    # 策略 4: Threshold Exit
    for k, m in [(10, 20), (10, 30), (10, 50)]:
        strategies.append(ThresholdExitStrategy(k, m))

    # 策略 5: Score Weighted
    for k in [10, 20]:
        strategies.append(ScoreWeightedStrategy(k))

    return strategies


# ══════════════════════════════════════════════════════════════
# 主函數
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 70)
    print("多策略回測比較")
    print(f"範圍: {START_MODEL_WEEK} ~ {END_MODEL_WEEK}")
    print("=" * 70)

    # 1. 收集模型
    print("\n[1/5] 收集模型...")
    model_infos = collect_models(START_MODEL_WEEK, END_MODEL_WEEK)
    print(f"  共 {len(model_infos)} 週模型")
    fallback_count = sum(1 for m in model_infos if m.is_fallback)
    if fallback_count:
        print(f"  其中 {fallback_count} 週使用 fallback 模型")

    if not model_infos:
        print("ERROR: 沒有找到任何模型！")
        return

    # 2. 導出和預載資料
    print("\n[2/5] 準備資料...")
    first_predict_start = get_week_date_range(model_infos[0].predict_week)[0]
    last_predict_end = get_week_date_range(model_infos[-1].predict_week)[1]
    lookback_days = 400

    export_start = first_predict_start - timedelta(days=lookback_days)
    export_end = last_predict_end

    print(f"  導出 qlib 資料: {export_start} ~ {export_end}")
    export_qlib_data(export_start, export_end)

    print("  初始化 qlib...")
    init_qlib()

    instruments = get_instruments()
    print(f"  股票數: {len(instruments)}")

    # 用第一個模型的因子列表
    _, first_factors, _ = load_model(model_infos[0].model_name)
    features_cache, prices_cache = preload_data(
        first_factors, instruments, first_predict_start, last_predict_end,
    )

    # 建立 close_wide，index 統一用 date 物件
    close_wide = prices_cache["close"].unstack(level="instrument")
    trading_days_date = sorted([
        d.date() if hasattr(d, 'date') else d for d in close_wide.index
    ])
    close_wide.index = trading_days_date

    print(f"  交易日: {len(trading_days_date)}")

    # 3. 產生所有日期的分數
    print("\n[3/5] 產生每日預測分數...")
    daily_scores = generate_daily_scores(model_infos, features_cache)
    print(f"  共 {len(daily_scores)} 個交易日有分數")

    # 釋放特徵快取
    del features_cache

    # 4. 跑所有策略
    print("\n[4/5] 跑策略回測...")
    strategies = build_strategies()
    all_results: list[StrategyResult] = []
    all_metrics: list[Metrics] = []

    total_runs = len(strategies) * 2  # compound + fixed
    run_idx = 0

    for strategy in strategies:
        for mode in ["compound", "fixed"]:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {strategy.name()} ({mode})...")

            result = simulate(
                strategy=strategy,
                daily_scores=daily_scores,
                close_wide=close_wide,
                trading_days=trading_days_date,
                mode=mode,
            )
            all_results.append(result)

            metrics = compute_metrics(result)
            all_metrics.append(metrics)

    # 5. 輸出結果
    print("\n[5/5] 輸出結果...")

    # 分別印 compound 和 fixed
    print("\n" + "=" * 70)
    print("COMPOUND MODE（複利模式，初始資金 100 萬）")
    print("=" * 70)
    compound_metrics = [m for m in all_metrics if m.mode == "compound"]
    print_metrics_table(compound_metrics)

    print("\n" + "=" * 70)
    print("FIXED MODE（定額模式，每支 5 萬）")
    print("=" * 70)
    fixed_metrics = [m for m in all_metrics if m.mode == "fixed"]
    print_metrics_table(fixed_metrics)

    # 存 CSV
    save_metrics_csv(all_metrics)

    # 圖表
    compound_results = [r for r in all_results if r.mode == "compound"]
    fixed_results = [r for r in all_results if r.mode == "fixed"]
    plot_equity_curves(compound_results, "compound")
    plot_equity_curves(fixed_results, "fixed")
    plot_turnover_vs_return(all_metrics)

    elapsed = time.time() - t_start
    print(f"\n完成！耗時 {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
