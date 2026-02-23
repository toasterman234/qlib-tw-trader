"""
模型診斷腳本

診斷 walk-forward 模型的弱點：信號品質、最佳預測 horizon、IC 穩定性、
市場 regime 依賴、勝負聚集、分數區分度，產出結構化報告和圖表。

診斷項目：
  1. Quantile Return Spread — 分數分組的收益是否單調？
  2. Multi-Horizon IC — 最佳預測 horizon 是幾天？
  3. Rolling IC + 結構斷裂 — IC 是否穩定？
  4. Market Regime 分析 — 多頭/空頭表現差異
  5. Win/Loss 聚集檢定 — 連敗是隨機還是聚集？
  6. Score 分佈品質 — 模型區分度夠嗎？
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# 加入 scripts/ 和 project root 到 path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from compare_strategies import (
    OUTPUT_DIR,
    START_MODEL_WEEK,
    END_MODEL_WEEK,
    export_qlib_data,
    init_qlib,
    get_instruments,
    load_model,
    preload_data,
    collect_models,
    generate_daily_scores,
    get_week_date_range,
    WeekModelInfo,
    HoldDropStrategy,
    simulate,
)

# ── 常數 ──
HORIZONS = [1, 2, 3, 5, 10, 20]
N_QUANTILES = 5
TOP_K = 10
HOLD_DAYS = 3
DROP_LIMIT = 1


# ══════════════════════════════════════════════════════════════
# 資料準備
# ══════════════════════════════════════════════════════════════

def load_extended_prices(
    instruments: list[str],
    start_date: date,
    end_date: date,
    extra_days: int = 40,
) -> tuple[pd.DataFrame, list[date]]:
    """載入延伸日期範圍的收盤價（供 multi-horizon IC 使用）"""
    from qlib.data import D

    extended_end = end_date + timedelta(days=extra_days)
    prices = D.features(
        instruments=instruments,
        fields=["$close"],
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=extended_end.strftime("%Y-%m-%d"),
    )
    prices.columns = ["close"]
    close_wide = prices["close"].unstack(level="instrument")
    trading_days = sorted([
        d.date() if hasattr(d, 'date') else d for d in close_wide.index
    ])
    close_wide.index = trading_days
    return close_wide, trading_days


# ══════════════════════════════════════════════════════════════
# 共用 IC 計算
# ══════════════════════════════════════════════════════════════

def compute_daily_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list[date],
    horizon: int = 1,
) -> pd.Series:
    """計算每日截面 Spearman IC（scores vs H 日 forward return）"""
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    ic_data = {}

    for pred_date in sorted(daily_scores.keys()):
        if pred_date not in day_to_idx:
            continue
        idx = day_to_idx[pred_date]
        if idx + 1 + horizon >= len(trading_days):
            continue

        t_buy = trading_days[idx + 1]
        t_sell = trading_days[idx + 1 + horizon]

        if t_buy not in close_wide.index or t_sell not in close_wide.index:
            continue

        returns = (close_wide.loc[t_sell] - close_wide.loc[t_buy]) / close_wide.loc[t_buy]
        scores = daily_scores[pred_date]

        common = scores.dropna().index.intersection(returns.dropna().index)
        if len(common) < 10:
            continue

        corr, _ = stats.spearmanr(scores[common], returns[common])
        if not np.isnan(corr):
            ic_data[pred_date] = corr

    return pd.Series(ic_data).sort_index()


# ══════════════════════════════════════════════════════════════
# Analysis 1: Quantile Return Spread
# ══════════════════════════════════════════════════════════════

@dataclass
class QuantileSpreadResult:
    avg_returns_by_quantile: pd.Series  # Q1..Q5 → avg daily return
    monotonicity_corr: float
    monotonicity_pvalue: float
    spread_series: pd.Series            # date → Q5-Q1 daily spread
    avg_daily_spread: float
    spread_t_stat: float
    spread_pvalue: float


def analyze_quantile_spread(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list[date],
) -> QuantileSpreadResult:
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    quantile_returns: dict[date, dict[int, float]] = {}

    for pred_date in sorted(daily_scores.keys()):
        if pred_date not in day_to_idx:
            continue
        idx = day_to_idx[pred_date]
        if idx + 2 >= len(trading_days):
            continue

        t1 = trading_days[idx + 1]
        t2 = trading_days[idx + 2]
        if t1 not in close_wide.index or t2 not in close_wide.index:
            continue

        returns = (close_wide.loc[t2] - close_wide.loc[t1]) / close_wide.loc[t1]
        scores = daily_scores[pred_date]
        common = scores.dropna().index.intersection(returns.dropna().index)
        if len(common) < N_QUANTILES * 2:
            continue

        s = scores[common]
        r = returns[common]
        try:
            q_labels = pd.qcut(s.rank(method='first'), N_QUANTILES, labels=False)
        except ValueError:
            q_labels = pd.cut(s, N_QUANTILES, labels=False)

        day_q_returns = {}
        for q in range(N_QUANTILES):
            mask = q_labels == q
            if mask.any():
                day_q_returns[q] = float(r[mask].mean())
        if len(day_q_returns) == N_QUANTILES:
            quantile_returns[pred_date] = day_q_returns

    # 彙總
    qr_df = pd.DataFrame(quantile_returns).T
    avg_returns = qr_df.mean()

    # 單調性
    mono_corr, mono_p = stats.spearmanr(range(N_QUANTILES), avg_returns.values)

    # Q5-Q1 spread
    spread = qr_df[N_QUANTILES - 1] - qr_df[0]
    spread_mean = float(spread.mean())
    t_val, p_val = stats.ttest_1samp(spread.dropna(), 0)

    return QuantileSpreadResult(
        avg_returns_by_quantile=avg_returns,
        monotonicity_corr=float(mono_corr),
        monotonicity_pvalue=float(mono_p),
        spread_series=spread,
        avg_daily_spread=spread_mean,
        spread_t_stat=float(t_val) if not np.isnan(t_val) else 0,
        spread_pvalue=float(p_val) if not np.isnan(p_val) else 1,
    )


def plot_quantile_spread(result: QuantileSpreadResult) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Quantile Return Spread Analysis", fontsize=14)

    # 上：各分位平均收益率
    labels = [f"Q{i+1}" for i in range(N_QUANTILES)]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, N_QUANTILES))
    ax1.bar(labels, result.avg_returns_by_quantile.values * 10000, color=colors)
    ax1.set_ylabel("Avg Daily Return (bps)")
    ax1.set_title(f"Return by Score Quantile (mono rho={result.monotonicity_corr:.2f})")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # 下：累積 Q5-Q1 spread
    cum_spread = result.spread_series.cumsum() * 100
    ax2.plot(cum_spread.index, cum_spread.values, color="steelblue")
    ax2.set_ylabel("Cumulative Q5-Q1 Spread (%)")
    ax2.set_title(f"Q5-Q1 Spread: {result.avg_daily_spread*10000:.1f} bps/day (t={result.spread_t_stat:.2f})")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_quantile_spread.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Analysis 2: Multi-Horizon IC
# ══════════════════════════════════════════════════════════════

@dataclass
class MultiHorizonIcResult:
    horizon_ic: dict[int, float]
    horizon_icir: dict[int, float]
    horizon_ic_pos_rate: dict[int, float]
    horizon_ic_series: dict[int, pd.Series]
    optimal_horizon: int
    current_horizon_ic: float


def analyze_multi_horizon_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list[date],
) -> MultiHorizonIcResult:
    horizon_ic = {}
    horizon_icir = {}
    horizon_ic_pos_rate = {}
    horizon_ic_series = {}

    for h in HORIZONS:
        ic_series = compute_daily_ic(daily_scores, close_wide, trading_days, horizon=h)
        if len(ic_series) < 10:
            continue
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        horizon_ic[h] = mean_ic
        horizon_icir[h] = mean_ic / std_ic if std_ic > 0 else 0
        horizon_ic_pos_rate[h] = float((ic_series > 0).mean())
        horizon_ic_series[h] = ic_series

    optimal = max(horizon_ic, key=horizon_ic.get) if horizon_ic else 1

    return MultiHorizonIcResult(
        horizon_ic=horizon_ic,
        horizon_icir=horizon_icir,
        horizon_ic_pos_rate=horizon_ic_pos_rate,
        horizon_ic_series=horizon_ic_series,
        optimal_horizon=optimal,
        current_horizon_ic=horizon_ic.get(1, 0),
    )


def plot_multi_horizon_ic(result: MultiHorizonIcResult) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Multi-Horizon IC Analysis", fontsize=14)

    horizons = sorted(result.horizon_ic.keys())
    ics = [result.horizon_ic[h] for h in horizons]
    icirs = [result.horizon_icir[h] for h in horizons]
    labels = [f"{h}d" for h in horizons]

    colors = ["#e74c3c" if h == result.optimal_horizon else "#3498db" for h in horizons]

    ax1.bar(labels, ics, color=colors)
    ax1.set_ylabel("Mean IC")
    ax1.set_title(f"Mean IC by Horizon (optimal: {result.optimal_horizon}-day)")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.bar(labels, icirs, color=colors)
    ax2.set_ylabel("ICIR (IC / std)")
    ax2.set_title("ICIR by Horizon")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_multi_horizon_ic.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Analysis 3: Rolling IC + Structural Break
# ══════════════════════════════════════════════════════════════

@dataclass
class RollingIcResult:
    daily_ic: pd.Series
    rolling_ic_20d: pd.Series
    mean_ic: float
    icir: float
    ic_pos_rate: float
    ic_by_year: dict[int, float]
    ic_by_quarter: dict[str, float]
    cusum_values: np.ndarray
    cusum_break_indices: list[int]
    ic_autocorr: list[float]
    ic_half_life: float | None


def analyze_rolling_ic(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list[date],
) -> RollingIcResult:
    daily_ic = compute_daily_ic(daily_scores, close_wide, trading_days, horizon=1)

    rolling_20 = daily_ic.rolling(20, min_periods=10).mean()
    mean_ic = float(daily_ic.mean())
    std_ic = float(daily_ic.std())
    icir = mean_ic / std_ic if std_ic > 0 else 0
    ic_pos_rate = float((daily_ic > 0).mean())

    # 按年、按季
    ic_df = daily_ic.to_frame("ic")
    ic_df["year"] = [d.year for d in ic_df.index]
    ic_df["quarter"] = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in ic_df.index]
    ic_by_year = ic_df.groupby("year")["ic"].mean().to_dict()
    ic_by_quarter = ic_df.groupby("quarter")["ic"].mean().to_dict()

    # CUSUM
    ic_vals = daily_ic.values
    cusum = np.cumsum(ic_vals - mean_ic) / (std_ic if std_ic > 0 else 1)
    n = len(cusum)
    # 簡易閾值：±sqrt(n) * 0.5
    threshold = np.sqrt(n) * 0.5
    breaks = []
    for i in range(1, n):
        if abs(cusum[i]) > threshold and abs(cusum[i - 1]) <= threshold:
            breaks.append(i)

    # IC 自相關
    ic_series_pd = pd.Series(ic_vals)
    autocorrs = []
    for lag in range(1, 11):
        ac = float(ic_series_pd.autocorr(lag))
        autocorrs.append(ac if not np.isnan(ac) else 0)

    # Half-life from AR(1)
    phi = autocorrs[0] if autocorrs else 0
    half_life = None
    if 0 < phi < 1:
        half_life = -np.log(2) / np.log(phi)

    return RollingIcResult(
        daily_ic=daily_ic,
        rolling_ic_20d=rolling_20,
        mean_ic=mean_ic,
        icir=icir,
        ic_pos_rate=ic_pos_rate,
        ic_by_year=ic_by_year,
        ic_by_quarter=ic_by_quarter,
        cusum_values=cusum,
        cusum_break_indices=breaks,
        ic_autocorr=autocorrs,
        ic_half_life=half_life,
    )


def plot_rolling_ic(result: RollingIcResult) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Rolling IC & Structural Break Analysis", fontsize=14)

    # 上：每日 IC + rolling mean + 斷裂點
    dates = result.daily_ic.index
    ax1.scatter(dates, result.daily_ic.values, alpha=0.15, s=8, color="gray", label="Daily IC")
    ax1.plot(result.rolling_ic_20d.index, result.rolling_ic_20d.values,
             color="steelblue", linewidth=1.5, label="20-day Rolling Mean")
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.axhline(y=result.mean_ic, color="red", linestyle="--", alpha=0.5,
                label=f"Overall Mean={result.mean_ic:.4f}")

    for bi in result.cusum_break_indices:
        if bi < len(dates):
            ax1.axvline(x=dates[bi], color="red", linestyle=":", alpha=0.7)

    ax1.set_ylabel("Spearman IC")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 下：按季 IC
    quarters = sorted(result.ic_by_quarter.keys())
    quarter_ics = [result.ic_by_quarter[q] for q in quarters]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in quarter_ics]
    ax2.bar(quarters, quarter_ics, color=colors)
    ax2.set_ylabel("Mean IC")
    ax2.set_title("IC by Quarter")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_rolling_ic.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Analysis 4: Market Regime
# ══════════════════════════════════════════════════════════════

@dataclass
class RegimeAnalysisResult:
    regime_labels: pd.Series
    regime_stats: dict[str, dict]
    regime_transitions: int


def analyze_market_regimes(
    daily_scores: dict[date, pd.Series],
    close_wide: pd.DataFrame,
    trading_days: list[date],
    daily_ic: pd.Series,
    daily_records: list,
) -> RegimeAnalysisResult:
    # 等權市場日收益
    market_daily = close_wide.pct_change(fill_method=None).mean(axis=1).dropna()

    # 累積收益 + SMA50 + vol20
    cum_market = (1 + market_daily).cumprod()
    sma50 = cum_market.rolling(50, min_periods=30).mean()
    vol20 = market_daily.rolling(20, min_periods=10).std()
    vol_median = float(vol20.median())

    # 分類 regime
    regime_labels = pd.Series(index=market_daily.index, dtype=str)
    for d in regime_labels.index:
        if pd.isna(sma50.get(d)) or pd.isna(vol20.get(d)):
            regime_labels[d] = "unknown"
            continue
        is_uptrend = cum_market[d] > sma50[d]
        is_high_vol = vol20[d] > vol_median
        if is_uptrend and not is_high_vol:
            regime_labels[d] = "bull"
        elif not is_uptrend and is_high_vol:
            regime_labels[d] = "bear"
        else:
            regime_labels[d] = "sideways"

    # 建立 daily_records 的 excess return lookup
    excess_by_date = {}
    for r in daily_records:
        excess_by_date[r.dt] = r.net_return - r.market_return

    # 按 regime 統計
    regime_stats = {}
    for regime in ["bull", "bear", "sideways"]:
        regime_dates = set(regime_labels[regime_labels == regime].index)

        # IC
        ic_in_regime = daily_ic[[d for d in daily_ic.index if d in regime_dates]]
        # Excess return
        excess_in_regime = [excess_by_date[d] for d in excess_by_date if d in regime_dates]

        n_days = len(regime_dates)
        regime_stats[regime] = {
            "n_days": n_days,
            "mean_ic": float(ic_in_regime.mean()) if len(ic_in_regime) > 0 else 0,
            "mean_excess": float(np.mean(excess_in_regime)) * 10000 if excess_in_regime else 0,  # bps
            "win_rate": sum(1 for e in excess_in_regime if e > 0) / len(excess_in_regime) * 100 if excess_in_regime else 0,
        }

    # Regime transitions
    valid = regime_labels[regime_labels != "unknown"]
    transitions = sum(1 for i in range(1, len(valid)) if valid.iloc[i] != valid.iloc[i - 1])

    return RegimeAnalysisResult(
        regime_labels=regime_labels,
        regime_stats=regime_stats,
        regime_transitions=transitions,
    )


def plot_market_regimes(result: RegimeAnalysisResult, close_wide: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Market Regime Analysis", fontsize=14)

    # 上：累積市場收益 + regime 背景
    market_daily = close_wide.pct_change(fill_method=None).mean(axis=1).dropna()
    cum_market = (1 + market_daily).cumprod()

    ax1.plot(cum_market.index, cum_market.values, color="black", linewidth=1)

    regime_colors = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#95a5a6", "unknown": "white"}
    for d in result.regime_labels.index:
        regime = result.regime_labels[d]
        if regime != "unknown":
            ax1.axvspan(d, d + timedelta(days=1), alpha=0.15, color=regime_colors[regime], linewidth=0)

    ax1.set_ylabel("Cumulative Market Return")
    ax1.set_title("Equal-Weight Market with Regime Shading")
    ax1.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=regime_colors[r], alpha=0.3, label=r) for r in ["bull", "bear", "sideways"]]
    ax1.legend(handles=legend_items, fontsize=8)

    # 下：各 regime 的 IC 和 excess return
    regimes = ["bull", "sideways", "bear"]
    x = np.arange(len(regimes))
    width = 0.35

    ics = [result.regime_stats[r]["mean_ic"] for r in regimes]
    excess = [result.regime_stats[r]["mean_excess"] for r in regimes]

    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width / 2, ics, width, label="Mean IC", color="#3498db", alpha=0.7)
    bars2 = ax2_twin.bar(x + width / 2, excess, width, label="Mean Excess (bps)", color="#e67e22", alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r}\n({result.regime_stats[r]['n_days']}d)" for r in regimes])
    ax2.set_ylabel("Mean IC")
    ax2_twin.set_ylabel("Mean Daily Excess (bps)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("IC and Excess Return by Regime")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_market_regimes.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Analysis 5: Win/Loss Clustering
# ══════════════════════════════════════════════════════════════

@dataclass
class ClusteringResult:
    win_rate: float
    n_wins: int
    n_losses: int
    runs_test_z: float
    runs_test_pvalue: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    max_consecutive_loss: int
    expected_max_loss_streak: float
    max_time_under_water: int
    total_days: int


def analyze_win_loss_clustering(daily_records: list) -> ClusteringResult:
    excess = [r.net_return - r.market_return for r in daily_records]
    wins = [1 if e > 0 else 0 for e in excess]
    n = len(wins)
    n_wins = sum(wins)
    n_losses = n - n_wins

    # Runs test
    runs = 1
    for i in range(1, n):
        if wins[i] != wins[i - 1]:
            runs += 1

    expected_runs = 1 + 2 * n_wins * n_losses / n if n > 0 else 0
    var_denom = n ** 2 * (n - 1) if n > 1 else 1
    var_runs = (2 * n_wins * n_losses * (2 * n_wins * n_losses - n)) / var_denom if var_denom > 0 else 0
    z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

    # Ljung-Box
    excess_arr = np.array(excess)
    lb_result = acorr_ljungbox(excess_arr, lags=[10], return_df=True)
    lb_stat = float(lb_result["lb_stat"].iloc[0])
    lb_p = float(lb_result["lb_pvalue"].iloc[0])

    # Max consecutive loss
    max_loss_streak = 0
    current_streak = 0
    for w in wins:
        if w == 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0

    # Expected max streak under random
    p_loss = n_losses / n if n > 0 else 0.5
    expected_max = np.log(n) / np.log(1 / p_loss) if 0 < p_loss < 1 else 0

    # Time under water
    cum_excess = np.cumsum(excess)
    peak = np.maximum.accumulate(cum_excess)
    underwater = cum_excess < peak
    max_tuw = 0
    current_tuw = 0
    for uw in underwater:
        if uw:
            current_tuw += 1
            max_tuw = max(max_tuw, current_tuw)
        else:
            current_tuw = 0

    return ClusteringResult(
        win_rate=n_wins / n * 100 if n > 0 else 0,
        n_wins=n_wins,
        n_losses=n_losses,
        runs_test_z=z_runs,
        runs_test_pvalue=p_runs,
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_p,
        max_consecutive_loss=max_loss_streak,
        expected_max_loss_streak=expected_max,
        max_time_under_water=max_tuw,
        total_days=n,
    )


# ══════════════════════════════════════════════════════════════
# Analysis 6: Score Distribution Quality
# ══════════════════════════════════════════════════════════════

@dataclass
class ScoreQualityResult:
    avg_unique_scores: float
    avg_score_std: float
    avg_concentration: float
    avg_topk_overlap: float
    avg_rank_correlation: float
    daily_stats: pd.DataFrame


def analyze_score_quality(
    daily_scores: dict[date, pd.Series],
) -> ScoreQualityResult:
    stats_rows = []
    prev_topk: set | None = None
    prev_scores: pd.Series | None = None

    for d in sorted(daily_scores.keys()):
        scores = daily_scores[d]
        n_unique = int(scores.nunique())
        score_std = float(scores.std())

        # 集中度
        mean_s = scores.mean()
        std_s = scores.std()
        within_1std = float(((scores - mean_s).abs() <= std_s).mean()) if std_s > 0 else 1.0

        # Top-K overlap
        topk_today = set(scores.nlargest(TOP_K).index)
        overlap = len(topk_today & prev_topk) / TOP_K if prev_topk is not None else np.nan

        # Rank correlation
        tau = np.nan
        if prev_scores is not None:
            common = scores.index.intersection(prev_scores.index)
            if len(common) >= 10:
                t, _ = stats.kendalltau(scores[common], prev_scores[common])
                tau = float(t) if not np.isnan(t) else np.nan

        stats_rows.append({
            "date": d,
            "unique": n_unique,
            "std": score_std,
            "concentration": within_1std,
            "topk_overlap": overlap,
            "rank_corr": tau,
        })
        prev_topk = topk_today
        prev_scores = scores

    df = pd.DataFrame(stats_rows).set_index("date")

    return ScoreQualityResult(
        avg_unique_scores=float(df["unique"].mean()),
        avg_score_std=float(df["std"].mean()),
        avg_concentration=float(df["concentration"].mean()),
        avg_topk_overlap=float(df["topk_overlap"].mean()),
        avg_rank_correlation=float(df["rank_corr"].mean()),
        daily_stats=df,
    )


def plot_score_quality(result: ScoreQualityResult) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle("Score Distribution Quality", fontsize=14)

    df = result.daily_stats

    ax1.plot(df.index, df["unique"].values, color="steelblue", alpha=0.7, linewidth=0.8)
    ax1.axhline(y=result.avg_unique_scores, color="red", linestyle="--",
                label=f"Mean={result.avg_unique_scores:.0f}")
    ax1.set_ylabel("Unique Scores per Day")
    ax1.set_title("Score Differentiation")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    overlap = df["topk_overlap"].dropna()
    ax2.plot(overlap.index, overlap.values * 100, color="steelblue", alpha=0.7, linewidth=0.8)
    ax2.axhline(y=result.avg_topk_overlap * 100, color="red", linestyle="--",
                label=f"Mean={result.avg_topk_overlap * 100:.0f}%")
    ax2.set_ylabel("Top-10 Overlap with Previous Day (%)")
    ax2.set_title("Top-K Stability")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_score_quality.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# Report Writer
# ══════════════════════════════════════════════════════════════

def write_report(
    report_path: Path,
    q_result: QuantileSpreadResult,
    h_result: MultiHorizonIcResult,
    r_result: RollingIcResult,
    reg_result: RegimeAnalysisResult,
    c_result: ClusteringResult,
    s_result: ScoreQualityResult,
    elapsed: float,
) -> None:
    lines = []
    lines.append("# Model Diagnosis Report\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Data range**: {START_MODEL_WEEK} ~ {END_MODEL_WEEK}")
    lines.append(f"**Runtime**: {elapsed:.1f}s\n")

    # ── Executive Summary ──
    lines.append("## Executive Summary\n")
    mono_ok = q_result.monotonicity_corr > 0.8
    spread_ok = q_result.spread_pvalue < 0.05
    optimal_h = h_result.optimal_horizon
    horizon_mismatch = optimal_h != 1
    ic_stable = all(v > 0 for v in r_result.ic_by_year.values())

    regime_ics = {k: v["mean_ic"] for k, v in reg_result.regime_stats.items()}
    all_positive_ic = all(v > 0 for v in regime_ics.values())

    lines.append(f"- **信號品質**: {'單調且顯著' if (mono_ok and spread_ok) else '需改善'} "
                 f"(mono rho={q_result.monotonicity_corr:.2f}, spread t={q_result.spread_t_stat:.2f})")
    lines.append(f"- **最佳 horizon**: {optimal_h} 天（目前 label: 1 天）"
                 f"{' ← **建議調整**' if horizon_mismatch else ''}")
    lines.append(f"- **IC 穩定性**: {'穩定' if ic_stable else '不穩定'} "
                 f"(overall ICIR={r_result.icir:.2f})")
    lines.append(f"- **Regime 依賴**: {'穩健' if all_positive_ic else 'Regime 依賴'}")
    lines.append(f"- **分數區分度**: {s_result.avg_unique_scores:.0f} unique/day "
                 f"({'良好' if s_result.avg_unique_scores > 50 else '不足'})")
    lines.append("")

    # ── 1. Quantile Return Spread ──
    lines.append("---\n")
    lines.append("## 1. Quantile Return Spread\n")
    lines.append("| Quantile | Avg Daily Return (bps) |")
    lines.append("|----------|----------------------|")
    for i in range(N_QUANTILES):
        ret_bps = q_result.avg_returns_by_quantile.iloc[i] * 10000
        lines.append(f"| Q{i+1} | {ret_bps:.2f} |")
    lines.append("")
    lines.append(f"- 單調性: Spearman rho = {q_result.monotonicity_corr:.3f} "
                 f"(p = {q_result.monotonicity_pvalue:.4f})")
    lines.append(f"- Q5-Q1 spread: {q_result.avg_daily_spread * 10000:.2f} bps/day "
                 f"(t = {q_result.spread_t_stat:.2f}, p = {q_result.spread_pvalue:.4f})")
    verdict = "GOOD: 單調且 spread 顯著" if (mono_ok and spread_ok) else \
              "WEAK: 非單調或 spread 不顯著" if not mono_ok else "MODERATE"
    lines.append(f"- **Verdict**: {verdict}")
    lines.append("")

    # ── 2. Multi-Horizon IC ──
    lines.append("---\n")
    lines.append("## 2. Multi-Horizon IC\n")
    lines.append("| Horizon | Mean IC | ICIR | IC>0% |")
    lines.append("|---------|---------|------|-------|")
    for h in sorted(h_result.horizon_ic.keys()):
        ic = h_result.horizon_ic[h]
        icir = h_result.horizon_icir[h]
        pos = h_result.horizon_ic_pos_rate[h] * 100
        marker = " **" if h == h_result.optimal_horizon else ""
        lines.append(f"| {h}-day{marker} | {ic:.4f} | {icir:.2f} | {pos:.1f}% |")
    lines.append("")
    lines.append(f"- 目前 label: 1-day, IC = {h_result.current_horizon_ic:.4f}")
    lines.append(f"- 最佳 horizon: {h_result.optimal_horizon}-day, "
                 f"IC = {h_result.horizon_ic.get(h_result.optimal_horizon, 0):.4f}")
    if horizon_mismatch:
        improvement = (h_result.horizon_ic.get(h_result.optimal_horizon, 0) - h_result.current_horizon_ic)
        lines.append(f"- **建議**: 改用 {h_result.optimal_horizon}-day return 作為 label "
                     f"(IC 提升 {improvement:.4f})")
    lines.append("")

    # ── 3. Rolling IC ──
    lines.append("---\n")
    lines.append("## 3. Rolling IC & Structural Breaks\n")
    lines.append(f"- Overall: mean IC = {r_result.mean_ic:.4f}, "
                 f"ICIR = {r_result.icir:.2f}, IC>0 = {r_result.ic_pos_rate:.1%}")
    lines.append("")
    lines.append("| Period | Mean IC |")
    lines.append("|--------|---------|")
    for y in sorted(r_result.ic_by_year.keys()):
        lines.append(f"| {y} | {r_result.ic_by_year[y]:.4f} |")
    lines.append("")
    lines.append("| Quarter | Mean IC |")
    lines.append("|---------|---------|")
    for q in sorted(r_result.ic_by_quarter.keys()):
        lines.append(f"| {q} | {r_result.ic_by_quarter[q]:.4f} |")
    lines.append("")
    n_breaks = len(r_result.cusum_break_indices)
    lines.append(f"- CUSUM 結構斷裂: {'未檢測到' if n_breaks == 0 else f'{n_breaks} 個'}")
    lines.append(f"- IC 自相關: lag1={r_result.ic_autocorr[0]:.3f}, "
                 f"lag5={r_result.ic_autocorr[4]:.3f}")
    if r_result.ic_half_life is not None:
        lines.append(f"- IC half-life: {r_result.ic_half_life:.1f} 天")
    lines.append("")

    # ── 4. Market Regime ──
    lines.append("---\n")
    lines.append("## 4. Market Regime Analysis\n")
    lines.append("| Regime | Days | Mean IC | Daily Excess (bps) | Win Rate |")
    lines.append("|--------|------|---------|-------------------|----------|")
    for regime in ["bull", "sideways", "bear"]:
        s = reg_result.regime_stats[regime]
        lines.append(f"| {regime} | {s['n_days']} | {s['mean_ic']:.4f} | "
                     f"{s['mean_excess']:.2f} | {s['win_rate']:.1f}% |")
    lines.append("")
    lines.append(f"- Regime 轉換次數: {reg_result.regime_transitions}")
    verdict = "穩健（所有 regime IC > 0）" if all_positive_ic else "Regime 依賴（部分 regime IC < 0）"
    lines.append(f"- **Verdict**: {verdict}")
    lines.append("")

    # ── 5. Win/Loss Clustering ──
    lines.append("---\n")
    lines.append("## 5. Win/Loss Clustering\n")
    lines.append(f"- Win rate: {c_result.win_rate:.1f}% "
                 f"({c_result.n_wins} wins / {c_result.n_losses} losses)")
    runs_verdict = "隨機" if c_result.runs_test_pvalue > 0.05 else \
                   ("聚集" if c_result.runs_test_z < 0 else "交替")
    lines.append(f"- Runs test: z = {c_result.runs_test_z:.2f} "
                 f"(p = {c_result.runs_test_pvalue:.4f}) → {runs_verdict}")
    lb_verdict = "無自相關" if c_result.ljung_box_pvalue > 0.05 else "有自相關"
    lines.append(f"- Ljung-Box (lag=10): stat = {c_result.ljung_box_stat:.2f} "
                 f"(p = {c_result.ljung_box_pvalue:.4f}) → {lb_verdict}")
    lines.append(f"- 最大連敗: {c_result.max_consecutive_loss} 天 "
                 f"(隨機預期: {c_result.expected_max_loss_streak:.0f} 天)")
    lines.append(f"- 最長 time-under-water: {c_result.max_time_under_water} 天 "
                 f"({c_result.max_time_under_water / c_result.total_days * 100:.0f}%)")
    lines.append("")

    # ── 6. Score Quality ──
    lines.append("---\n")
    lines.append("## 6. Score Distribution Quality\n")
    lines.append(f"- 每日 unique 分數數: {s_result.avg_unique_scores:.0f} / ~100 支股票")
    lines.append(f"- 平均 score std: {s_result.avg_score_std:.4f}")
    lines.append(f"- 集中度 (within 1 std): {s_result.avg_concentration:.1%}")
    lines.append(f"- Top-10 隔日重疊率: {s_result.avg_topk_overlap:.1%}")
    lines.append(f"- 連日排名相關 (Kendall tau): {s_result.avg_rank_correlation:.3f}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {report_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("Model Diagnosis")
    print(f"Range: {START_MODEL_WEEK} ~ {END_MODEL_WEEK}")
    print("=" * 70)

    # ── 1. 收集模型 ──
    print("\n[1/8] Collecting models...")
    model_infos = collect_models(START_MODEL_WEEK, END_MODEL_WEEK)
    print(f"  {len(model_infos)} weeks")

    # ── 2. 導出資料 ──
    print("\n[2/8] Exporting qlib data...")
    first_predict_start = get_week_date_range(model_infos[0].predict_week)[0]
    last_predict_end = get_week_date_range(model_infos[-1].predict_week)[1]
    lookback_days = 180
    export_start = first_predict_start - timedelta(days=lookback_days)
    export_end = last_predict_end
    export_qlib_data(export_start, export_end)

    print("  Initializing qlib...")
    init_qlib()
    instruments = get_instruments()
    print(f"  {len(instruments)} instruments")

    # ── 3. 載入資料 ──
    print("\n[3/8] Loading data...")
    _, first_factors, _ = load_model(model_infos[0].model_name)
    features_cache, _ = preload_data(first_factors, instruments, first_predict_start, last_predict_end)

    print("  Loading extended prices (for multi-horizon IC)...")
    close_wide, trading_days = load_extended_prices(instruments, first_predict_start, last_predict_end)
    print(f"  {len(trading_days)} trading days")

    # ── 4. 產生分數 ──
    print("\n[4/8] Generating daily scores...")
    daily_scores = generate_daily_scores(model_infos, features_cache)
    print(f"  {len(daily_scores)} days with scores")
    del features_cache

    # ── 5. 跑最佳策略模擬 ──
    print("\n[5/8] Simulating HoldDrop(K=10,H=3,D=1)...")
    strategy = HoldDropStrategy(k=TOP_K, h=HOLD_DAYS, d=DROP_LIMIT)
    sim_result = simulate(strategy, daily_scores, close_wide, trading_days, mode="compound")
    print(f"  {len(sim_result.records)} records")

    # ── 6. 執行 6 項診斷 ──
    print("\n[6/8] Running diagnostics...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  [1/6] Quantile return spread...")
    q_result = analyze_quantile_spread(daily_scores, close_wide, trading_days)

    print("  [2/6] Multi-horizon IC...")
    h_result = analyze_multi_horizon_ic(daily_scores, close_wide, trading_days)

    print("  [3/6] Rolling IC + structural breaks...")
    r_result = analyze_rolling_ic(daily_scores, close_wide, trading_days)

    print("  [4/6] Market regime analysis...")
    reg_result = analyze_market_regimes(
        daily_scores, close_wide, trading_days,
        r_result.daily_ic, sim_result.records,
    )

    print("  [5/6] Win/loss clustering...")
    c_result = analyze_win_loss_clustering(sim_result.records)

    print("  [6/6] Score distribution quality...")
    s_result = analyze_score_quality(daily_scores)

    # ── 7. 圖表 ──
    print("\n[7/8] Generating charts...")
    plot_quantile_spread(q_result)
    plot_multi_horizon_ic(h_result)
    plot_rolling_ic(r_result)
    plot_market_regimes(reg_result, close_wide)
    plot_score_quality(s_result)

    # ── 8. 報告 ──
    print("\n[8/8] Writing report...")
    elapsed = time.time() - t_start
    report_path = OUTPUT_DIR / "model_diagnosis.md"
    write_report(report_path, q_result, h_result, r_result, reg_result, c_result, s_result, elapsed)

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
