# Model Diagnosis Report

**Generated**: 2026-02-23 20:54
**Data range**: 2023W01 ~ 2025W51
**Runtime**: 64.6s

## Executive Summary

- **信號品質**: 需改善 (mono rho=0.40, spread t=1.04)
- **最佳 horizon**: 2 天（目前 label: 1 天） ← **建議調整**
- **IC 穩定性**: 穩定 (overall ICIR=0.06)
- **Regime 依賴**: 穩健
- **分數區分度**: 91 unique/day (良好)

---

## 1. Quantile Return Spread

| Quantile | Avg Daily Return (bps) |
|----------|----------------------|
| Q1 | 11.89 |
| Q2 | 8.37 |
| Q3 | 9.53 |
| Q4 | 11.62 |
| Q5 | 16.04 |

- 單調性: Spearman rho = 0.400 (p = 0.5046)
- Q5-Q1 spread: 4.15 bps/day (t = 1.04, p = 0.2991)
- **Verdict**: WEAK: 非單調或 spread 不顯著

---

## 2. Multi-Horizon IC

| Horizon | Mean IC | ICIR | IC>0% |
|---------|---------|------|-------|
| 1-day | 0.0092 | 0.06 | 53.2% |
| 2-day ** | 0.0145 | 0.09 | 54.9% |
| 3-day | 0.0102 | 0.07 | 54.0% |
| 5-day | 0.0101 | 0.07 | 53.5% |
| 10-day | 0.0021 | 0.01 | 50.1% |
| 20-day | 0.0096 | 0.06 | 53.9% |

- 目前 label: 1-day, IC = 0.0092
- 最佳 horizon: 2-day, IC = 0.0145
- **建議**: 改用 2-day return 作為 label (IC 提升 0.0053)

---

## 3. Rolling IC & Structural Breaks

- Overall: mean IC = 0.0092, ICIR = 0.06, IC>0 = 53.2%

| Period | Mean IC |
|--------|---------|
| 2023 | 0.0056 |
| 2024 | 0.0084 |
| 2025 | 0.0136 |

| Quarter | Mean IC |
|---------|---------|
| 2023Q1 | -0.0030 |
| 2023Q2 | -0.0061 |
| 2023Q3 | 0.0249 |
| 2023Q4 | 0.0035 |
| 2024Q1 | -0.0202 |
| 2024Q2 | 0.0335 |
| 2024Q3 | 0.0014 |
| 2024Q4 | 0.0166 |
| 2025Q1 | -0.0140 |
| 2025Q2 | 0.0390 |
| 2025Q3 | 0.0216 |
| 2025Q4 | 0.0038 |

- CUSUM 結構斷裂: 5 個
- IC 自相關: lag1=0.038, lag5=0.004
- IC half-life: 0.2 天

---

## 4. Market Regime Analysis

| Regime | Days | Mean IC | Daily Excess (bps) | Win Rate |
|--------|------|---------|-------------------|----------|
| bull | 311 | 0.0032 | 0.57 | 46.9% |
| sideways | 260 | 0.0121 | 7.42 | 53.8% |
| bear | 116 | 0.0206 | 5.22 | 43.1% |

- Regime 轉換次數: 88
- **Verdict**: 穩健（所有 regime IC > 0）

---

## 5. Win/Loss Clustering

- Win rate: 48.0% (343 wins / 372 losses)
- Runs test: z = 0.08 (p = 0.9350) → 隨機
- Ljung-Box (lag=10): stat = 36.87 (p = 0.0001) → 有自相關
- 最大連敗: 7 天 (隨機預期: 10 天)
- 最長 time-under-water: 372 天 (52%)

---

## 6. Score Distribution Quality

- 每日 unique 分數數: 91 / ~100 支股票
- 平均 score std: 0.0258
- 集中度 (within 1 std): 74.6%
- Top-10 隔日重疊率: 39.7%
- 連日排名相關 (Kendall tau): 0.353
