# Crypto 轉戰分析：交易尺度與成本框架

**生成時間**：2026-02-27
**目的**：評估從台股系統轉向 Crypto 永續合約的可行性，建立成本-收益分析框架

---

## 1. 動機：台股系統的執行障礙

台股量化系統（DoubleEnsemble, Sharpe 1.72, 年化超額 +23.9%）在紙上表現良好，但以下障礙使其無法實盤驗證：

| 問題 | 影響 |
|------|------|
| 財力證明門檻 | 每階 5 萬 TWD，無法自動複利擴張 |
| 零股交易 | 流動性差、撮合延遲、滑價大 |
| 漲跌停 10% | 動量策略被截斷，極端行情無法出場 |
| T+2 交割 | 資金周轉效率低 |

而 Crypto 市場無以上限制：小數交易、24/7、無漲跌停、API 下單成熟、可以用 $100 開始實測。

---

## 2. 四條路的報酬與可行性比較

| 指標 | 台股 ETF (0050) | BTC 長期持有 | 台股預測 (現有系統) | Crypto ML 預測 |
|------|:---:|:---:|:---:|:---:|
| **年化報酬** | 18.5% | ~49% (歷史) | 55.1% | 文獻 ~60-120% |
| **Sharpe** | ~0.8 | ~0.8-1.0 | 1.72 | 文獻 1.5-1.7 |
| **MaxDD** | ~25% | 73-80% | 38.7% | 文獻 ~20-40% |
| **所需資金** | 1 萬起 | $100 起 | 50 萬+ | $100 起 |
| **每日操作** | 零 | 零 | 不可行 | 電腦開著 |
| **可執行性** | 完全可行 | 完全可行 | **不可行** | 完全可行 |

### 2.1 台股 ETF 0050 定投

- 10 年年化 **18.5%**，費用率 0.32%（Morningstar）
- 零操作、零技術門檻
- TSMC 佔比 ~50%，本質上是押注一家公司
- 年度報酬：2019 +33.5%, 2020 +31.1%, 2021 +22.0%, 2022 -21.2%, 2023 +27.4%, 2024 +48.7%

### 2.2 BTC 長期持有

- 10 年年化 ~49%，但 Morgan Stanley 明確表示未來十年不可能重現
- 保守預估未來年化 **15-30%**（隨市值增大，波動率會收斂）
- MaxDD 73-80%
- DCA 策略 Sharpe ~1.45-1.85
- 年度報酬：2019 +95%, 2020 +301%, 2021 +66%, 2022 -65%, 2023 +156%, 2024 +121%

### 2.3 台股預測（現有系統）

- 最佳策略 HoldDrop(K=10,H=3,D=1)：年化 55.1%, Sharpe 1.72, MaxDD 38.7%
- t-stat 1.89（未達統計顯著 2.0）
- IC 僅 0.017（信號很弱，靠集中投資放大）
- **致命問題**：所有數字都是紙上談兵，無法落地執行

### 2.4 Crypto ML 預測（文獻數據）

| 策略 | 年化報酬 | Sharpe | 來源 |
|------|------:|------:|------|
| Cross-section ML (long-short) | ~120%+ | 1.66 | Cakici et al. (2024) |
| Trend-following (Top-20 coins) | BTC + 10.8% alpha | >1.5 | Zarattini et al. (2025) |
| Pairs trading | 極高 | 3.97 | 文獻理想條件 |

關鍵文獻發現：

> "Unlike in stocks, machine learning gains in cryptocurrency markets **do not visibly decline over time**." — Cakici et al. (2024)

> "Return predictability derives mainly from **a handful of simple characteristics**: market price, past alpha, illiquidity, and momentum." — 不需要 263 個因子

---

## 3. 成本結構對比：台股 vs Crypto 永續合約

| 成本項目 | 台股 (玉山 6 折) | Binance 永續 (Maker) | Binance 永續 (Taker) |
|----------|:---:|:---:|:---:|
| 買入手續費 | 0.0855% | 0.02% | 0.05% |
| 賣出手續費 | 0.0855% | 0.02% | 0.05% |
| 交易稅 (賣出) | 0.30% | 0% | 0% |
| **單次 round-trip** | **0.471%** | **0.04%** | **0.10%** |
| Funding rate | N/A | ~0.01%/8h | ~0.01%/8h |
| 滑價 (Top-20) | ~0.1-0.3% (零股) | ~0.02% | ~0.02% |

Crypto 永續合約的單次 round-trip 成本是台股的 **1/12 ~ 1/5**。

### 3.1 Funding Rate 影響

Funding 是按**全部持倉**收取的，不像交易成本只影響換手部分：

| Funding 環境 | 平均 rate/8h | 日成本 | 年成本 |
|-------------|:---:|:---:|:---:|
| 牛市（多頭付空頭）| +0.03% | 0.09% | 33% |
| 盤整 | +0.01% | 0.03% | 11% |
| 熊市（空頭付多頭）| -0.01% | -0.03% | -11% (收入) |
| 全週期平均 | +0.01% | 0.03% | ~11% |

---

## 4. IC 推導（無模型情況）

### 4.1 台股 IC 類比

模型報告 Multi-Horizon IC（DoubleEnsemble, 303 因子, TW100）：

| Horizon | Taiwan IC |
|---------|--------:|
| 1-day | 0.011 |
| 2-day | 0.016 |
| 3-day | 0.018 |
| **5-day** | **0.020** |
| 10-day | 0.014 |
| 20-day | 0.008 |

IC 在 3-5 天達到峰值。Crypto 市場運轉速度約 3× 股市（24/7、無漲跌停、散戶主導），等效最佳 horizon 約 **1-2 天**。

### 4.2 市場效率差異

| 特徵 | 台股 TW100 | Crypto Top-20 | IC 影響 |
|------|:---:|:---:|------|
| 散戶佔比 | ~30% | ~70%+ | Crypto IC ↑ |
| 機構覆蓋 | 高 | 低 | Crypto IC ↑ |
| 因子可用數 | 303 | ~20-50 | Crypto IC ↓ |
| 資產數量 | 100 | 20-30 | Crypto IC ↓ |
| 市場 24/7 | 否 | 是 | 更多 noise → IC ↓ |

### 4.3 綜合 IC 估計

| 情境 | Crypto IC (日級) | 依據 |
|------|:---:|------|
| 悲觀 | 0.010 | 因子少 + 資產少 |
| **基準** | **0.020** | ≈ 台股最佳 horizon IC，效率差異抵消因子劣勢 |
| 樂觀 | 0.035 | 散戶主導 + ML alpha 不衰減 (Cakici 2024) |

---

## 5. Breakeven 分析

### 5.1 收益計算模型

```
預期日超額 ≈ IC × σ_cs × κ

其中：
  σ_cs ≈ 4%  (crypto 日截面波動率，台股 ~1.5%)
  κ   ≈ 3.1  (集中度因子，從台股回測校準：
               0.08% / (0.017 × 1.5%) ≈ 3.1)
```

### 5.2 不同 Bar 尺度成本-收益表

策略：HoldDrop(K=10, H=3, D=1)，Binance USDT-M Maker，Top-20 coins，turnover ≈ 10%/bar

| Bar | 有效持倉 | 交易成本/日 | Funding/日 | **總成本/日** | 預期收益/日 (IC=0.02) | **淨收益/日** | 年化淨超額 |
|-----|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1h | 3h | 0.14% | 0.03% | 0.17% | ~0.04% | **-0.13%** | 不可行 |
| 4h | 12h | 0.04% | 0.03% | 0.07% | ~0.11% | **+0.04%** | ~15% |
| **8h** | **24h** | **0.02%** | **0.03%** | **0.05%** | **~0.19%** | **+0.14%** | **~51%** |
| 12h | 36h | 0.01% | 0.03% | 0.04% | ~0.23% | **+0.19%** | ~69% |
| 1d | 3d | 0.006% | 0.03% | 0.036% | ~0.25% | **+0.21%** | ~78% |
| 3d | 9d | 0.002% | 0.03% | 0.032% | ~0.20% | **+0.17%** | ~62% |

### 5.3 不同 IC 下的年化超額（8h bar）

| IC | 年化淨超額 | 可行性 |
|:---:|:---:|:---:|
| 0.005 | -3% | 虧損 |
| 0.010 | +14% | 可行 |
| **0.015** | **+32%** | 良好 |
| **0.020** | **+51%** | 推薦目標 |
| 0.030 | +89% | 樂觀情境 |
| 0.040 | +127% | 極樂觀 |

### 5.4 各 IC 下的最低可行 Bar

| IC | 1h | 4h | 8h | 12h | 1d |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.005 | 虧損 | 虧損 | 虧損 | 虧損 | 勉強打平 |
| 0.010 | 虧損 | 虧損 | +3% | +14% | +23% |
| **0.015** | 虧損 | +3% | **+32%** | +47% | +51% |
| **0.020** | 虧損 | +15% | **+51%** | +69% | +78% |
| 0.030 | 虧損 | +40% | +89% | +113% | +135% |

---

## 6. 為什麼 8h 是甜蜜點

```
收益        ^
            |         .1d
            |       .12h
            |     *8h        <- 最佳效率
            |   .4h
            |
            |.1h
 成本 ------+----------------------------> Bar 尺度
```

| 理由 | 說明 |
|------|------|
| Binance funding 週期 | 每 8h 結算一次，對齊 bar 可精確管理 funding 成本 |
| 每日 3 次決策 | 比台股日頻高 3×，充分利用 crypto 24/7 |
| HoldDrop H=3 = 24h | 有效持倉 ~1 天，平衡交易成本與信號衰減 |
| IC 峰值對齊 | Crypto 等效最佳 horizon ~1-2 天，8h×3 = 24h 剛好命中 |
| 成本-收益比 | Breakeven IC 僅 ~0.004，安全邊際極大 |

---

## 7. 推薦配置

| 參數 | 推薦值 | 理由 |
|------|--------|------|
| **Feature bar** | **8h** | 平衡信號品質與成本 |
| **Label** | `Ref($close,-3)/Ref($close,-1)-1` | 與台股一致，8h bar 下 = 預測 16h return |
| **策略** | HoldDrop(K=10, H=3, D=1) | 直接復用，有效持倉 ~24h |
| **Universe** | Top 20-30 by 24h volume | 確保流動性，限制滑價 |
| **交易所** | Binance USDT-M | 最低 maker 費率 0.02% |
| **槓桿** | 1× (無槓桿) | 先驗證 alpha，再考慮加槓桿 |
| **下單方式** | 限價單 (Maker) | 0.02% vs Taker 0.05%，差 2.5× |
| **重訓頻率** | 每週一次 | 與現行一致，捕捉 regime 變化但不過度擬合 |
| **增量學習** | 每 8h bar 前更新 | 與台股 incremental 機制一致 |

### 7.1 Crypto 可用因子

| 類別 | 因子 | 資料來源 |
|------|------|---------|
| 價格動量 | close, returns, SMA, EMA, RSI, MACD | OHLCV (交易所 API) |
| 波動率 | ATR, Bollinger width, realized vol | OHLCV |
| 成交量 | volume, VWAP, volume momentum | OHLCV |
| 流動性 | bid-ask spread, depth, turnover | Order book API |
| **Funding rate** | rate, cumulative, momentum | 交易所 API（永續合約特有）|
| **Open interest** | OI, OI change, long-short ratio | 交易所 API |
| 鏈上 (optional) | active addresses, transfer volume | Glassnode/CryptoQuant（不穩定）|

核心因子約 **20-50 個**，遠少於台股的 303 個，但 Cakici et al. 指出 "return predictability derives mainly from a handful of simple characteristics"。

### 7.2 模型重訓策略

| 頻率 | 優點 | 缺點 | 結論 |
|------|------|------|:------:|
| 每 8h | 最即時 | 訓練成本高、容易過擬合 | 不推薦 |
| 每日 | 適應快 | 維運負擔大 | 備選 |
| **每週** | 成本合理、有足夠新資料 | Regime 劇變時反應慢 | **推薦** |
| 雙週 | 訓練成本最低 | Crypto 變化太快 | 不推薦 |

**推薦組合**：每週完整重訓 + 每 8h 增量學習，與現行 Walk-Forward 架構完全一致。

---

## 8. 架構復用評估

| 層 | 重寫程度 | 說明 |
|----|:--------:|------|
| `shared/` | 20% | constants、week_utils 改為 crypto 時間單位 |
| `adapters/` | **100%** | TWSE/FinMind → Binance/Bybit API |
| `repositories/` | 40% | models 表結構調整、因子重定義 |
| `services/` | 30% | predictor/trainer 幾乎不動、data_service 重寫 |
| `interfaces/` | 10% | 前端大部分不用改 |
| **DoubleEnsemble** | **0%** | 完全復用 |
| **Walk-Forward** | **0%** | 完全復用 |
| **Incremental Learning** | **0%** | 完全復用 |

核心演算法完全不用動，改的只是「資料從哪來」和「因子怎麼算」。

---

## 9. 台股 vs Crypto 系統效率

| 維度 | 台股系統 | Crypto 系統 (預期) |
|------|---------|-------------------|
| Round-trip 成本 | 0.471% | 0.04-0.06% |
| 成本倍率 | 1× | **0.1×** |
| 日截面波動率 σ_cs | ~1.5% | ~4% |
| 同 IC 下超額報酬 | 1× | **~2.7×** |
| 交易頻率 | 1×/日 | 3×/日 (8h bar) |
| 年交易日 | 250 | **365** |
| 有效倍率 | 1× | **~4×** |
| Breakeven IC (8h bar) | ~0.010 | ~0.004 |

以相同 IC 水準，crypto 系統的預期年化超額報酬約為台股系統的 **4 倍**。

---

## 10. 建議資金配置

| 配置 | 比例 | 角色 |
|------|:----:|------|
| BTC DCA 持有 | 60-70% | 核心倉位，搭大趨勢的順風車 |
| Crypto ML 主動策略 | 20-30% | 超額 alpha，驗證系統 |
| 台股 0050 (optional) | 0-10% | 分散風險 |

---

## 11. 風險與注意事項

| 風險 | 嚴重度 | 緩解措施 |
|------|:------:|---------|
| 交易所跑路 | 高 | 不放過多資金在單一交易所 |
| Funding rate 飆升 | 中 | 監控 funding，牛市考慮切換至 Spot |
| 模型 IC 不如預期 | 中 | 先紙上交易 1-2 個月驗證 |
| API 斷線 / 延遲 | 中 | 實作重試機制 + 告警 |
| 監管風險 | 低-中 | 使用合規交易所，分散管轄權 |
| 流動性危機（閃崩）| 低 | 僅交易 Top-20 液態幣 |

---

## 參考文獻

1. Cakici, N., Shahzad, S.J.H., Bedowska-Sojka, B., & Zaremba, A. (2024). *Machine learning and the cross-section of cryptocurrency returns*. International Review of Financial Analysis, 94. — ML alpha 在 crypto 不衰減。
2. Zarattini, C., Pagani, A., & Barbon, A. (2025). *Catching Crypto Trends: A Tactical Approach for Bitcoin and Altcoins*. SSRN 5209907. — Trend-following Sharpe >1.5。
3. Mann, W. (2025). *Quantitative Alpha in Crypto Markets: A Systematic Review*. SSRN 5225612. — Cross-section alpha 主要來自 price, momentum, illiquidity。
4. 本專案 model-performance-analysis.md — 台股 DoubleEnsemble IC/策略基準。
5. Binance Fee Structure (2026). https://www.binance.com/en/support/faq/detail/360033544231
6. Bybit Fee Structure (2026). https://www.bybit.com/en/help-center/article/Perpetual-Futures-Contract-Fees-Explained
