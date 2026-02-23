# IC 增量選擇法（因子篩選）

## 核心洞察

一個因子單獨擁有高 IC，不代表將該因子加入模型後整體 IC 會提高。

因子之間存在共線性與交互效應，單因子 IC 高不等於對模型有增量貢獻。

## 方法流程

### 步驟

1. **計算所有因子的單因子 IC**，按 IC 由高到低排序
2. **迭代訓練 LightGBM**，總訓練次數 = 因子數量
3. **依 IC 排名依序加入因子**：
   - 加入該因子後重新訓練模型
   - 若模型整體 IC 提高 → 保留該因子
   - 若模型整體 IC 未提高 → 剔除該因子
4. 迭代結束後，留下的因子即為最終因子池

### 流程圖

```
所有因子（N 個）
    ↓
計算單因子 IC → 排序（高→低）
    ↓
因子池 = [排名第 1 的因子]
    ↓
for i = 2 to N:
    ├─ 暫時加入第 i 個因子
    ├─ 訓練 LightGBM
    ├─ 計算模型 IC
    ├─ IC 提高？ → Yes → 正式加入因子池
    └─          → No  → 剔除
    ↓
最終因子池（M 個，M ≤ N）
```

## 歷史數據（框架測試階段）

來源：`incremental-learning-design.md`（2026-02-03）

**注意**：此數據為框架測試時的觀察值，未經嚴謹統計分析（無 p 值、信賴區間、年度分解）。

| 指標 | 數值 |
|------|------|
| 平均驗證期 IC | 0.2757 |
| 平均實盤期 IC | 0.0716 |
| IC 衰減 | 76% |
| Valid-Live IC 相關性 | 0.49 |

### 與現行 156 週分析對比

| 指標 | 舊方法（框架測試） | 現行（156 週嚴謹分析） |
|------|-------------------|----------------------|
| Valid IC | 0.2757 | 0.0274 |
| Live IC | 0.0716 | 0.0051 |
| IC 衰減 | 76% | 94.3% (ICIR) |
| Valid-Live 相關 | 0.49 | -0.01 |

### 待釐清

- [ ] 舊方法測試的週數（樣本量）
- [ ] 當時使用的因子數量
- [ ] 當時的 TRAIN_DAYS / VALID_DAYS 配置
- [ ] IC 計算方式是否與現行一致

## 實現細節

### 程式碼位置

| 檔案 | 說明 |
|------|------|
| `src/services/factor_selection/ic_incremental.py` | `ICIncrementalSelector` 類 |
| `src/services/factor_selection/robust.py` | `method="ic_incremental"` 支援 |
| `sandbox/experiment_ic_selection.py` | 實驗腳本（與 RD-Agent 對比） |

### 單因子 IC 計算

```python
# 每日截面 Spearman IC，取平均
daily_ic = factor_data.groupby(level="datetime").apply(
    lambda g: g[factor_name].corr(g["label"], method="spearman")
)
mean_ic = daily_ic.mean()
```

### 模型評估 IC

選擇階段使用快速 LightGBM（`num_boost_round=200`），計算驗證期每日截面 Spearman IC 的平均值。

### 資料處理流程

與現行模型訓練完全一致：

1. `_process_inf()` — 無窮大值替換為欄位均值
2. `_zscore_by_date()` — 每日截面 Z-score 標準化
3. `_rank_by_date()` — Label CSRankNorm（排名百分位 [0, 1]）

### 模型命名

IC 增量模型使用 `{week_id}-icincr` 格式，與現有 RD-Agent 模型 `{week_id}-8d9fdb` 分開存放。

## 實驗設計

見 `reports/ic-selection-experiment.md`（實驗完成後生成）。

控制變數：相同資料、超參數、Label 定義、特徵標準化、LightGBM 訓練設定。
自變數：因子選擇策略。
評判標準：Live IC、超額報酬、配對統計檢定。
