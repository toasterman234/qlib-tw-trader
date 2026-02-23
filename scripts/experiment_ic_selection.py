"""IC 增量選擇法 vs RD-Agent IC 去重複 — 實驗腳本

實驗設計：
- 控制變數：相同資料、超參數、Label、特徵標準化、LightGBM 訓練設定
- 自變數：因子選擇策略（ic_incremental vs none/dedup）
- 評判標準：Live IC, 超額報酬, 統計檢定

使用方式：
    python sandbox/experiment_ic_selection.py [--phase 1|2|3|4] [--incremental]

Phase 1: 初始化（驗證資料和超參數）
Phase 2: 訓練 IC 增量模型（支援斷點續傳）
Phase 3: Walk-Forward 回測（兩組模型同時回測，確保數據可比）
Phase 4: 比較與報告
"""

import argparse
import json
import logging
import pickle
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from tqdm import tqdm

# 確保專案根目錄在路徑中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repositories.database import get_session
from src.repositories.factor import FactorRepository
from src.repositories.hyperparams import HyperparamsRepository
from src.repositories.walk_forward import WalkForwardBacktestRepository
from src.services.factor_selection.ic_incremental import ICIncrementalSelector
from src.services.model_trainer import ModelTrainer, get_conservative_default_params
from src.shared.week_utils import get_next_week_id, parse_week_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
QLIB_DATA_DIR = Path("data/qlib")
IC_INCR_HASH = "icincr"


# ============================================================
# Phase 1: 初始化
# ============================================================

def phase1_init():
    """驗證資料庫和現有模型"""
    logger.info("=" * 60)
    logger.info("Phase 1: 初始化與驗證")
    logger.info("=" * 60)

    session = get_session()

    # 1. 讀取啟用因子
    factor_repo = FactorRepository(session)
    enabled_factors = factor_repo.get_enabled()
    logger.info(f"啟用因子數：{len(enabled_factors)}")

    # 2. 讀取超參數
    hp_repo = HyperparamsRepository(session)
    hp = hp_repo.get_latest()
    if hp:
        params = json.loads(hp.params_json)
        logger.info(f"超參數組：{hp.name}")
        logger.info(f"  num_leaves={params.get('num_leaves')}, max_depth={params.get('max_depth')}")
    else:
        logger.warning("無培養超參數，將使用保守預設值")

    # 3. 檢查現有 RD-Agent 模型
    existing = list(MODELS_DIR.glob("*-8d9fdb"))
    logger.info(f"現有 RD-Agent 模型：{len(existing)}")

    # 4. 檢查已完成的 IC 增量模型
    ic_incr = list(MODELS_DIR.glob(f"*-{IC_INCR_HASH}"))
    logger.info(f"已完成 IC 增量模型：{len(ic_incr)}")

    # 5. 讀取可訓練週
    trainer = ModelTrainer(QLIB_DATA_DIR)
    data_start, data_end = trainer.get_data_date_range()
    if data_start and data_end:
        logger.info(f"Qlib 資料範圍：{data_start} ~ {data_end}")
    else:
        logger.warning("無法讀取 Qlib 資料範圍，請先導出 Qlib 資料")

    # 6. 讀取 RD-Agent 基線（Walk-Forward 結果，可能分多段）
    wf_repo = WalkForwardBacktestRepository(session)
    recent = wf_repo.get_recent(limit=20)
    completed = [r for r in recent if r.status == "completed"]
    if completed:
        logger.info(f"RD-Agent 基線（{len(completed)} 段）：")
        total_weeks = 0
        for r in completed:
            weekly = json.loads(r.weekly_details)
            total_weeks += len(weekly)
            logger.info(f"  {r.start_week_id} ~ {r.end_week_id}: {len(weekly)} 週")
        logger.info(f"  總計：{total_weeks} 週")
    else:
        logger.warning("無 Walk-Forward 基線數據")

    session.close()
    logger.info("Phase 1 完成")


# ============================================================
# Phase 2: 訓練 IC 增量模型
# ============================================================

def phase2_train():
    """逐週訓練 IC 增量選擇模型（支援斷點續傳）"""
    logger.info("=" * 60)
    logger.info("Phase 2: 訓練 IC 增量模型")
    logger.info("=" * 60)

    session = get_session()

    # 讀取啟用因子
    factor_repo = FactorRepository(session)
    enabled_factors = factor_repo.get_enabled()
    logger.info(f"啟用因子數：{len(enabled_factors)}")

    # 列出所有 RD-Agent 模型的 week_id
    existing_rd = sorted(MODELS_DIR.glob("*-8d9fdb"))
    week_ids = [d.name.split("-")[0] for d in existing_rd]
    logger.info(f"RD-Agent 模型週數：{len(week_ids)}")
    logger.info(f"範圍：{week_ids[0]} ~ {week_ids[-1]}")

    # 讀取超參數（從第一個模型的 config.json）
    first_config_path = existing_rd[0] / "config.json"
    with open(first_config_path) as f:
        first_config = json.load(f)

    hp_from_config = first_config.get("hyperparameters", {})
    if hp_from_config:
        base_params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
            "device": "gpu",
            "gpu_use_dp": False,
            **hp_from_config,
        }
        logger.info(f"使用模型超參數：num_leaves={hp_from_config.get('num_leaves')}, max_depth={hp_from_config.get('max_depth')}")
    else:
        # 從資料庫讀取
        hp_repo = HyperparamsRepository(session)
        hp = hp_repo.get_latest()
        if hp:
            base_params = json.loads(hp.params_json)
            base_params["device"] = "gpu"
            base_params["gpu_use_dp"] = False
            logger.info(f"使用培養超參數：{hp.name}")
        else:
            base_params = get_conservative_default_params(len(enabled_factors))
            base_params["device"] = "gpu"
            base_params["gpu_use_dp"] = False
            logger.info("使用保守預設超參數")

    # 初始化 ModelTrainer（用於資料載入和處理）
    trainer = ModelTrainer(QLIB_DATA_DIR)

    # 導出 Qlib 資料（覆蓋範圍：所有模型的 train_start ~ valid_end）
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    all_configs = []
    for rd_dir in existing_rd:
        cfg_path = rd_dir / "config.json"
        with open(cfg_path) as f:
            all_configs.append(json.load(f))

    export_start = min(date.fromisoformat(c["train_start"]) for c in all_configs) - timedelta(days=60)
    export_end = max(date.fromisoformat(c["valid_end"]) for c in all_configs)

    logger.info(f"導出 Qlib 資料：{export_start} ~ {export_end}")
    exporter = QlibExporter(session)
    exporter.export(ExportConfig(
        start_date=export_start,
        end_date=export_end,
        output_dir=QLIB_DATA_DIR,
    ))

    # 逐週訓練
    completed = 0
    skipped = 0
    failed = 0

    # 先計算需要跳過的數量
    to_skip = sum(
        1 for w in week_ids
        if (MODELS_DIR / f"{w}-{IC_INCR_HASH}" / "model.pkl").exists()
    )
    to_train = len(week_ids) - to_skip
    if to_skip > 0:
        logger.info(f"斷點續傳：跳過 {to_skip} 個已完成模型，剩餘 {to_train} 個")

    pbar = tqdm(week_ids, desc="訓練 IC 增量模型", unit="週")
    for idx, week_id in enumerate(pbar):
        model_name = f"{week_id}-{IC_INCR_HASH}"
        model_dir = MODELS_DIR / model_name

        # 斷點續傳：檢查是否已存在
        if model_dir.exists() and (model_dir / "model.pkl").exists():
            skipped += 1
            pbar.set_postfix(done=completed, skip=skipped, fail=failed)
            continue

        pbar.set_description(f"訓練 {model_name}")

        try:
            # 讀取對應 RD-Agent 模型的 config（取得日期範圍）
            rd_config_path = MODELS_DIR / f"{week_id}-8d9fdb" / "config.json"
            with open(rd_config_path) as f:
                rd_config = json.load(f)

            train_start = date.fromisoformat(rd_config["train_start"])
            train_end = date.fromisoformat(rd_config["train_end"])
            valid_start = date.fromisoformat(rd_config["valid_start"])
            valid_end = date.fromisoformat(rd_config["valid_end"])

            # 載入資料（復用 ModelTrainer._load_data）
            all_data = trainer._load_data(enabled_factors, train_start, valid_end)

            if all_data.empty:
                logger.warning(f"  {week_id}: 無資料，跳過")
                failed += 1
                continue

            # 準備訓練/驗證資料（復用 ModelTrainer._prepare_train_valid_data）
            X_train, X_valid, y_train, y_valid = trainer._prepare_train_valid_data(
                all_data, train_start, train_end, valid_start, valid_end
            )

            if X_train.empty or X_valid.empty:
                logger.warning(f"  {week_id}: 資料不足，跳過")
                failed += 1
                continue

            # IC 增量選擇
            start_time = time.time()

            selector = ICIncrementalSelector(
                lgbm_params=base_params,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            result = selector.select(
                factors=enabled_factors,
                X=X_train,
                y=y_train,
            )

            selected_factors = result.selected_factors
            selection_time = time.time() - start_time

            if not selected_factors:
                logger.warning(f"  {week_id}: 未選出任何因子，跳過")
                failed += 1
                continue

            logger.info(f"  選出 {len(selected_factors)} 個因子 ({selection_time:.1f}s)")

            # 訓練最終模型（使用選出的因子，相同超參數）
            import lightgbm as lgb

            selected_names = [f.name for f in selected_factors]
            X_train_sel = X_train[selected_names]
            X_valid_sel = X_valid[selected_names]

            # 清理 NaN
            train_valid_mask = ~(X_train_sel.isna().any(axis=1) | y_train.isna())
            valid_valid_mask = ~(X_valid_sel.isna().any(axis=1) | y_valid.isna())

            X_tr = X_train_sel[train_valid_mask]
            y_tr = y_train[train_valid_mask]
            X_vl = X_valid_sel[valid_valid_mask]
            y_vl = y_valid[valid_valid_mask]

            # 標準化
            X_tr = trainer._process_inf(X_tr)
            X_tr = trainer._zscore_by_date(X_tr).fillna(0)
            X_vl = trainer._process_inf(X_vl)
            X_vl = trainer._zscore_by_date(X_vl).fillna(0)

            # LightGBM 資料集
            train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
            valid_data = lgb.Dataset(X_vl.values, label=y_vl.values, reference=train_data)

            # 訓練（與現行一致：500 rounds, early_stopping=50）
            best_model = lgb.train(
                base_params,
                train_data,
                num_boost_round=500,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            # 計算驗證期 IC
            model_ic = trainer._calculate_prediction_ic(best_model, X_valid_sel, y_valid)

            # 增量更新（與現行一致）
            factor_names = [f.name for f in selected_factors]
            X_valid_incr = all_data[factor_names]
            y_valid_incr = all_data["label"]

            valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & \
                         (all_data.index.get_level_values("datetime").date <= valid_end)
            X_valid_incr = X_valid_incr[valid_mask].dropna()
            y_valid_incr = y_valid_incr[valid_mask].dropna()
            common_idx = X_valid_incr.index.intersection(y_valid_incr.index)
            X_valid_incr = X_valid_incr.loc[common_idx]
            y_valid_incr = y_valid_incr.loc[common_idx]

            incremented_model = best_model
            if not X_valid_incr.empty:
                X_vi_processed = trainer._process_inf(X_valid_incr)
                X_vi_norm = trainer._zscore_by_date(X_vi_processed).fillna(0)
                y_vi_rank = trainer._rank_by_date(y_valid_incr)

                vi_data = lgb.Dataset(X_vi_norm.values, label=y_vi_rank.values)

                try:
                    incremented_model = lgb.train(
                        base_params,
                        vi_data,
                        num_boost_round=50,
                        init_model=best_model,
                        keep_training_booster=True,
                    )
                except Exception as e:
                    logger.warning(f"  增量更新失敗: {e}")
                    incremented_model = best_model

            # 儲存模型
            model_dir.mkdir(parents=True, exist_ok=True)

            # config.json
            config = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
                "week_id": week_id,
                "factor_pool_hash": IC_INCR_HASH,
                "model_ic": model_ic,
                "incremental_updated": incremented_model is not best_model,
                "selection_method": "ic_incremental",
                "selected_factor_count": len(selected_factors),
                "selection_time_seconds": round(selection_time, 1),
                "hyperparameters": {k: v for k, v in base_params.items()
                                    if k not in ("objective", "metric", "boosting_type", "verbosity", "seed")},
            }
            with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # factors.json
            factors_data = [
                {"id": fac.id, "name": fac.name, "expression": fac.expression}
                for fac in selected_factors
            ]
            with open(model_dir / "factors.json", "w", encoding="utf-8") as f:
                json.dump(factors_data, f, indent=2, ensure_ascii=False)

            # model.pkl
            with open(model_dir / "model.pkl", "wb") as f:
                pickle.dump(incremented_model, f)

            completed += 1
            pbar.set_postfix(
                done=completed, skip=skipped, fail=failed,
                ic=f"{model_ic:.4f}", fac=len(selected_factors),
            )

        except Exception as e:
            logger.error(f"  {week_id} 訓練失敗: {e}")
            failed += 1
            pbar.set_postfix(done=completed, skip=skipped, fail=failed)
            # 清理失敗的目錄
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)

    pbar.close()
    session.close()

    logger.info("=" * 60)
    logger.info(f"Phase 2 完成：完成={completed}, 跳過={skipped}, 失敗={failed}")
    logger.info("=" * 60)


# ============================================================
# Phase 3: Walk-Forward 回測
# ============================================================

def phase3_walkforward(enable_incremental: bool = False):
    """同時回測兩組模型，確保數據可比（使用 WalkForwardBacktester API）"""
    logger.info("=" * 60)
    logger.info("Phase 3: Walk-Forward 回測（兩組模型同時回測）")
    logger.info(f"  增量學習: {'啟用' if enable_incremental else '停用'}")
    logger.info("=" * 60)

    session = get_session()

    # 收集兩組模型共同的 week_ids
    rd_dirs = sorted(MODELS_DIR.glob("*-8d9fdb"))
    ic_dirs = sorted(MODELS_DIR.glob(f"*-{IC_INCR_HASH}"))

    rd_weeks = {d.name.split("-")[0] for d in rd_dirs}
    ic_weeks = {d.name.split("-")[0] for d in ic_dirs}
    common_week_ids = sorted(rd_weeks & ic_weeks)

    if not common_week_ids:
        logger.error("無共同模型週，請先執行 Phase 2")
        return

    logger.info(f"共同模型週數：{len(common_week_ids)}")
    logger.info(f"範圍：{common_week_ids[0]} ~ {common_week_ids[-1]}")

    # 計算日期範圍
    first_predict_week = get_next_week_id(common_week_ids[0])
    last_predict_week = get_next_week_id(common_week_ids[-1])

    y, w = parse_week_id(first_predict_week)
    first_predict_date = date.fromisocalendar(y, w, 1)
    y, w = parse_week_id(last_predict_week)
    last_predict_date = date.fromisocalendar(y, w, 5)

    # 匯出 Qlib 資料（一次，全範圍）
    from src.services.qlib_exporter import ExportConfig, QlibExporter
    from src.services.walk_forward_backtester import WalkForwardBacktester

    lookback_days = 180
    export_start = first_predict_date - timedelta(days=lookback_days)

    logger.info(f"導出 Qlib 資料：{export_start} ~ {last_predict_date}")
    exporter = QlibExporter(session)
    exporter.export(ExportConfig(
        start_date=export_start,
        end_date=last_predict_date,
        output_dir=QLIB_DATA_DIR,
    ))

    # 使用 WalkForwardBacktester（共用同一份快取）
    backtester = WalkForwardBacktester(session, QLIB_DATA_DIR)
    backtester._init_qlib()

    # 預載入全量特徵（RD-Agent 250 因子）
    rd_first_dir = MODELS_DIR / f"{common_week_ids[0]}-8d9fdb"
    with open(rd_first_dir / "factors.json") as f:
        rd_factors = json.load(f)

    logger.info("預載入特徵和價格資料...")
    backtester._preload_data(
        factors=rd_factors,
        start_date=first_predict_date,
        end_date=last_predict_date,
        trade_price="open",
    )

    # 初始化 IncrementalLearner（如果啟用）
    incremental_learner = None
    if enable_incremental:
        from src.services.incremental_learner import IncrementalLearner
        incremental_learner = IncrementalLearner(session)

    # 逐週同時回測兩組模型
    rd_results = []
    ic_results = []

    for week_id in tqdm(common_week_ids, desc="Walk-Forward 回測", unit="週"):
        predict_week = get_next_week_id(week_id)
        y, w = parse_week_id(predict_week)
        predict_start = date.fromisocalendar(y, w, 1)
        predict_end = date.fromisocalendar(y, w, 5)

        # 回測兩組模型（共用 backtester 的 cache 和計算方法）
        rd_entry = _backtest_one_model(
            backtester, week_id, "8d9fdb",
            predict_week, predict_start, predict_end,
            incremental_learner,
        )
        ic_entry = _backtest_one_model(
            backtester, week_id, IC_INCR_HASH,
            predict_week, predict_start, predict_end,
            incremental_learner,
        )

        # 統一 market_return：使用 RD-Agent 的（股票池較大）
        if rd_entry.get("market_return") is not None:
            ic_entry["market_return"] = rd_entry["market_return"]

        rd_results.append(rd_entry)
        ic_results.append(ic_entry)

    backtester._clear_cache()

    # 儲存結果
    rd_path = Path("data/experiment_rd_results.json")
    ic_path = Path("data/experiment_ic_incr_results.json")
    with open(rd_path, "w", encoding="utf-8") as f:
        json.dump(rd_results, f, indent=2, ensure_ascii=False)
    with open(ic_path, "w", encoding="utf-8") as f:
        json.dump(ic_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Phase 3 完成：{len(rd_results)} 週")
    logger.info(f"  RD-Agent 結果 → {rd_path}")
    logger.info(f"  IC 增量結果 → {ic_path}")

    session.close()


def _backtest_one_model(
    backtester,
    week_id: str,
    model_hash: str,
    predict_week: str,
    predict_start: date,
    predict_end: date,
    incremental_learner=None,
) -> dict:
    """使用 WalkForwardBacktester API 回測單一模型的單一週"""
    model_name = f"{week_id}-{model_hash}"

    empty_result = {
        "predict_week": predict_week,
        "model_week": week_id,
        "valid_ic": None,
        "live_ic": None,
        "week_return": None,
        "market_return": None,
        "factor_count": 0,
    }

    try:
        model, factors, config = backtester._load_model(model_name)

        # 增量學習（如果啟用）
        if incremental_learner is not None:
            train_end_str = config.get("train_end")
            if train_end_str:
                model_train_end = date.fromisoformat(train_end_str)
                target_date = predict_start - timedelta(days=1)
                result = incremental_learner.update_to_date(
                    base_model=model,
                    factors=factors,
                    model_train_end=model_train_end,
                    target_date=target_date,
                )
                if result is not None:
                    model, _ = result

        # 預測（使用 backtester 的 _predict_week，共用 features_cache）
        predictions = backtester._predict_week(model, factors, predict_start, predict_end)

        if predictions.empty:
            return empty_result

        # 計算 Live IC（使用 backtester 的方法，共用 price_cache）
        live_ic = backtester._calculate_live_ic(predictions, predict_start, predict_end)

        # 計算週收益
        week_return, market_return = backtester._calculate_week_return(
            predictions, predict_start, predict_end, 10, "open"
        )

        return {
            "predict_week": predict_week,
            "model_week": week_id,
            "valid_ic": config.get("model_ic"),
            "live_ic": live_ic,
            "week_return": week_return,
            "market_return": market_return,
            "factor_count": len(factors),
        }

    except Exception as e:
        logger.error(f"  {model_name} @ {predict_week} 失敗: {e}")
        return empty_result


# ============================================================
# Phase 4: 比較與報告
# ============================================================

def phase4_report():
    """比較兩種方法並生成報告"""
    logger.info("=" * 60)
    logger.info("Phase 4: 比較與報告")
    logger.info("=" * 60)

    # 1. 讀取兩組結果（來自 Phase 3 同時回測）
    rd_path = Path("data/experiment_rd_results.json")
    ic_path = Path("data/experiment_ic_incr_results.json")

    if not rd_path.exists() or not ic_path.exists():
        logger.error("無回測結果，請先執行 Phase 3")
        return

    with open(rd_path) as f:
        rd_weekly = json.load(f)
    with open(ic_path) as f:
        ic_incr_results = json.load(f)

    logger.info(f"RD-Agent 結果：{len(rd_weekly)} 週")
    logger.info(f"IC 增量結果：{len(ic_incr_results)} 週")

    # 2. 對齊兩組資料（by predict_week）
    rd_map = {r["predict_week"]: r for r in rd_weekly}
    ic_map = {r["predict_week"]: r for r in ic_incr_results}

    common_weeks = sorted(set(rd_map.keys()) & set(ic_map.keys()))
    logger.info(f"共同週數：{len(common_weeks)}")

    # 3. 計算比較指標
    rd_live_ics = []
    ic_live_ics = []
    rd_returns = []
    ic_returns = []
    rd_market_returns = []
    ic_market_returns = []
    factor_counts = []

    for week in common_weeks:
        rd = rd_map[week]
        ic = ic_map[week]

        if rd.get("live_ic") is not None and ic.get("live_ic") is not None:
            rd_live_ics.append(rd["live_ic"])
            ic_live_ics.append(ic["live_ic"])

        if rd.get("week_return") is not None and ic.get("week_return") is not None:
            rd_returns.append(rd["week_return"])
            ic_returns.append(ic["week_return"])
            if rd.get("market_return") is not None and ic.get("market_return") is not None:
                rd_market_returns.append(rd["market_return"])
                ic_market_returns.append(ic["market_return"])

        if ic.get("factor_count"):
            factor_counts.append(ic["factor_count"])

    # 4. IC 統計
    rd_ic_arr = np.array(rd_live_ics)
    ic_ic_arr = np.array(ic_live_ics)

    rd_ic_stats = _compute_ic_stats(rd_ic_arr)
    ic_ic_stats = _compute_ic_stats(ic_ic_arr)

    # 5. 收益統計
    rd_ret_arr = np.array(rd_returns)
    ic_ret_arr = np.array(ic_returns)
    rd_mkt_arr = np.array(rd_market_returns)
    ic_mkt_arr = np.array(ic_market_returns)

    rd_ret_stats = _compute_return_stats(rd_ret_arr, rd_mkt_arr)
    ic_ret_stats = _compute_return_stats(ic_ret_arr, ic_mkt_arr)

    # 6. 配對統計檢定
    paired_tests = _compute_paired_tests(rd_live_ics, ic_live_ics, rd_returns, ic_returns, rd_market_returns, ic_market_returns)

    # 7. 因子分析
    factor_analysis = _analyze_factors()

    # 8. 年度分解
    yearly = _compute_yearly_breakdown(common_weeks, rd_map, ic_map)

    # 9. 生成報告
    report = _generate_report(
        rd_ic_stats=rd_ic_stats,
        ic_ic_stats=ic_ic_stats,
        rd_ret_stats=rd_ret_stats,
        ic_ret_stats=ic_ret_stats,
        paired_tests=paired_tests,
        factor_analysis=factor_analysis,
        yearly=yearly,
        n_common_weeks=len(common_weeks),
        n_ic_weeks=len(rd_live_ics),
        n_ret_weeks=len(rd_returns),
    )

    report_path = Path("reports/ic-selection-experiment.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"報告已儲存到 {report_path}")


# ============================================================
# 統計分析函式
# ============================================================

def _compute_ic_stats(ic_arr: np.ndarray) -> dict:
    """計算 IC 統計量"""
    if len(ic_arr) == 0:
        return {"mean": 0, "std": 0, "icir": 0, "positive_rate": 0, "p_value": 1.0, "ci_lower": 0, "ci_upper": 0}

    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    icir = mean_ic / std_ic if std_ic > 0 else 0

    # t-test: IC 顯著不為零
    t_stat, p_value = stats.ttest_1samp(ic_arr, 0)

    # 95% 信賴區間
    se = std_ic / np.sqrt(len(ic_arr))
    ci_lower = mean_ic - 1.96 * se
    ci_upper = mean_ic + 1.96 * se

    positive_rate = float(np.sum(ic_arr > 0) / len(ic_arr) * 100)

    return {
        "mean": round(mean_ic, 6),
        "std": round(std_ic, 6),
        "icir": round(icir, 4),
        "positive_rate": round(positive_rate, 1),
        "p_value": round(float(p_value), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "n": len(ic_arr),
    }


def _compute_return_stats(ret_arr: np.ndarray, mkt_arr: np.ndarray) -> dict:
    """計算收益統計量"""
    if len(ret_arr) == 0:
        return {"cumulative": 0, "market": 0, "excess": 0, "sharpe": 0, "win_rate": 0}

    # 累積收益
    cumulative = 1.0
    for r in ret_arr:
        cumulative *= (1 + r / 100)
    cumulative_return = (cumulative - 1) * 100

    # 市場累積
    market_cumulative = 1.0
    for r in mkt_arr:
        market_cumulative *= (1 + r / 100)
    market_return = (market_cumulative - 1) * 100

    excess = cumulative_return - market_return

    # Sharpe（年化）
    sharpe = 0.0
    if len(ret_arr) >= 2 and np.std(ret_arr) > 0:
        sharpe = float(np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(52))

    # 週勝率（超越市場）
    excess_arr = ret_arr - mkt_arr[:len(ret_arr)]
    win_rate = float(np.sum(excess_arr > 0) / len(excess_arr) * 100) if len(excess_arr) > 0 else 0

    # 超額收益 t-test
    t_stat, p_value = stats.ttest_1samp(excess_arr, 0) if len(excess_arr) >= 2 else (0, 1)

    return {
        "cumulative": round(cumulative_return, 2),
        "market": round(market_return, 2),
        "excess": round(excess, 2),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 1),
        "p_value": round(float(p_value), 6),
        "n": len(ret_arr),
    }


def _compute_paired_tests(
    rd_ics: list, ic_ics: list,
    rd_rets: list, ic_rets: list,
    rd_mkts: list, ic_mkts: list,
) -> dict:
    """配對統計檢定"""
    result = {}

    # IC 配對 t-test
    if len(rd_ics) >= 2 and len(rd_ics) == len(ic_ics):
        ic_diff = np.array(ic_ics) - np.array(rd_ics)
        t_stat, p_value = stats.ttest_1samp(ic_diff, 0)
        w_stat, w_pvalue = stats.wilcoxon(ic_diff)
        result["ic"] = {
            "mean_diff": round(float(np.mean(ic_diff)), 6),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(p_value), 6),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_pvalue": round(float(w_pvalue), 6),
        }

    # 超額收益配對 t-test
    if len(rd_rets) >= 2 and len(rd_rets) == len(ic_rets):
        # 計算各自的超額收益
        rd_excess = np.array(rd_rets) - np.array(rd_mkts[:len(rd_rets)])
        ic_excess = np.array(ic_rets) - np.array(ic_mkts[:len(ic_rets)])
        excess_diff = ic_excess - rd_excess
        t_stat, p_value = stats.ttest_1samp(excess_diff, 0)
        w_stat, w_pvalue = stats.wilcoxon(excess_diff)
        result["excess_return"] = {
            "mean_diff": round(float(np.mean(excess_diff)), 4),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(p_value), 6),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_pvalue": round(float(w_pvalue), 6),
        }

    return result


def _analyze_factors() -> dict:
    """分析 IC 增量選出的因子"""
    ic_incr_dirs = sorted(MODELS_DIR.glob(f"*-{IC_INCR_HASH}"))
    if not ic_incr_dirs:
        return {}

    factor_freq: dict[str, int] = {}
    factor_counts = []
    all_sets: list[set[str]] = []

    for model_dir in ic_incr_dirs:
        factors_path = model_dir / "factors.json"
        if factors_path.exists():
            with open(factors_path) as f:
                factors = json.load(f)
            names = {fac["name"] for fac in factors}
            factor_counts.append(len(names))
            all_sets.append(names)
            for name in names:
                factor_freq[name] = factor_freq.get(name, 0) + 1

    # Top-20 最常被選中的因子
    top20 = sorted(factor_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    # 因子穩定性（相鄰週 Jaccard 相似度）
    jaccard_scores = []
    for i in range(1, len(all_sets)):
        intersection = len(all_sets[i] & all_sets[i - 1])
        union = len(all_sets[i] | all_sets[i - 1])
        if union > 0:
            jaccard_scores.append(intersection / union)

    return {
        "mean_factor_count": round(float(np.mean(factor_counts)), 1) if factor_counts else 0,
        "std_factor_count": round(float(np.std(factor_counts)), 1) if factor_counts else 0,
        "min_factor_count": int(np.min(factor_counts)) if factor_counts else 0,
        "max_factor_count": int(np.max(factor_counts)) if factor_counts else 0,
        "top20": top20,
        "mean_jaccard": round(float(np.mean(jaccard_scores)), 4) if jaccard_scores else 0,
    }


def _compute_yearly_breakdown(
    common_weeks: list[str],
    rd_map: dict,
    ic_map: dict,
) -> dict:
    """年度分解分析"""
    yearly = {}

    for week in common_weeks:
        rd = rd_map[week]
        ic = ic_map[week]

        # 從 predict_week 取年份
        year = week[:4]

        if year not in yearly:
            yearly[year] = {"rd_ics": [], "ic_ics": [], "rd_rets": [], "ic_rets": [],
                            "rd_mkts": [], "ic_mkts": []}

        if rd.get("live_ic") is not None and ic.get("live_ic") is not None:
            yearly[year]["rd_ics"].append(rd["live_ic"])
            yearly[year]["ic_ics"].append(ic["live_ic"])

        if rd.get("week_return") is not None and ic.get("week_return") is not None:
            yearly[year]["rd_rets"].append(rd["week_return"])
            yearly[year]["ic_rets"].append(ic["week_return"])
            if rd.get("market_return") is not None and ic.get("market_return") is not None:
                yearly[year]["rd_mkts"].append(rd["market_return"])
                yearly[year]["ic_mkts"].append(ic["market_return"])

    result = {}
    for year, data in sorted(yearly.items()):
        result[year] = {
            "rd_ic": _compute_ic_stats(np.array(data["rd_ics"])) if data["rd_ics"] else None,
            "ic_ic": _compute_ic_stats(np.array(data["ic_ics"])) if data["ic_ics"] else None,
            "rd_ret": _compute_return_stats(np.array(data["rd_rets"]), np.array(data["rd_mkts"])) if data["rd_rets"] else None,
            "ic_ret": _compute_return_stats(np.array(data["ic_rets"]), np.array(data["ic_mkts"])) if data["ic_rets"] else None,
        }

    return result


def _generate_report(
    rd_ic_stats: dict,
    ic_ic_stats: dict,
    rd_ret_stats: dict,
    ic_ret_stats: dict,
    paired_tests: dict,
    factor_analysis: dict,
    yearly: dict,
    n_common_weeks: int,
    n_ic_weeks: int,
    n_ret_weeks: int,
) -> str:
    """生成 Markdown 報告"""

    lines = [
        "# IC 增量選擇法 vs RD-Agent IC 去重複 — 實驗報告",
        "",
        "## 實驗設定",
        "",
        "| 項目 | 設定 |",
        "|------|------|",
        f"| 共同回測週數 | {n_common_weeks} |",
        f"| IC 比較週數 | {n_ic_weeks} |",
        f"| 收益比較週數 | {n_ret_weeks} |",
        "| 訓練期 | 504 交易日（~2 年） |",
        "| 驗證期 | 100 交易日（~4 個月） |",
        "| Embargo | 7 交易日 |",
        "| Label | Ref($close, -2) / Ref($close, -1) - 1 + CSRankNorm |",
        "| Top-K | 10（等權重） |",
        "| 交易價格 | Open |",
        "| 超參數 | 相同（從資料庫培養超參數） |",
        "| 唯一差異 | 因子選擇策略 |",
        "",
        "### 數據可比性",
        "",
        "兩組模型在 Phase 3 中**同時回測**，確保：",
        "- 同一份 Qlib 導出（相同 instruments）",
        "- 同一份 features_cache 和 price_cache",
        "- 相同的市場收益計算邏輯（使用 RD-Agent 的股票池）",
        "- 增量學習設定一致",
        "",
        "## 1. Live IC 比較",
        "",
        "| 指標 | RD-Agent | IC 增量 | 差異 |",
        "|------|----------|--------|------|",
    ]

    # IC 比較表
    for key, label in [
        ("mean", "Mean IC"),
        ("std", "Std IC"),
        ("icir", "ICIR"),
        ("positive_rate", "Positive Rate (%)"),
        ("p_value", "p-value (IC≠0)"),
        ("ci_lower", "95% CI Lower"),
        ("ci_upper", "95% CI Upper"),
    ]:
        rd_val = rd_ic_stats.get(key, "N/A")
        ic_val = ic_ic_stats.get(key, "N/A")
        if isinstance(rd_val, (int, float)) and isinstance(ic_val, (int, float)):
            diff = ic_val - rd_val
            diff_str = f"{diff:+.6f}" if key not in ("positive_rate",) else f"{diff:+.1f}"
        else:
            diff_str = "-"
        lines.append(f"| {label} | {rd_val} | {ic_val} | {diff_str} |")

    lines += [
        "",
        "## 2. 超額報酬比較",
        "",
        "| 指標 | RD-Agent | IC 增量 | 差異 |",
        "|------|----------|--------|------|",
    ]

    for key, label in [
        ("cumulative", "累積收益 (%)"),
        ("market", "市場收益 (%)"),
        ("excess", "超額收益 (%)"),
        ("sharpe", "Sharpe Ratio"),
        ("win_rate", "週勝率 (%)"),
        ("p_value", "p-value (excess≠0)"),
    ]:
        rd_val = rd_ret_stats.get(key, "N/A")
        ic_val = ic_ret_stats.get(key, "N/A")
        if isinstance(rd_val, (int, float)) and isinstance(ic_val, (int, float)):
            diff = ic_val - rd_val
            diff_str = f"{diff:+.2f}" if key != "p_value" else f"{diff:+.6f}"
        else:
            diff_str = "-"
        lines.append(f"| {label} | {rd_val} | {ic_val} | {diff_str} |")

    # 統計檢定
    lines += [
        "",
        "## 3. 統計檢定（配對比較）",
        "",
    ]

    if "ic" in paired_tests:
        ic_test = paired_tests["ic"]
        lines += [
            "### Live IC 差異",
            "",
            f"- IC 增量 - RD-Agent 平均差異: **{ic_test['mean_diff']}**",
            f"- Paired t-test: t={ic_test['paired_t_stat']}, p={ic_test['paired_t_pvalue']}",
            f"- Wilcoxon signed-rank: W={ic_test['wilcoxon_stat']}, p={ic_test['wilcoxon_pvalue']}",
            "",
        ]

    if "excess_return" in paired_tests:
        ret_test = paired_tests["excess_return"]
        lines += [
            "### 超額報酬差異",
            "",
            f"- IC 增量 - RD-Agent 平均差異: **{ret_test['mean_diff']}%**",
            f"- Paired t-test: t={ret_test['paired_t_stat']}, p={ret_test['paired_t_pvalue']}",
            f"- Wilcoxon signed-rank: W={ret_test['wilcoxon_stat']}, p={ret_test['wilcoxon_pvalue']}",
            "",
        ]

    # 因子分析
    lines += [
        "## 4. 因子選擇分析",
        "",
    ]

    if factor_analysis:
        lines += [
            f"- 平均選出因子數：{factor_analysis['mean_factor_count']} ± {factor_analysis['std_factor_count']}",
            f"- 因子數範圍：{factor_analysis['min_factor_count']} ~ {factor_analysis['max_factor_count']}",
            f"- 相鄰週 Jaccard 相似度：{factor_analysis['mean_jaccard']}",
            "",
            "### 最常被選中的因子 Top-20",
            "",
            "| 排名 | 因子名稱 | 被選中次數 |",
            "|------|----------|------------|",
        ]
        for i, (name, count) in enumerate(factor_analysis.get("top20", []), 1):
            lines.append(f"| {i} | {name} | {count} |")

    # 年度分解
    lines += [
        "",
        "## 5. 年度分解",
        "",
    ]

    for year, data in sorted(yearly.items()):
        lines += [f"### {year}", ""]

        if data.get("rd_ic") and data.get("ic_ic"):
            lines += [
                "| IC 指標 | RD-Agent | IC 增量 |",
                "|---------|----------|--------|",
                f"| Mean IC | {data['rd_ic']['mean']} | {data['ic_ic']['mean']} |",
                f"| ICIR | {data['rd_ic']['icir']} | {data['ic_ic']['icir']} |",
                f"| Positive Rate | {data['rd_ic']['positive_rate']}% | {data['ic_ic']['positive_rate']}% |",
                "",
            ]

        if data.get("rd_ret") and data.get("ic_ret"):
            lines += [
                "| 收益指標 | RD-Agent | IC 增量 |",
                "|----------|----------|--------|",
                f"| 累積收益 | {data['rd_ret']['cumulative']}% | {data['ic_ret']['cumulative']}% |",
                f"| 超額收益 | {data['rd_ret']['excess']}% | {data['ic_ret']['excess']}% |",
                f"| 週勝率 | {data['rd_ret']['win_rate']}% | {data['ic_ret']['win_rate']}% |",
                "",
            ]

    # 結論
    lines += [
        "## 6. 結論",
        "",
        _generate_conclusion(rd_ic_stats, ic_ic_stats, rd_ret_stats, ic_ret_stats, paired_tests),
    ]

    return "\n".join(lines)


def _generate_conclusion(
    rd_ic: dict, ic_ic: dict,
    rd_ret: dict, ic_ret: dict,
    paired: dict,
) -> str:
    """根據數據自動生成結論"""
    parts = []

    # IC 比較
    ic_diff = ic_ic.get("mean", 0) - rd_ic.get("mean", 0)
    if ic_diff > 0:
        parts.append(f"IC 增量選擇法的 Live IC ({ic_ic['mean']}) 高於 RD-Agent ({rd_ic['mean']})")
    else:
        parts.append(f"RD-Agent 的 Live IC ({rd_ic['mean']}) 高於 IC 增量選擇法 ({ic_ic['mean']})")

    # ICIR 比較
    icir_diff = ic_ic.get("icir", 0) - rd_ic.get("icir", 0)
    if icir_diff > 0:
        parts.append(f"ICIR 方面 IC 增量 ({ic_ic['icir']}) 優於 RD-Agent ({rd_ic['icir']})")
    else:
        parts.append(f"ICIR 方面 RD-Agent ({rd_ic['icir']}) 優於 IC 增量 ({ic_ic['icir']})")

    # 超額收益比較
    excess_diff = ic_ret.get("excess", 0) - rd_ret.get("excess", 0)
    if excess_diff > 0:
        parts.append(f"超額報酬 IC 增量 ({ic_ret['excess']}%) 高於 RD-Agent ({rd_ret['excess']}%)")
    else:
        parts.append(f"超額報酬 RD-Agent ({rd_ret['excess']}%) 高於 IC 增量 ({ic_ret['excess']}%)")

    # 統計顯著性
    if "ic" in paired:
        p = paired["ic"]["paired_t_pvalue"]
        if p < 0.05:
            parts.append(f"Live IC 差異**統計顯著** (p={p})")
        else:
            parts.append(f"Live IC 差異**不顯著** (p={p})")

    if "excess_return" in paired:
        p = paired["excess_return"]["paired_t_pvalue"]
        if p < 0.05:
            parts.append(f"超額報酬差異**統計顯著** (p={p})")
        else:
            parts.append(f"超額報酬差異**不顯著** (p={p})")

    return "- " + "\n- ".join(parts)


# ============================================================
# 主程式
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IC 增量選擇法 vs RD-Agent 實驗")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="執行特定階段（1=初始化, 2=訓練, 3=回測, 4=報告）",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Phase 3 啟用增量學習（預設停用）",
    )
    args = parser.parse_args()

    if args.phase:
        phases = [args.phase]
    else:
        phases = [1, 2, 3, 4]

    for phase in phases:
        if phase == 1:
            phase1_init()
        elif phase == 2:
            phase2_train()
        elif phase == 3:
            phase3_walkforward(enable_incremental=args.incremental)
        elif phase == 4:
            phase4_report()
