"""
訓練測試腳本

直接測試訓練流程，不經過 API，方便調試。

使用方式：
    python sandbox/train_test.py

輸出：
    - 訓練過程詳細日誌
    - 模型診斷（樹數量、葉子數、importance）
    - IC 值和超參數
"""

import sys
from pathlib import Path

# 加入專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 設置詳細日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 關閉一些噪音
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


def get_session():
    """建立資料庫連線"""
    db_path = project_root / "data" / "data.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return Session()


def diagnose_model(model_path: Path) -> dict:
    """診斷模型狀態"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    dump = model.dump_model()
    num_trees = len(dump.get("tree_info", []))

    first_tree_leaves = 0
    if num_trees > 0:
        first_tree_leaves = dump["tree_info"][0].get("num_leaves", 0)

    importance = model.feature_importance()
    non_zero_imp = (importance > 0).sum()

    return {
        "num_trees": num_trees,
        "first_tree_leaves": first_tree_leaves,
        "total_features": len(importance),
        "non_zero_importance": non_zero_imp,
        "max_importance": int(importance.max()),
        "is_healthy": num_trees > 1 and first_tree_leaves > 1 and non_zero_imp > 0,
    }


def progress_callback(progress: float, message: str):
    """進度回調"""
    bar_width = 30
    filled = int(bar_width * progress / 100)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {progress:5.1f}% {message[:50]:<50}", end="", flush=True)
    if progress >= 100:
        print()


def main():
    print("=" * 60)
    print("訓練測試腳本")
    print("=" * 60)

    # 1. 檢查資料
    session = get_session()

    from src.repositories.factor import FactorRepository
    factor_repo = FactorRepository(session)
    factors = factor_repo.get_enabled()
    print(f"\n[1] 啟用因子數: {len(factors)}")

    # 2. 檢查 Qlib 資料
    qlib_dir = project_root / "data" / "qlib"
    instruments_file = qlib_dir / "instruments" / "all.txt"
    if instruments_file.exists():
        with open(instruments_file) as f:
            instruments = [line.strip().split()[0] for line in f if line.strip()]
        print(f"[2] Qlib 股票數: {len(instruments)}")
    else:
        print("[2] Qlib instruments 檔案不存在，需要先導出資料")
        return

    # 3. 取得可訓練的週
    from src.shared.week_utils import get_trainable_weeks
    from src.repositories.models import StockDaily
    from sqlalchemy import func

    # 查詢資料庫日期範圍
    stmt = session.query(func.min(StockDaily.date), func.max(StockDaily.date))
    data_start, data_end = stmt.one()
    print(f"[3] 資料庫日期範圍: {data_start} ~ {data_end}")

    trainable_weeks = get_trainable_weeks(data_start, data_end, session)
    if not trainable_weeks:
        print("[3] 沒有可訓練的週")
        return

    # 使用最新的可訓練週
    week = trainable_weeks[0]
    print(f"\n[4] 訓練配置:")
    print(f"    Week ID: {week.week_id}")
    print(f"    訓練期: {week.train_start} ~ {week.train_end}")
    print(f"    驗證期: {week.valid_start} ~ {week.valid_end}")

    # 4. 檢查並導出 Qlib 資料
    cal_file = qlib_dir / "calendars" / "day.txt"
    qlib_end = None
    if cal_file.exists():
        with open(cal_file) as f:
            dates = [line.strip() for line in f if line.strip()]
        if dates:
            qlib_end = date.fromisoformat(dates[-1])
            print(f"[4] Qlib 資料範圍: {dates[0]} ~ {dates[-1]}")

    # 如果 Qlib 資料不足，重新導出（與 API 一致使用 90 天 lookback）
    lookback_days = 90  # 因子計算需要的回看期
    required_start = week.train_start - timedelta(days=lookback_days)
    required_end = week.valid_end

    if qlib_end is None or qlib_end < required_end:
        print(f"    需要導出 Qlib 資料: {required_start} ~ {required_end}")
        from src.services.qlib_exporter import QlibExporter, ExportConfig
        exporter = QlibExporter(session)
        export_result = exporter.export(ExportConfig(
            start_date=required_start,
            end_date=required_end,
            output_dir=qlib_dir,
        ))
        print(f"    導出完成: {export_result.stocks_exported} 股票, {export_result.calendar_days} 天")

    # 5. 計算 factor_pool_hash（與 API 一致）
    from src.shared.week_utils import compute_factor_pool_hash
    factor_pool_hash = compute_factor_pool_hash([f.id for f in factors])

    # 6. 初始化訓練器
    from src.services.model_trainer import ModelTrainer

    trainer = ModelTrainer(qlib_data_dir=qlib_dir)

    print(f"\n[6] 開始訓練...")
    print("-" * 60)

    try:
        result = trainer.train(
            session=session,
            train_start=week.train_start,
            train_end=week.train_end,
            valid_start=week.valid_start,
            valid_end=week.valid_end,
            week_id=week.week_id,
            factor_pool_hash=factor_pool_hash,
            on_progress=progress_callback,
        )

        print("\n" + "-" * 60)
        print("[7] 訓練結果:")
        print(f"    模型名稱: {result.model_name}")
        print(f"    選出因子數: {len(result.selected_factor_ids)}")
        print(f"    模型 IC: {result.model_ic:.4f}")
        print(f"    ICIR: {result.icir:.4f}" if result.icir else "    ICIR: N/A")

        # 6. 診斷模型
        model_dir = project_root / "data" / "models" / result.model_name
        model_path = model_dir / "model.pkl"

        if model_path.exists():
            diag = diagnose_model(model_path)
            print(f"\n[8] 模型診斷:")
            print(f"    樹數量: {diag['num_trees']}")
            print(f"    第一棵樹葉子數: {diag['first_tree_leaves']}")
            print(f"    非零 importance 因子: {diag['non_zero_importance']}/{diag['total_features']}")
            print(f"    最大 importance: {diag['max_importance']}")

            if diag["is_healthy"]:
                print(f"\n    ✓ 模型健康")
            else:
                print(f"\n    ✗ 模型異常！（可能正則化過強或資料問題）")

            # 讀取超參數
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                hp = config.get("hyperparameters", {})
                print(f"\n[9] 使用的超參數:")
                print(f"    num_leaves: {hp.get('num_leaves')}")
                print(f"    max_depth: {hp.get('max_depth')}")
                print(f"    lambda_l1: {hp.get('lambda_l1')}")
                print(f"    lambda_l2: {hp.get('lambda_l2')}")
                print(f"    learning_rate: {hp.get('learning_rate')}")

        # 7. 顯示選出的因子
        if result.selected_factor_ids:
            print(f"\n[10] 選出的因子 (按 IC 排序):")
            selected = [r for r in result.all_results if r.selected]
            selected.sort(key=lambda x: abs(x.ic_value), reverse=True)
            for i, r in enumerate(selected[:10]):  # 只顯示前 10 個
                print(f"    {i+1}. {r.factor_name}: IC={r.ic_value:.4f}")
            if len(selected) > 10:
                print(f"    ... 還有 {len(selected) - 10} 個因子")

        # 8. 判斷結果
        print("\n" + "=" * 60)
        is_healthy = diag["is_healthy"] if model_path.exists() else False
        if result.model_ic != 0 and is_healthy:
            print("結果: 訓練成功！")
            print(f"IC = {result.model_ic:.4f}")
        else:
            print("結果: 訓練失敗！")
            if result.model_ic == 0:
                print("  - IC = 0，模型沒有學到任何信號")
            if not is_healthy:
                print("  - 模型結構異常或不存在")
            print("\n可能原因：")
            print("  1. 正則化太強（L1/L2 太大）")
            print("  2. Label 標準化問題")
            print("  3. 資料品質問題")
            print("  4. Qlib 資料範圍不足")

    except Exception as e:
        print(f"\n[ERROR] 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 60)


if __name__ == "__main__":
    main()
