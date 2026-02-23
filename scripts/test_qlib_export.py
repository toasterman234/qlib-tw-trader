"""
測試 Qlib 導出器
"""

from datetime import date
from pathlib import Path

import numpy as np

from src.repositories.database import get_session
from src.services.qlib_exporter import ExportConfig, QlibExporter


def main():
    # 設定導出參數
    config = ExportConfig(
        start_date=date(2022, 1, 1),
        end_date=date(2025, 1, 1),
        output_dir=Path("data/qlib"),
        include_fields=None,  # 全部導出
    )

    # 執行導出
    session = get_session()
    exporter = QlibExporter(session)

    print(f"開始導出: {config.start_date} ~ {config.end_date}")
    print(f"輸出目錄: {config.output_dir}")
    print()

    result = exporter.export(config)

    print("=== 導出結果 ===")
    print(f"導出股票數: {result.stocks_exported}")
    print(f"每股欄位數: {result.fields_per_stock}")
    print(f"總檔案數: {result.total_files}")
    print(f"交易日數: {result.calendar_days}")
    print(f"輸出路徑: {result.output_path}")

    if result.errors:
        print(f"\n錯誤數: {len(result.errors)}")
        for err in result.errors[:5]:  # 只顯示前 5 個
            print(f"  - {err['stock_id']}: {err['error']}")

    # 驗證 .bin 檔案
    print("\n=== 驗證 .bin 檔案 ===")
    test_stock = "2330"
    test_fields = ["close", "adj_close", "volume", "foreign_buy", "revenue"]

    for field in test_fields:
        bin_path = Path(config.output_dir) / "features" / test_stock / f"{field}.day.bin"
        if bin_path.exists():
            data = np.fromfile(bin_path, dtype="<f")
            nan_count = np.isnan(data).sum()
            valid_count = len(data) - nan_count
            print(f"{field}: {len(data)} records, {valid_count} valid, {nan_count} NaN")

            # 顯示最後 5 個有效值
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"  最後 5 個值: {valid_data[-5:]}")
        else:
            print(f"{field}: 檔案不存在")

    session.close()


if __name__ == "__main__":
    main()
