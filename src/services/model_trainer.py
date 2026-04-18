"""
Model training service - DoubleEnsemble with robust factor selection.
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor
from src.repositories.training import TrainingRepository
from src.services.factor_selection import RobustFactorSelector
from src.shared.constants import LABEL_EXPR, LABEL_EXTEND_DAYS, TRAIN_DAYS, TZ_APP, VALID_DAYS
from src.shared.market import get_market

MODELS_DIR = Path("data/models")
TZ_RUNTIME = TZ_APP if isinstance(TZ_APP, ZoneInfo) else ZoneInfo(get_market().timezone)
LGB_DEVICE = os.getenv("LGB_DEVICE", "cpu").strip().lower() or "cpu"
LGB_USE_GPU_DP = LGB_DEVICE == "gpu"


@dataclass
class FactorEvalResult:
    factor_id: int
    factor_name: str
    ic_value: float
    selected: bool


@dataclass
class TrainingResult:
    run_id: int
    model_name: str
    model_ic: float
    icir: float | None
    selected_factor_ids: list[int]
    all_results: list[FactorEvalResult]


DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 5.0,
    "lambda_l2": 5.0,
    "min_data_in_leaf": 30,
    "verbosity": -1,
    "seed": 42,
    "feature_pre_filter": False,
    "device": LGB_DEVICE,
    "gpu_use_dp": LGB_USE_GPU_DP,
}


class ModelTrainer:
    OPTUNA_N_TRIALS = 50
    OPTUNA_TIMEOUT = 300
    DE_NUM_MODELS = 3
    DE_DECAY = 0.5
    DE_EPOCHS = 50

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False
        self._last_ic_std: float | None = None
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._optimized_params: dict | None = None
        self._auto_optuna: bool = False

    def _init_qlib(self, force: bool = False) -> None:
        if self._qlib_initialized and not force:
            return

        try:
            import qlib

            region = get_market().qlib_region
            qlib.init(provider_uri=str(self.data_dir), region=region)
            self._qlib_initialized = True
            self._data_cache.clear()
        except ImportError:
            raise RuntimeError("qlib is not installed. Please run: pip install pyqlib")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def _get_instruments(self) -> list[str]:
        instruments_file = self.data_dir / "instruments" / "all.txt"
        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]
        features_dir = self.data_dir / "features"
        if features_dir.exists():
            return [d.name for d in features_dir.iterdir() if d.is_dir()]
        return []

    def get_data_date_range(self) -> tuple[date | None, date | None]:
        instruments_file = self.data_dir / "instruments" / "all.txt"
        if not instruments_file.exists():
            return None, None
        min_start = None
        max_end = None
        with open(instruments_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        start = date.fromisoformat(parts[1])
                        end = date.fromisoformat(parts[2])
                        if min_start is None or start < min_start:
                            min_start = start
                        if max_end is None or end > max_end:
                            max_end = end
                    except ValueError:
                        continue
        return min_start, max_end

    def _load_data(self, factors: list[Factor], start_date: date, end_date: date) -> pd.DataFrame:
        self._init_qlib(force=True)
        from qlib.data import D
        from datetime import timedelta

        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found in qlib data directory")

        fields = [f.expression for f in factors]
        names = [f.name for f in factors]
        label_expr = LABEL_EXPR
        all_fields = fields + [label_expr]
        all_names = names + ["label"]
        extended_end = end_date + timedelta(days=LABEL_EXTEND_DAYS)

        df = D.features(
            instruments=instruments,
            fields=all_fields,
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=extended_end.strftime("%Y-%m-%d"),
        )
        if df.empty:
            return df
        df.columns = all_names
        dates = df.index.get_level_values("datetime")
        if hasattr(dates[0], "date"):
            mask = pd.Series([d.date() <= end_date for d in dates], index=df.index)
        else:
            mask = dates.date <= end_date
        return df[mask]

    def _prepare_train_valid_data(self, df: pd.DataFrame, train_start: date, train_end: date, valid_start: date, valid_end: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols]
        y = self._rank_by_date(df["label"])
        train_mask = (df.index.get_level_values("datetime").date >= train_start) & (df.index.get_level_values("datetime").date <= train_end)
        valid_mask = (df.index.get_level_values("datetime").date >= valid_start) & (df.index.get_level_values("datetime").date <= valid_end)
        X_train = X[train_mask].dropna()
        X_valid = X[valid_mask].dropna()
        y_train = y[train_mask].dropna()
        y_valid = y[valid_mask].dropna()
        common_train = X_train.index.intersection(y_train.index)
        common_valid = X_valid.index.intersection(y_valid.index)
        return X_train.loc[common_train], X_valid.loc[common_valid], y_train.loc[common_train], y_valid.loc[common_valid]

    def _process_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            mask = np.isinf(df[col])
            if mask.any():
                col_mean = df.loc[~mask, col].mean()
                df.loc[mask, col] = col_mean if not np.isnan(col_mean) else 0
        return df

    def _zscore_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(level="datetime", group_keys=False).apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    def _rank_by_date(self, series: pd.Series) -> pd.Series:
        def rank_pct(x: pd.Series) -> pd.Series:
            return x.rank(pct=True, method="average")
        return series.groupby(level="datetime", group_keys=False).apply(rank_pct)

    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, n_trials: int | None = None, timeout: int | None = None, on_progress: Callable[[float, str], None] | None = None) -> dict:
        import optuna
        from src.services.double_ensemble import DoubleEnsembleModel
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = n_trials or self.OPTUNA_N_TRIALS
        timeout = timeout or self.OPTUNA_TIMEOUT
        X_train_processed = self._process_inf(X_train)
        X_valid_processed = self._process_inf(X_valid)
        X_train_norm = self._zscore_by_date(X_train_processed).fillna(0)
        X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
        n_samples = len(X_train)
        scale_factor = max(0.1, min(1.0, n_samples / 100000))
        lambda_max = max(1.0, 50.0 * scale_factor)
        best_ic = [0.0]
        trial_count = [0]

        def objective(trial: optuna.Trial) -> float:
            lgb_params = {
                "objective": "regression",
                "metric": "mse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": 42,
                "feature_pre_filter": False,
                "device": LGB_DEVICE,
                "gpu_use_dp": LGB_USE_GPU_DP,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.01, lambda_max, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, lambda_max, log=True),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": 5,
            }
            model = DoubleEnsembleModel(num_models=self.DE_NUM_MODELS, epochs=self.DE_EPOCHS, decay=self.DE_DECAY, early_stopping_rounds=10, **lgb_params)
            model.fit(X_train_norm.values, y_train.values, X_valid_norm.values, y_valid.values)
            predictions = model.predict(X_valid_norm.values)
            if np.unique(predictions).size <= 1:
                return 0.0
            pred_df = pd.DataFrame({"pred": predictions, "label": y_valid.values}, index=y_valid.index)
            def calc_ic(g: pd.DataFrame) -> float:
                if len(g) < 10:
                    return np.nan
                if g["pred"].nunique() == 1 or g["label"].nunique() == 1:
                    return np.nan
                return g["pred"].corr(g["label"], method="spearman")
            daily_ic = pred_df.groupby(level="datetime").apply(calc_ic)
            mean_ic = daily_ic.mean()
            ic = float(mean_ic) if not np.isnan(mean_ic) else 0.0
            trial_count[0] += 1
            if ic > best_ic[0]:
                best_ic[0] = ic
            if on_progress:
                on_progress(round(2.0 + (trial_count[0] / n_trials) * 8.0, 1), f"Optuna trial {trial_count[0]}/{n_trials}: IC={ic:.4f} (best: {best_ic[0]:.4f})")
            return ic

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best = study.best_params
        return {"lgb_params": {"objective": "regression", "metric": "mse", "boosting_type": "gbdt", "verbosity": -1, "seed": 42, "feature_pre_filter": False, "device": LGB_DEVICE, "gpu_use_dp": LGB_USE_GPU_DP, "bagging_freq": 5, **best}}

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, params: dict | None = None) -> Any:
        from src.services.double_ensemble import DoubleEnsembleModel
        X_train = self._process_inf(X_train)
        X_valid = self._process_inf(X_valid)
        X_train_norm = self._zscore_by_date(X_train).fillna(0)
        X_valid_norm = self._zscore_by_date(X_valid).fillna(0)
        if params is None:
            params = self._optimized_params or {}
        lgb_params = params.get("lgb_params", DEFAULT_LGB_PARAMS)
        model = DoubleEnsembleModel(num_models=self.DE_NUM_MODELS, epochs=self.DE_EPOCHS, decay=self.DE_DECAY, early_stopping_rounds=10, **lgb_params)
        model.fit(X_train_norm.values, y_train.values, X_valid_norm.values, y_valid.values)
        return model

    def _calculate_prediction_ic(self, model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> float:
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
        predictions = model.predict(X_valid_norm.values)
        pred_df = pd.DataFrame({"pred": predictions, "label": y_valid.values}, index=y_valid.index)
        def calc_spearman_ic(group: pd.DataFrame) -> float:
            if len(group) < 10:
                return np.nan
            if group["pred"].nunique() == 1 or group["label"].nunique() == 1:
                return np.nan
            return group["pred"].corr(group["label"], method="spearman")
        daily_ic = pred_df.groupby(level="datetime").apply(calc_spearman_ic)
        self._last_ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else None
        mean_ic = daily_ic.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    def _calculate_daily_ic(self, model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> np.ndarray:
        X_valid_processed = self._process_inf(X_valid)
        X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
        predictions = model.predict(X_valid_norm.values)
        pred_df = pd.DataFrame({"pred": predictions, "label": y_valid.values}, index=y_valid.index)
        def calc_spearman_ic(group: pd.DataFrame) -> float:
            if len(group) < 10:
                return np.nan
            if group["pred"].nunique() == 1 or group["label"].nunique() == 1:
                return np.nan
            return group["pred"].corr(group["label"], method="spearman")
        daily_ic = pred_df.groupby(level="datetime").apply(calc_spearman_ic)
        return daily_ic.dropna().values

    def train(self, session: Session, train_start: date, train_end: date, valid_start: date, valid_end: date, week_id: str | None = None, factor_pool_hash: str | None = None, on_progress: Callable[[float, str], None] | None = None) -> TrainingResult:
        from src.shared.constants import EMBARGO_DAYS
        factor_repo = FactorRepository(session)
        training_repo = TrainingRepository(session)
        enabled_factors = factor_repo.get_all(enabled=True)
        if not enabled_factors:
            raise ValueError("No enabled factors found")
        candidate_ids = [f.id for f in enabled_factors]
        temp_name = f"{week_id or valid_end.strftime('%Y%m')}-pending"
        run = training_repo.create_run(name=temp_name, train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end, week_id=week_id, factor_pool_hash=factor_pool_hash, embargo_days=EMBARGO_DAYS)
        run.candidate_factor_ids = json.dumps(candidate_ids)
        run.status = "running"
        session.commit()
        if on_progress:
            on_progress(0.0, "Initializing training...")
        try:
            if on_progress:
                on_progress(2.0, "Loading factor data...")
            all_data = self._load_data(factors=enabled_factors, start_date=train_start, end_date=valid_end)
            if all_data.empty:
                raise ValueError("No data available for the specified date range")
            self._optimized_params = {"lgb_params": dict(DEFAULT_LGB_PARAMS)}
            self._auto_optuna = True
            if on_progress:
                on_progress(10.0, "Will auto-tune with Optuna after factor selection")
            selected_factors, all_results, best_model, selection_stats = self._robust_factor_selection(factors=enabled_factors, all_data=all_data, train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end, on_progress=on_progress)
            if best_model is not None and selected_factors:
                factor_names = [f.name for f in selected_factors]
                X_valid = all_data[factor_names]
                y_valid = all_data["label"]
                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid = X_valid[valid_mask].dropna()
                y_valid = y_valid[valid_mask].dropna()
                common_idx = X_valid.index.intersection(y_valid.index)
                X_valid = X_valid.loc[common_idx]
                y_valid = y_valid.loc[common_idx]
                model_ic = self._calculate_prediction_ic(best_model, X_valid, y_valid)
            else:
                model_ic = 0.0
            icir = self._calculate_icir(model_ic, len(selected_factors))
            for result in all_results:
                training_repo.add_factor_result(run_id=run.id, factor_id=result.factor_id, ic_value=result.ic_value, selected=result.selected)
            if week_id and factor_pool_hash:
                model_name = f"{week_id}-{factor_pool_hash}"
            else:
                hash_input = f"{run.id}-{valid_end.isoformat()}-{len(selected_factors)}"
                short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
                model_name = f"{valid_end.strftime('%Y%m')}-{short_hash}"
            incremented_model = best_model
            if best_model is not None and selected_factors:
                if on_progress:
                    on_progress(96.0, "Incremental update with validation data...")
                factor_names = [f.name for f in selected_factors]
                X_valid_incr = all_data[factor_names]
                y_valid_incr = all_data["label"]
                valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & (all_data.index.get_level_values("datetime").date <= valid_end)
                X_valid_incr = X_valid_incr[valid_mask].dropna()
                y_valid_incr = y_valid_incr[valid_mask].dropna()
                common_idx = X_valid_incr.index.intersection(y_valid_incr.index)
                X_valid_incr = X_valid_incr.loc[common_idx]
                y_valid_incr = y_valid_incr.loc[common_idx]
                if not X_valid_incr.empty:
                    X_valid_processed = self._process_inf(X_valid_incr)
                    X_valid_norm = self._zscore_by_date(X_valid_processed).fillna(0)
                    y_valid_rank = self._rank_by_date(y_valid_incr)
                    try:
                        best_model.incremental_update(X_valid_norm.values, y_valid_rank.values, num_boost_round=50)
                        incremented_model = best_model
                        if on_progress:
                            on_progress(98.0, "Incremental update completed")
                    except Exception as e:
                        if on_progress:
                            on_progress(98.0, f"Incremental update failed: {e}, using original model")
                        incremented_model = best_model
            run.name = model_name
            run.selected_factor_ids = json.dumps([f.id for f in selected_factors])
            selection_config = {"method": selection_stats["method"], "incremental_update": True}
            run.selection_method = selection_stats["method"]
            run.selection_config = json.dumps(selection_config)
            run.selection_stats = json.dumps(selection_stats)
            training_repo.complete_run(run_id=run.id, model_ic=model_ic, icir=icir, factor_count=len(selected_factors))
            config = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_start": valid_start.isoformat(),
                "valid_end": valid_end.isoformat(),
                "week_id": week_id,
                "factor_pool_hash": factor_pool_hash,
                "model_ic": model_ic,
                "icir": icir,
                "incremental_updated": incremented_model is not best_model,
                "market": get_market().code,
                "qlib_region": get_market().qlib_region,
                "lgb_device": LGB_DEVICE,
            }
            if self._optimized_params:
                config["hyperparameters"] = self._optimized_params
            self._save_model_files(model_name=model_name, selected_factors=selected_factors, config=config, model=incremented_model)
            if on_progress:
                on_progress(100.0, "Training completed")
            try:
                from src.services.stability import QualityMonitor
                quality_monitor = QualityMonitor(session)
                quality_monitor.compute_and_save(run)
                logger.info(f"Computed quality metrics for training run {run.id}")
            except Exception as qe:
                logger.warning(f"Failed to compute quality metrics: {qe}")
            return TrainingResult(run_id=run.id, model_name=model_name, model_ic=model_ic, icir=icir, selected_factor_ids=[f.id for f in selected_factors], all_results=all_results)
        except Exception as e:
            run.status = "failed"
            run.completed_at = datetime.now(TZ_RUNTIME)
            session.commit()
            raise e

    def _calculate_single_factor_ic(self, factor: Factor, all_data: pd.DataFrame, train_start: date, train_end: date) -> float:
        try:
            train_mask = (all_data.index.get_level_values("datetime").date >= train_start) & (all_data.index.get_level_values("datetime").date <= train_end)
            factor_data = all_data[[factor.name, "label"]][train_mask].dropna()
            if len(factor_data) < 100:
                return 0.0
            def calc_spearman_ic(group: pd.DataFrame) -> float:
                if len(group) < 10:
                    return np.nan
                if group[factor.name].nunique() == 1 or group["label"].nunique() == 1:
                    return np.nan
                return group[factor.name].corr(group["label"], method="spearman")
            daily_ic = factor_data.groupby(level="datetime").apply(calc_spearman_ic)
            mean_ic = daily_ic.mean()
            return float(mean_ic) if not np.isnan(mean_ic) else 0.0
        except Exception:
            return 0.0

    def _robust_factor_selection(self, factors: list[Factor], all_data: pd.DataFrame, train_start: date, train_end: date, valid_start: date, valid_end: date, on_progress: Callable[[float, str], None] | None = None) -> tuple[list[Factor], list[FactorEvalResult], Any, dict]:
        if on_progress:
            on_progress(11.0, "Robust selection: Preparing data...")
        factor_names = [f.name for f in factors]
        X = all_data[factor_names]
        y = all_data["label"]
        train_mask = (all_data.index.get_level_values("datetime").date >= train_start) & (all_data.index.get_level_values("datetime").date <= train_end)
        X_train = X[train_mask]
        y_train = y[train_mask]
        def robust_progress(p: float, msg: str) -> None:
            if on_progress:
                progress = 11.0 + p * 0.79
                on_progress(progress, msg)
        robust_selector = RobustFactorSelector(method="none")
        result = robust_selector.select(factors=factors, X=X_train, y=y_train, on_progress=robust_progress)
        selected_factors = result.selected_factors
        if on_progress:
            on_progress(90.0, f"Factor selection ({result.method}): {len(selected_factors)} factors selected")
        all_results: list[FactorEvalResult] = []
        selected_names = {f.name for f in selected_factors}
        for factor in factors:
            ic = self._calculate_single_factor_ic(factor, all_data, train_start, train_end)
            all_results.append(FactorEvalResult(factor_id=factor.id, factor_name=factor.name, ic_value=ic, selected=factor.name in selected_names))
        best_model = None
        if selected_factors:
            if on_progress:
                on_progress(92.0, "Training final model...")
            selected_factor_names = [f.name for f in selected_factors]
            X_train_selected = X_train[selected_factor_names]
            valid_mask = (all_data.index.get_level_values("datetime").date >= valid_start) & (all_data.index.get_level_values("datetime").date <= valid_end)
            X_valid = X[valid_mask][selected_factor_names]
            y_valid = y[valid_mask]
            train_valid = ~(X_train_selected.isna().any(axis=1) | y_train.isna())
            valid_valid = ~(X_valid.isna().any(axis=1) | y_valid.isna())
            X_train_clean = X_train_selected[train_valid]
            y_train_clean = y_train[train_valid]
            X_valid_clean = X_valid[valid_valid]
            y_valid_clean = y_valid[valid_valid]
            if self._auto_optuna:
                if on_progress:
                    on_progress(92.0, f"Auto-tuning DoubleEnsemble with Optuna ({len(selected_factors)} factors)...")
                def optuna_progress(p: float, msg: str) -> None:
                    if on_progress:
                        progress = 92.0 + p * 0.02
                        on_progress(progress, msg)
                optimized_params = self._optimize_hyperparameters(X_train=X_train_clean, y_train=y_train_clean, X_valid=X_valid_clean, y_valid=y_valid_clean, n_trials=15, timeout=300, on_progress=optuna_progress)
                self._optimized_params = optimized_params
                lgb_best = optimized_params.get("lgb_params", {})
                logger.info(f"Optuna found best LGB params: lr={lgb_best.get('learning_rate', 0):.4f}, leaves={lgb_best.get('num_leaves')}, device={lgb_best.get('device', LGB_DEVICE)}")
                if on_progress:
                    on_progress(94.0, f"Optuna done: lr={lgb_best.get('learning_rate', 0):.4f}, leaves={lgb_best.get('num_leaves')}")
            try:
                best_model = self._train_model(X_train_clean, y_train_clean, X_valid_clean, y_valid_clean, params=self._optimized_params)
            except Exception as e:
                if on_progress:
                    on_progress(95.0, f"Model training failed: {e}")
        selection_stats = {"method": result.method, "initial_factors": len(factors), "final_factors": len(selected_factors), **result.selection_stats, "stage_results": result.stage_results}
        if on_progress:
            on_progress(95.0, f"Factor selection ({result.method}) completed")
        return selected_factors, all_results, best_model, selection_stats

    def _calculate_icir(self, ic: float, factor_count: int) -> float | None:
        if factor_count == 0 or ic == 0:
            return None
        ic_std = self._last_ic_std
        if ic_std is None or ic_std == 0:
            return None
        return ic / ic_std

    def _save_model_files(self, model_name: str, selected_factors: list[Factor], config: dict, model: Any = None) -> None:
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        factors_data = [{"id": f.id, "name": f.name, "expression": f.expression} for f in selected_factors]
        with open(model_dir / "factors.json", "w", encoding="utf-8") as f:
            json.dump(factors_data, f, indent=2, ensure_ascii=False)
        if model is not None:
            with open(model_dir / "model.pkl", "wb") as f:
                pickle.dump(model, f)


def run_training(session: Session, qlib_data_dir: Path | str, train_end: date | None = None, on_progress: Callable[[int, str], None] | None = None) -> TrainingResult:
    from datetime import timedelta
    today = date.today()
    if train_end is None:
        train_end = today - timedelta(days=VALID_DAYS)
    train_start = train_end - timedelta(days=TRAIN_DAYS)
    valid_start = train_end + timedelta(days=1)
    valid_end = today
    trainer = ModelTrainer(qlib_data_dir)
    return trainer.train(session=session, train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end, on_progress=on_progress)
