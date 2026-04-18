"""Model management API."""

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.interfaces.schemas.model import (
    DataRange,
    FactorSummary,
    ModelHistoryResponse,
    ModelMetrics,
    ModelResponse,
    ModelSummary,
    Period,
    QualityMetricsItem,
    QualityResponse,
    SelectionInfo,
    TrainBatchRequest,
    TrainRequest,
    TrainResponse,
    WeekModel,
    WeekSlot,
    WeeksResponse,
)
from src.repositories.factor import FactorRepository
from src.repositories.training import TrainingRepository
from src.services.job_manager import broadcast_data_updated
from src.shared.constants import LOOKBACK_DAYS
from src.shared.week_utils import compute_factor_pool_hash, get_trainable_weeks

router = APIRouter()
MODELS_DIR = Path("data/models")


def _parse_factor_ids(json_str: str | None) -> list[int]:
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []


def _run_to_summary(run) -> ModelSummary:
    train_period = None
    valid_period = None
    if run.train_start and run.train_end:
        train_period = Period(start=run.train_start, end=run.train_end)
    if run.valid_start and run.valid_end:
        valid_period = Period(start=run.valid_start, end=run.valid_end)
    candidate_ids = _parse_factor_ids(run.candidate_factor_ids)
    selected_ids = _parse_factor_ids(run.selected_factor_ids)
    return ModelSummary(
        id=f"m{run.id:03d}",
        name=run.name,
        status=run.status,
        trained_at=run.completed_at or run.started_at,
        train_period=train_period,
        valid_period=valid_period,
        metrics=ModelMetrics(ic=float(run.model_ic) if run.model_ic else None, icir=float(run.icir) if run.icir else None),
        factor_count=run.factor_count or len(selected_ids),
        candidate_count=len(candidate_ids) if candidate_ids else None,
        selection_method=run.selection_method,
    )


def _run_to_response(run, factors: list[str], candidate_factors: list[FactorSummary] | None = None, selected_factors: list[FactorSummary] | None = None) -> ModelResponse:
    train_period = None
    valid_period = None
    if run.train_start and run.train_end:
        train_period = Period(start=run.train_start, end=run.train_end)
    if run.valid_start and run.valid_end:
        valid_period = Period(start=run.valid_start, end=run.valid_end)
    duration = None
    if run.started_at and run.completed_at:
        duration = int((run.completed_at - run.started_at).total_seconds())
    selection = None
    if run.selection_method:
        selection = SelectionInfo(method=run.selection_method, config=json.loads(run.selection_config) if run.selection_config else None, stats=json.loads(run.selection_stats) if run.selection_stats else None)
    return ModelResponse(
        id=f"m{run.id:03d}",
        name=run.name,
        status=run.status,
        trained_at=run.completed_at or run.started_at,
        factor_count=run.factor_count,
        factors=factors,
        train_period=train_period,
        valid_period=valid_period,
        metrics=ModelMetrics(ic=float(run.model_ic) if run.model_ic else None, icir=float(run.icir) if run.icir else None),
        training_duration_seconds=duration,
        candidate_factors=candidate_factors or [],
        selected_factors=selected_factors or [],
        selection=selection,
    )


def _parse_model_id(model_id: str) -> int:
    if model_id.startswith("m"):
        try:
            return int(model_id[1:])
        except ValueError:
            pass
    try:
        return int(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")


@router.get("/history", response_model=ModelHistoryResponse)
async def get_model_history(limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_db)):
    repo = TrainingRepository(session)
    runs = repo.get_history(limit=limit)
    items = [_run_to_summary(run) for run in runs]
    return ModelHistoryResponse(items=items, total=len(items))


@router.get("/weeks", response_model=WeeksResponse)
async def list_weeks(session: Session = Depends(get_db)):
    from sqlalchemy import func
    from src.repositories.models import StockDaily

    db_range = session.query(func.min(StockDaily.date), func.max(StockDaily.date)).first()
    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=404, detail="No stock data found")
    data_start, data_end = db_range[0], db_range[1]
    factor_repo = FactorRepository(session)
    enabled_factors = factor_repo.get_all(enabled=True)
    current_hash = compute_factor_pool_hash([f.id for f in enabled_factors])
    all_weeks = get_trainable_weeks(data_start=data_start, data_end=data_end, session=session, include_insufficient=True)
    training_repo = TrainingRepository(session)
    all_runs = training_repo.get_all()
    runs_by_week = {}
    for run in all_runs:
        if run.week_id and run.status == "completed":
            runs_by_week[run.week_id] = run
    slots = []
    for week in all_weeks:
        run = runs_by_week.get(week.week_id)
        if not week.is_trainable:
            status = "insufficient_data"
            model = None
        elif run:
            is_outdated = run.factor_pool_hash != current_hash if run.factor_pool_hash else True
            model = WeekModel(id=f"m{run.id:03d}", name=run.name or "", model_ic=float(run.model_ic) if run.model_ic else 0.0, factor_count=run.factor_count or 0, factor_pool_hash=run.factor_pool_hash, is_outdated=is_outdated)
            status = "trained"
        else:
            model = None
            status = "trainable"
        slots.append(WeekSlot(week_id=week.week_id, valid_end=week.valid_end, valid_start=week.valid_start, train_end=week.train_end, train_start=week.train_start, status=status, model=model))
    return WeeksResponse(slots=slots, current_factor_pool_hash=current_hash, data_range=DataRange(start=data_start, end=data_end))


@router.get("/quality", response_model=QualityResponse)
async def get_quality_metrics(limit: int = Query(10, ge=1, le=50), session: Session = Depends(get_db)):
    from src.services.stability import QualityMonitor
    from src.shared.constants import QUALITY_IC_STD_MAX, QUALITY_ICIR_MIN, QUALITY_JACCARD_MIN
    monitor = QualityMonitor(session)
    metrics = monitor.get_latest_metrics(limit=limit)
    items = [QualityMetricsItem(**m) for m in metrics]
    return QualityResponse(items=items, thresholds={"jaccard_min": QUALITY_JACCARD_MIN, "ic_std_max": QUALITY_IC_STD_MAX, "icir_min": QUALITY_ICIR_MIN})


@router.post("/train", response_model=TrainResponse)
async def trigger_training(data: TrainRequest, session: Session = Depends(get_db)):
    from datetime import timedelta
    from pathlib import Path
    from sqlalchemy import func
    from src.repositories.database import get_session
    from src.repositories.models import StockDaily
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    week_id = data.week_id
    factor_repo = FactorRepository(session)
    db_range = session.query(func.min(StockDaily.date), func.max(StockDaily.date)).first()
    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=400, detail="No stock data found in database")
    db_start, db_end = db_range[0], db_range[1]
    trainable_weeks = get_trainable_weeks(data_start=db_start, data_end=db_end, session=session)
    week_slot = next((w for w in trainable_weeks if w.week_id == week_id), None)
    if not week_slot:
        raise HTTPException(status_code=400, detail=f"Week {week_id} is not trainable")
    train_start = week_slot.train_start
    train_end = week_slot.train_end
    valid_start = week_slot.valid_start
    valid_end = week_slot.valid_end
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")
    factor_pool_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

    async def training_task(progress_callback, **kwargs):
        task_session = get_session()
        loop = asyncio.get_event_loop()
        try:
            await progress_callback(0, "Exporting qlib data...")
            export_start = train_start - timedelta(days=LOOKBACK_DAYS)
            if export_start < db_start:
                export_start = db_start
            def do_export():
                exporter = QlibExporter(task_session)
                export_config = ExportConfig(start_date=export_start, end_date=valid_end, output_dir=Path("data/qlib"))
                return exporter.export(export_config)
            export_result = await asyncio.to_thread(do_export)
            await progress_callback(5, f"Exported {export_result.stocks_exported} symbols")
            trainer = ModelTrainer(qlib_data_dir="data/qlib")
            def sync_progress(progress: int, message: str):
                adjusted_progress = 5 + int(progress * 0.95)
                loop.call_soon_threadsafe(lambda p=adjusted_progress, m=message: asyncio.create_task(progress_callback(p, m)))
            result = await asyncio.to_thread(trainer.train, session=task_session, week_id=week_id, train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end, factor_pool_hash=factor_pool_hash, on_progress=sync_progress)
            return {"run_id": result.run_id, "model_name": result.model_name, "model_ic": result.model_ic, "icir": result.icir, "selected_factor_count": len(result.selected_factor_ids)}
        finally:
            task_session.close()

    job_id = await job_manager.create_job(job_type="train", task_fn=training_task, message=f"Training model: {week_id}")
    return TrainResponse(job_id=job_id, status="queued", message=f"Training job queued ({week_id})")


@router.post("/train-batch", response_model=TrainResponse)
async def trigger_batch_training(data: TrainBatchRequest, session: Session = Depends(get_db)):
    from datetime import timedelta
    from pathlib import Path
    from sqlalchemy import func
    from src.repositories.database import get_session
    from src.repositories.models import StockDaily
    from src.services.job_manager import job_manager
    from src.services.model_trainer import ModelTrainer
    from src.services.qlib_exporter import ExportConfig, QlibExporter

    year = data.year
    factor_repo = FactorRepository(session)
    training_repo = TrainingRepository(session)
    db_range = session.query(func.min(StockDaily.date), func.max(StockDaily.date)).first()
    if not db_range or not db_range[0] or not db_range[1]:
        raise HTTPException(status_code=400, detail="No stock data found in database")
    db_start, db_end = db_range[0], db_range[1]
    trainable_weeks = get_trainable_weeks(data_start=db_start, data_end=db_end, session=session)
    year_weeks = [w for w in trainable_weeks if w.week_id.startswith(year)]
    all_runs = training_repo.get_all()
    trained_week_ids = {r.week_id for r in all_runs if r.week_id and r.status == "completed"}
    untrained_weeks = [w for w in year_weeks if w.week_id not in trained_week_ids]
    if not untrained_weeks:
        raise HTTPException(status_code=400, detail=f"No remaining trainable weeks found for {year}")
    untrained_weeks.sort(key=lambda w: w.week_id)
    week_ids = [w.week_id for w in untrained_weeks]
    enabled_factors = factor_repo.get_all(enabled=True)
    if not enabled_factors:
        raise HTTPException(status_code=400, detail="No enabled factors found")
    factor_pool_hash = compute_factor_pool_hash([f.id for f in enabled_factors])

    async def batch_training_task(progress_callback, **kwargs):
        task_session = get_session()
        loop = asyncio.get_event_loop()
        try:
            total_weeks = len(week_ids)
            results = []
            for idx, week_id in enumerate(week_ids):
                base_progress = (idx / total_weeks) * 100
                week_progress_range = 100 / total_weeks
                await progress_callback(base_progress, f"[{idx + 1}/{total_weeks}] Training {week_id}...")
                week_slot = next((w for w in untrained_weeks if w.week_id == week_id), None)
                if not week_slot:
                    continue
                train_start = week_slot.train_start
                train_end = week_slot.train_end
                valid_start = week_slot.valid_start
                valid_end = week_slot.valid_end
                export_start = train_start - timedelta(days=LOOKBACK_DAYS)
                if export_start < db_start:
                    export_start = db_start
                def do_export():
                    exporter = QlibExporter(task_session)
                    export_config = ExportConfig(start_date=export_start, end_date=valid_end, output_dir=Path("data/qlib"))
                    return exporter.export(export_config)
                await asyncio.to_thread(do_export)
                trainer = ModelTrainer(qlib_data_dir="data/qlib")
                def sync_progress(progress: float, message: str):
                    adjusted = base_progress + (progress / 100) * week_progress_range
                    loop.call_soon_threadsafe(lambda p=adjusted, m=f"[{idx + 1}/{total_weeks}] {message}": asyncio.create_task(progress_callback(p, m)))
                result = await asyncio.to_thread(trainer.train, session=task_session, week_id=week_id, train_start=train_start, train_end=train_end, valid_start=valid_start, valid_end=valid_end, factor_pool_hash=factor_pool_hash, on_progress=sync_progress)
                results.append({"week_id": week_id, "model_name": result.model_name, "model_ic": result.model_ic})
            return {"total_trained": len(results), "results": results}
        finally:
            task_session.close()

    job_id = await job_manager.create_job(job_type="train_batch", task_fn=batch_training_task, message=f"Batch training {year}: {len(week_ids)} weeks")
    return TrainResponse(job_id=job_id, status="queued", message=f"Batch training queued ({year}: {len(week_ids)} weeks)")


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, session: Session = Depends(get_db)):
    run_id = _parse_model_id(model_id)
    training_repo = TrainingRepository(session)
    factor_repo = FactorRepository(session)
    run = training_repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")
    all_results = training_repo.get_all_factor_results(run_id)
    result_map = {r.factor_id: r for r in all_results}
    candidate_ids = _parse_factor_ids(run.candidate_factor_ids)
    selected_ids = _parse_factor_ids(run.selected_factor_ids)
    candidate_factors = []
    selected_factors = []
    factor_names = []
    if candidate_ids:
        for fid in candidate_ids:
            factor = factor_repo.get_by_id(fid)
            if factor:
                result = result_map.get(fid)
                summary = FactorSummary(id=f"f{factor.id:03d}", name=factor.name, display_name=factor.display_name, category=factor.category, ic_value=float(result.ic_value) if result else None)
                candidate_factors.append(summary)
        for fid in selected_ids:
            factor = factor_repo.get_by_id(fid)
            if factor:
                result = result_map.get(fid)
                summary = FactorSummary(id=f"f{factor.id:03d}", name=factor.name, display_name=factor.display_name, category=factor.category, ic_value=float(result.ic_value) if result else None)
                selected_factors.append(summary)
                factor_names.append(factor.name)
    else:
        for result in all_results:
            factor = result.factor
            summary = FactorSummary(id=f"f{factor.id:03d}", name=factor.name, display_name=factor.display_name, category=factor.category, ic_value=float(result.ic_value))
            candidate_factors.append(summary)
            if result.selected:
                selected_factors.append(summary)
                factor_names.append(factor.name)
    return _run_to_response(run, factor_names, candidate_factors, selected_factors)


@router.delete("/all")
async def delete_all_models(session: Session = Depends(get_db)):
    repo = TrainingRepository(session)
    runs = repo.get_all()
    deleted_count = 0
    for run in runs:
        if run.name:
            model_dir = MODELS_DIR / run.name
            if model_dir.exists():
                shutil.rmtree(model_dir)
        repo.delete(run.id)
        deleted_count += 1
    await broadcast_data_updated("models", "delete", "all")
    return {"deleted_count": deleted_count}


@router.delete("/{model_id}")
async def delete_model(model_id: str, session: Session = Depends(get_db)):
    from src.services.job_manager import job_manager
    run_id = _parse_model_id(model_id)
    repo = TrainingRepository(session)
    run = repo.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model not found")
    if run.status in ("running", "queued"):
        from src.repositories.job import JobRepository
        job_repo = JobRepository(session)
        active_jobs = job_repo.get_active()
        for job in active_jobs:
            if job.job_type == "train":
                await job_manager.cancel_job(job.id)
                session.delete(job)
        session.commit()
    if run.name:
        model_dir = MODELS_DIR / run.name
        if model_dir.exists():
            shutil.rmtree(model_dir)
    repo.delete(run_id)
    await broadcast_data_updated("models", "delete", model_id)
    return {"status": "deleted", "id": model_id}
