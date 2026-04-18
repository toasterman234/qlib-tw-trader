"""FastAPI application entrypoint."""

import logging
import time

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.interfaces.exceptions import register_exception_handlers
from src.interfaces.routers import (
    backtest,
    dashboard,
    datasets,
    factor,
    model,
    portfolio,
    qlib,
    sync_us,
    system,
    universe,
    websocket,
)
from src.repositories.database import init_db
from src.shared.market import get_market

perf_logger = logging.getLogger("perf")


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        path = request.url.path
        if path.startswith("/api/"):
            if elapsed_ms > 100:
                perf_logger.warning(f"SLOW {request.method} {path} {elapsed_ms:.0f}ms")
            else:
                perf_logger.info(f"{request.method} {path} {elapsed_ms:.0f}ms")
        return response


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    market = get_market()
    app = FastAPI(
        title=market.app_title,
        description=market.app_description,
        version="1.0.0-us",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(TimingMiddleware)

    app.include_router(sync_us.router, prefix="/api/v1/sync", tags=["sync"])
    app.include_router(universe.router, prefix="/api/v1/universe", tags=["universe"])
    app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    app.include_router(factor.router, prefix="/api/v1/factors", tags=["factors"])
    app.include_router(model.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])
    app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["backtest"])
    app.include_router(qlib.router, prefix="/api/v1", tags=["qlib"])

    register_exception_handlers(app)
    init_db()

    return app


app = create_app()
