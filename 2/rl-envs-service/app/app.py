from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import datasets_router, envs_router, health_router
from app.utils.openapi import save_openapi_spec


def get_application() -> FastAPI:
    app = FastAPI(
        title="RL Envs Service API",
        version="1.0.0",
        description="Service for generating/verifying RL environment tasks.",
        openapi_tags=[
            {"name": "Health", "description": "Health-check приложения."},
            {"name": "Envs", "description": "Генерация и верификация задач."},
            {"name": "Datasets", "description": "Генерация фиксированных тестовых датасетов."},
        ],
    )

    app.include_router(health_router.router, tags=["Health"])
    app.include_router(envs_router.router, tags=["Envs"])
    app.include_router(datasets_router.router, tags=["Datasets"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = get_application()


@app.on_event("startup")
async def _save_openapi() -> None:
    save_openapi_spec(app)
