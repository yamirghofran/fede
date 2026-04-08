"""Local entrypoint for running the FastAPI backend."""

from __future__ import annotations

import uvicorn

from .settings import BackendSettings


def main() -> None:
    settings = BackendSettings()
    uvicorn.run(
        "vector_db.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()
