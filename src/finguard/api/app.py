"""HTTP layer: FastAPI (interactive docs at ``/docs``), or plain Starlette when FastAPI is absent.

Every route in ``service.ROUTES`` becomes one endpoint that hands the request to
``Service.handle`` in a worker thread (an investigation takes seconds; the event loop stays free)
and returns its ``Reply``. The OpenAPI description of each endpoint comes from the route's JSON
schemas. ``/`` serves the dashboard, one self-contained HTML file.

    uvicorn finguard.api.app:app            # or: python -m finguard.api serve
"""
# no ``from __future__ import annotations`` here: FastAPI reads the endpoint's ``Request``
# annotation at runtime.
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from finguard import __version__
from finguard.api.service import ROUTES, Service

STATIC = Path(__file__).with_name("static")
DESCRIPTION = ("FinGuard AI decision support for card-not-present fraud review. Every response carries the model, "
               "knowledge-base index and prompt versions (X-FinGuard-* headers and a `versions` field). "
               "Flagged for review, not a finding of fraud: the analyst decides.")


def framework(name: str = "auto") -> str:
    if name in ("auto", "fastapi"):
        try:
            import fastapi  # noqa: F401
            return "fastapi"
        except ImportError:
            if name == "fastapi":
                raise
    return "starlette"


def _endpoint(service: Service, route):
    async def endpoint(request: Request):
        body = await request.body() if route.body is not None else None
        reply = await run_in_threadpool(service.handle, route, dict(request.path_params),
                                        list(request.query_params.multi_items()), body)
        return Response(content=reply.content(), status_code=reply.status, media_type=reply.media_type,
                        headers=reply.headers)
    endpoint.__name__ = route.name
    return endpoint


def create_app(service: Service | None = None, use: str = "auto", start: bool = True):
    """The ASGI app. ``start=False`` when the service was started (and loaded) by the caller."""
    service = service or Service.from_env()

    @asynccontextmanager
    async def lifespan(app):
        if start:
            service.start(background=True)                       # the server answers while the data loads
        yield
        service.stop()

    async def dashboard(request: Request):
        return HTMLResponse((STATIC / "dashboard.html").read_text(encoding="utf-8"), headers=service.version_headers())

    async def fallthrough(request: Request):
        """Unknown path or wrong method: a JSON 404/405 that carries the versions like every reply."""
        reply = service.dispatch(request.method, request.url.path)
        return Response(content=reply.content(), status_code=reply.status, media_type=reply.media_type, headers=reply.headers)

    every = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    if framework(use) == "fastapi":
        from fastapi import FastAPI
        app = FastAPI(title="FinGuard AI", version=__version__, description=DESCRIPTION, lifespan=lifespan)
        for r in ROUTES:
            app.add_api_route(r.path, _endpoint(service, r), methods=[r.method], name=r.name, operation_id=r.name,
                              summary=r.summary, openapi_extra=r.openapi() or None, response_class=Response)
        app.add_api_route("/", dashboard, methods=["GET"], include_in_schema=False)
        app.add_api_route("/api/{path:path}", fallthrough, methods=every, include_in_schema=False)
    else:
        from starlette.applications import Starlette
        from starlette.routing import Route
        routes = [Route(r.path, _endpoint(service, r), methods=[r.method], name=r.name) for r in ROUTES]
        app = Starlette(routes=routes + [Route("/", dashboard, methods=["GET"]),
                                         Route("/api/{path:path}", fallthrough, methods=every)], lifespan=lifespan)
    app.state.service = service
    return app


def __getattr__(name):
    """``finguard.api.app:app`` for uvicorn, built on first access from the environment."""
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
