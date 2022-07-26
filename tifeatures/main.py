"""tifeatures app."""

import logging
from typing import Any, List

import jinja2

from tifeatures import __version__ as tifeatures_version
from tifeatures.db import close_db_connection, connect_to_db, register_table_catalog
from tifeatures.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from tifeatures.factory import Endpoints
from tifeatures.layer import FunctionRegistry
from tifeatures.logger import logger
from tifeatures.middleware import CacheControlMiddleware
from tifeatures.resources.enums import OptionalHeader
from tifeatures.settings import APISettings

from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware

settings = APISettings()

app = FastAPI(
    title=settings.name,
    version=tifeatures_version,
    openapi_url="/api",
    docs_url="/api.html",
    debug=settings.debug,
)

# custom template directory
templates_location: List[Any] = (
    [jinja2.FileSystemLoader(settings.template_directory)]
    if settings.template_directory
    else []
)
# default template directory
templates_location.append(jinja2.PackageLoader(__package__, "templates"))

templates = Jinja2Templates(
    directory="",  # we need to set a dummy directory variable, see https://github.com/encode/starlette/issues/1214
    loader=jinja2.ChoiceLoader(templates_location),
)

# Register endpoints.
if settings.debug:
    optional_headers = [OptionalHeader.server_timing]
    logger.setLevel(logging.DEBUG)
else:
    optional_headers = []

endpoints = Endpoints(
    title=settings.name, templates=templates, optional_headers=optional_headers
)
app.include_router(endpoints.router)

# We add the function registry to the application state
app.state.tifeatures_function_catalog = FunctionRegistry()

# Set all CORS enabled origins
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

app.add_middleware(CacheControlMiddleware, cachecontrol=settings.cachecontrol)
app.add_middleware(CompressionMiddleware)
add_exception_handlers(app, DEFAULT_STATUS_CODES)


@app.on_event("startup")
async def startup_event() -> None:
    """Connect to database on startup."""
    await connect_to_db(app)
    await register_table_catalog(app)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close database connection."""
    await close_db_connection(app)


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping():
    """Health check."""
    return {"ping": "pong!"}
