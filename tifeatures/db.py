"""tifeatures.db: database events."""

import json
from typing import Optional

from buildpg import asyncpg

from tifeatures.dbmodel import table_index
from tifeatures.settings import PostgresSettings

from fastapi import FastAPI


async def con_init(conn):
    """Use json for json returns."""
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def connect_to_db(
    app: FastAPI, settings: Optional[PostgresSettings] = None
) -> None:
    """Connect."""
    if not settings:
        settings = PostgresSettings()

    app.state.pool = await asyncpg.create_pool_b(
        settings.database_url,
        min_size=settings.db_min_conn_size,
        max_size=settings.db_max_conn_size,
        max_queries=settings.db_max_queries,
        max_inactive_connection_lifetime=settings.db_max_inactive_conn_lifetime,
        init=con_init,
    )


async def register_table_catalog(app: FastAPI) -> None:
    """Register Table catalog."""
    app.state.table_catalog = await table_index(app.state.pool)


async def close_db_connection(app: FastAPI) -> None:
    """Close connection."""
    await app.state.pool.close()
