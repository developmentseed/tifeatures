"""tifeatures.db: database events."""

import json
from typing import Callable, List, Optional, Type

from buildpg import asyncpg
from tifeatures.filter.filters import parse_bbox

from tifeatures.settings import PostgresSettings
from tifeatures.filter.evaluate import to_filter
from tifeatures.layer import CollectionLayer, Items
from geojson_pydantic import Feature, FeatureCollection

from fastapi import FastAPI

from typing import List, Optional

from pydantic import BaseModel

from buildpg import V, S, render, logic, funcs
from buildpg.clauses import Select, From, Where, Limit, OrderBy, Offset

class Column(BaseModel):
    name: str
    type: str
    description: Optional[str]


class GeometryColumn(BaseModel):
    name: str
    bounds: List[float]
    srid: int
    geometry_type: str


class Table(CollectionLayer):
    id: str
    table: str
    dbschema: str
    description: Optional[str]
    id_column: Optional[str]
    geometry_columns: Optional[List[GeometryColumn]]
    properties: List[Column]

    @property
    def datetime_col(self)-> Optional[str]:
        for c in self.properties:
            if c.type == 'timestamptz':
                return c.name

    @property
    def geom_col(self) -> Optional[GeometryColumn]:
        """Return the name of the first geometry column."""
        if self.geometry_columns is not None and len(self.geometry_columns) > 0:
            return self.geometry_columns[0]

    @property
    def id_column_info(self):
        for c in self.properties:
            if c.name == self.id_column:
                return c

    def columns(self, properties:Optional[List[str]]) -> List[str]:
        """Return table columns optionally filtered to only include columns from properties."""
        cols = [c.name for c in self.properties]
        if properties is not None:
            if self.id_column is not None and self.id_column not in properties:
                properties.append(self.id_column)
            if self.geom_col:
                properties.append(self.geom_col.name)
            cols = [c for c in cols if c in properties]
        if len(cols) < 1:
            raise TypeError("No columns selected")
        return cols

    def select(self, properties: Optional[List[str]]):
        return Select(self.columns(properties))

    def select_count(self):
        return Select(funcs.count('*'))

    def _from(self):
        return From(self.id)

    def where(
        self,
        ids: Optional[List[str]] = None,
        datetime: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        filter: Optional[Callable] = None,
    ):
        wheres = [S(True)]

        if ids is not None:
            if len(ids) == 1:
                wheres.append(
                    V(self.id_column) == funcs.cast(
                        funcs.cast(ids[0], 'text'),
                        self.id_column_info.type
                        )
                )
            else:
                w = [
                    V(self.id_column) == S(funcs.cast(i, self.id_column_info.type))
                    for i in ids
                ]
                wheres.append(
                    funcs.OR(*w)
                )

        if bbox is not None and self.geom_col is not None:
            wheres.append(
                logic.Func(
                    'ST_Intersects',
                    S(parse_bbox(bbox)),
                    V(self.geom_col.name)
                )
            )

        if datetime is not None and self.datetime_col is not None:
            dt = datetime.split('/')
            if len(dt) > 2:
                raise TypeError('Datetime not valid')
            if len(dt) == 1:
                wheres.append(
                    V(self.datetime_col) == S(dt[0])
                )
            else:
                wheres.append(
                    funcs.AND(
                        V(self.datetime_col) >= S(dt[0]),
                        V(self.datetime_col) < S(dt[1])
                    )
                )
        if filter is not None:
            wheres.append(
                to_filter(filter, [p.name for p in self.properties])
            )

        return Where(funcs.AND(*wheres))

    def features_query(
        self,
        ids: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        datetime: Optional[str] = None,
        filter: Optional[Callable] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ):
        return (
            self.select(properties) +
            self._from() +
            self.where(ids, datetime, bbox, filter) +
            Limit(limit or 10) +
            Offset(offset or 0)
        )

    def features_count_query(
        self,
        ids: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        datetime: Optional[str] = None,
        filter: Optional[Callable] = None,
    ):
        return (
            self.select_count() +
            self._from() +
            self.where(ids, datetime, bbox, filter)
        )

    async def geojson(
        self,
        pool: asyncpg.BuildPgPool,
        ids: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        datetime: Optional[str] = None,
        filter: Optional[Callable] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ):
        if not (self.geom_col is not None and self.geometry_columns is not None and len(self.geometry_columns)>0):
            raise TypeError('Must have geometry column for geojson output.')

        query = """
            WITH
                features AS (
                    :features_q
                ),
                total_count AS (
                    :count_q
                )
            SELECT json_build_object(
                'type', 'FeatureCollection',
                'features', json_agg(
                    json_build_object(
                        'type', 'Feature',
                        'id', :id_column,
                        'geometry', ST_AsGeoJSON(
                            CASE
                                WHEN :srid = 4326 THEN :geometry_column
                                ELSE ST_Transform(:geometry_column, 4326)
                            END
                            )::json,
                        'properties', to_jsonb( features.* ) - :geom_columns
                    )
                ),
                'total_count', total_count.count
            )
            FROM features, total_count
            GROUP BY total_count.count;
        """
        q, p = render(
            query,
            features_q=self.features_query(ids, bbox,datetime, filter, properties, limit, offset),
            count_q=self.features_count_query(ids, bbox,datetime, filter),
            id_column=V(self.id_column),
            srid=self.geom_col.srid,
            geometry_column=V(self.geom_col.name),
            geom_columns=self.geom_col.name
        )
        print(q,p)
        async with pool.acquire() as conn:
            return await conn.fetchval(q, *p)

    async def features(
        self,
        pool: asyncpg.BuildPgPool,
        ids: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        datetime: Optional[str] = None,
        filter: Optional[Callable] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Items:
        """Return a FeatureCollection."""
        return await self.geojson(
            pool=pool,
            ids=ids,
            bbox=bbox,
            datetime=datetime,
            filter=filter,
            properties=properties,
            limit=limit,
            offset=offset
        )


    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
        properties: Optional[List[str]] = None,
    ) -> Feature:
        """Return a Feature."""
        geojson = await self.geojson(
            pool=pool,
            ids=[item_id],
            properties=properties
        )
        return geojson['features'][0]




class Database(BaseModel):
    tables: List[Table]

async def table_index(db_pool: asyncpg.BuildPgPool) -> Database:
    """Fetch Table index."""

    query = """
        WITH t AS (
            SELECT
                schemaname,
                tablename,
                format('%I.%I', schemaname, tablename) as id,
                format('%I.%I', schemaname, tablename)::regclass as t_oid,
                obj_description(format('%I.%I', schemaname, tablename)::regclass, 'pg_class') as description,
                (
                    SELECT
                        attname
                    FROM
                        pg_index i
                        JOIN pg_attribute a ON
                            a.attrelid = i.indrelid
                            AND a.attnum = ANY(i.indkey)
                    WHERE
                        i.indrelid = format('%I.%I', schemaname, tablename)::regclass
                        AND
                        (i.indisunique OR i.indisprimary)
                    ORDER BY i.indisprimary
                    LIMIT 1
                ) as pk,
                (
                    SELECT
                        jsonb_agg(
                            jsonb_build_object(
                                'name', attname,
                                'type', format_type(atttypid, null),
                                'description', col_description(attrelid, attnum)
                            )
                        )
                    FROM
                        pg_attribute
                    WHERE
                        attnum>0
                        AND attrelid=format('%I.%I', schemaname, tablename)::regclass
                ) as columns,
                (
                    SELECT
                        jsonb_agg(
                            jsonb_build_object(
                                'name', f_geometry_column,
                                'srid', srid,
                                'geometry_type', type,
                                'bounds',
                                    CASE WHEN srid IS NOT NULL AND srid != 0 THEN
                                        (
                                            SELECT
                                                ARRAY[
                                                    ST_XMin(extent.geom),
                                                    ST_YMin(extent.geom),
                                                    ST_XMax(extent.geom),
                                                    ST_YMax(extent.geom)
                                                ]
                                            FROM (
                                                SELECT
                                                    coalesce(
                                                        ST_Transform(
                                                            ST_SetSRID(
                                                                ST_EstimatedExtent(f_table_schema, f_table_name, f_geometry_column),
                                                                srid
                                                            ),
                                                            4326
                                                        ),
                                                        ST_MakeEnvelope(-180, -90, 180, 90, 4326)
                                                    ) as geom
                                                ) AS extent
                                        )
                                    ELSE ARRAY[-180,-90,180,90]
                                    END
                            )
                        )
                    FROM geometry_columns
                    WHERE
                        f_table_schema = schemaname
                        AND f_table_name = tablename
                ) as geometry_columns
            FROM
                pg_tables
            WHERE
                schemaname NOT IN ('pg_catalog', 'information_schema')
                AND tablename NOT IN ('spatial_ref_sys','geometry_columns')
        )
        SELECT
            jsonb_agg(
                jsonb_build_object(
                    'id', id,
                    'dbschema', schemaname,
                    'table', tablename,
                    'geometry_columns', geometry_columns,
                    'id_column', pk,
                    'properties', columns,
                    'description', description
                )
            ) as tables
        FROM t;
    """

    async with db_pool.acquire() as conn:
        q = await conn.prepare(query)
        content = await q.fetchval()

    return Database(tables=content)


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
