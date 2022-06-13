"""tifeatures.db: database events."""

from typing import Dict, List, Optional

from buildpg import asyncpg
from pydantic import BaseModel


class Column(BaseModel):
    """Model for database Column."""

    name: str
    type: str
    description: Optional[str]


class GeometryColumn(BaseModel):
    """Model for PostGIS geometry/geography column."""

    name: str
    bounds: List[float]
    srid: int
    geometry_type: str


class Table(BaseModel):
    """Model for DB Table."""

    id: str
    table: str
    dbschema: str
    description: Optional[str]
    id_col: Optional[str]
    geometry_columns: Optional[List[GeometryColumn]]
    properties: List[Column]

    @property
    def datetime_columns(self) -> Optional[List[Column]]:
        """Return the name of all timestamptz columns."""
        return [p for p in self.properties if p.type.startswith("timestamp")]

    def datetime_col(self, dtcol: Optional[str] = None):
        """Return the Column for either the passed in tstz column or the first tstz column."""
        if self.datetime_columns is not None:
            for col in self.datetime_columns:
                if dtcol is None or col.name == dtcol:
                    return col

    def geom_col(self, gcol: Optional[str] = None) -> Optional[GeometryColumn]:
        """Return the name of the first geometry column."""
        if self.geometry_columns is not None and len(self.geometry_columns) > 0:
            for c in self.geometry_columns:
                if gcol is None or c.name == gcol:
                    return c

        return None

    @property
    def id_column_info(self):
        """Return Column for a unique identifier."""
        for c in self.properties:
            if c.name == self.id_col:
                return c

    def columns(self, properties: Optional[List[str]]) -> List[str]:
        """Return table columns optionally filtered to only include columns from properties."""
        cols = [c.name for c in self.properties]
        if properties is not None:
            if self.id_col is not None and self.id_col not in properties:
                properties.append(self.id_col)

            geom_col = self.geom_col()
            if geom_col:
                properties.append(geom_col.name)

            cols = [c for c in cols if c in properties]

        if len(cols) < 1:
            raise TypeError("No columns selected")
        return cols


class Database(BaseModel):
    """Keyed tables for a Database."""

    tables: Dict[str, Table]


async def table_index(
    db_pool: asyncpg.BuildPgPool,
    schemas: Optional[List[str]] = ["public"],
    tables: Optional[List[str]] = None,
    spatial: Optional[bool] = True,
) -> Database:
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
                        coalesce(jsonb_agg(
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
                        ),'[]'::jsonb)
                    FROM
                        (
                        SELECT f_table_schema, f_table_name, f_geometry_column, srid, type
                        FROM geometry_columns
                        UNION ALL
                        SELECT f_table_schema, f_table_name, f_geography_column, 4326, type
                        FROM geography_columns
                        ) as geo
                    WHERE
                        f_table_schema = schemaname
                        AND f_table_name = tablename
                ) as geometry_columns
            FROM
                pg_tables
            WHERE
                schemaname NOT IN ('pg_catalog', 'information_schema')
                AND tablename NOT IN ('spatial_ref_sys','geometry_columns')
                AND (:schemas::text[] IS NULL OR schemaname = ANY (:schemas))
                AND (:tables::text[] IS NULL OR tablename = ANY (:tables))

        )
        SELECT
                id,
                schemaname as dbschema,
                tablename as tablename,
                geometry_columns,
                pk as id_col,
                columns as properties,
                description
        FROM t
        WHERE :spatial = FALSE OR jsonb_array_length(geometry_columns)>=1
        ;
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch_b(
            query, schemas=schemas, tables=tables, spatial=spatial
        )
        d = {}
        for row in rows:
            d[row["id"]] = Table(
                id=row[0],
                dbschema=row[1],
                table=row[2],
                geometry_columns=row[3],
                id_col=row[4],
                properties=row[5],
                description=row[6],
            )
    return Database(tables=d)
