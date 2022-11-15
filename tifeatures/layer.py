"""tifeatures.layers."""

import abc
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from buildpg import RawDangerous, asyncpg, clauses
from buildpg import funcs as pg_funcs
from buildpg import logic, render
from ciso8601 import parse_rfc3339
from geojson_pydantic import Feature, FeatureCollection  # type:ignore
from pydantic import BaseModel, root_validator
from pygeofilter.ast import AstType

from tifeatures.dbmodel import GeometryColumn
from tifeatures.dbmodel import Table as DBTable
from tifeatures.errors import (
    InvalidDatetime,
    InvalidDatetimeColumnName,
    InvalidGeometryColumnName,
    InvalidPropertyName,
    MissingDatetimeColumn,
)
from tifeatures.filter.evaluate import to_filter
from tifeatures.filter.filters import bbox_to_wkt
from tifeatures.resources.enums import MediaType

# Links to geojson schema
geojson_schema = {
    "GEOMETRY": "https://geojson.org/schema/Geometry.json",
    "POINT": "https://geojson.org/schema/Point.json",
    "MULTIPOINT": "https://geojson.org/schema/MultiPoint.json",
    "LINESTRING": "https://geojson.org/schema/LineString.json",
    "MULTILINESTRING": "https://geojson.org/schema/MultiLineString.json",
    "POLYGON": "https://geojson.org/schema/Polygon.json",
    "MULTIPOLYGON": "https://geojson.org/schema/MultiPolygon.json",
    "GEOMETRYCOLLECTION": "https://geojson.org/schema/GeometryCollection.json",
}


class CollectionLayer(BaseModel, metaclass=abc.ABCMeta):
    """Layer's Abstract BaseClass.

    Attributes:
        id (str): Layer's name.
        bounds (list): Layer's bounds (left, bottom, right, top).
        crs (str): Coordinate reference system of the Collection.
        title (str): Layer's title
        description (str): Layer's description

    """

    id: str
    bounds: Optional[List[float]]
    crs: Optional[str]
    title: Optional[str]
    description: Optional[str]

    @abc.abstractmethod
    async def features(
        self,
        pool: asyncpg.BuildPgPool,
        ids_filter: Optional[List[str]] = None,
        bbox_filter: Optional[List[float]] = None,
        datetime_filter: Optional[List[str]] = None,
        properties_filter: Optional[List[Tuple[str, Any]]] = None,
        cql_filter: Optional[AstType] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[FeatureCollection, int]:
        """Return a FeatureCollection and the number of matched items."""
        ...

    @abc.abstractmethod
    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
        properties: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Feature:
        """Return a Feature."""
        ...

    @property
    def queryables(self) -> Dict:
        """Return the queryables."""
        ...


class Table(CollectionLayer, DBTable):
    """Table Reader.

    Attributes:
        id (str): Layer's name.
        bounds (list): Layer's bounds (left, bottom, right, top).
        crs (str): Coordinate reference system of the Table.
        type (str): Layer's type.
        schema (str): Table's database schema (e.g public).
        geometry_type (str): Table's geometry type (e.g polygon).
        srid (int): Table's SRID
        geometry_column (str): Name of the geomtry column in the table.
        properties (Dict): Properties available in the table.

    """

    type: str = "Table"

    @root_validator
    def bounds_default(cls, values):
        """Get default bounds from the first geometry columns."""
        geoms = values.get("geometry_columns")
        if geoms:
            # Get the Extent of all the bounds
            minx, miny, maxx, maxy = zip(*[geom.bounds for geom in geoms])
            values["bounds"] = [min(minx), min(miny), max(maxx), max(maxy)]
            values["crs"] = f"http://www.opengis.net/def/crs/EPSG/0/{geoms[0].srid}"

        return values

    def _select(
        self,
        properties: Optional[List[str]],
        geometry_column: Optional[GeometryColumn],
        bbox_only: Optional[bool],
        simplify: Optional[float],
        media_type: MediaType = MediaType.geojson,
    ):
        columns = self.columns(properties)
        if columns:
            sel = clauses.Select(columns) + RawDangerous(",")
        else:
            sel = RawDangerous("SELECT ")

        if self.id_column:
            sel = sel + logic.V(self.id_column) + RawDangerous(" AS tifeatures_id, ")
        else:
            sel = sel + RawDangerous(" ROW_NUMBER () OVER () AS tifeatures_id, ")

        geom = self._geom(geometry_column, bbox_only, simplify)
        if media_type in [MediaType.geojson, MediaType.geojsonseq]:
            if geom:
                sel = (
                    sel
                    + pg_funcs.cast(logic.Func("st_asgeojson", geom), "json")
                    + RawDangerous(" AS tifeatures_geom ")
                )
            else:
                sel = sel + RawDangerous(" NULL::json AS tifeatures_geom ")
        else:
            if geom:
                sel = (
                    sel
                    + logic.Func("st_asgeojson", geom)
                    + RawDangerous(" AS tifeatures_geom ")
                )
            else:
                sel = sel + RawDangerous(" NULL::text AS tifeatures_geom ")
        return sel

    def _select_count(self):
        return clauses.Select(pg_funcs.count("*"))

    def _from(self):
        return clauses.From(self.id)

    def _geom(
        self,
        geometry_column: Optional[GeometryColumn],
        bbox_only: Optional[bool],
        simplify: Optional[float],
    ):
        if geometry_column is None:
            return pg_funcs.cast(None, "geometry")

        g = logic.V(geometry_column.name)
        g = pg_funcs.cast(g, "geometry")

        if geometry_column.srid == 4326:
            g = logic.Func("ST_Transform", g, pg_funcs.cast(4326, "int"))

        if bbox_only:
            g = logic.Func("ST_Envelope", g)
        elif simplify:
            g = logic.Func(
                "ST_SnapToGrid",
                logic.Func("ST_Simplify", g, simplify),
                simplify,
            )

        return g

    def _where(
        self,
        ids: Optional[List[str]] = None,
        datetime: Optional[List[str]] = None,
        bbox: Optional[List[float]] = None,
        properties: Optional[List[Tuple[str, Any]]] = None,
        cql: Optional[AstType] = None,
        geom: str = None,
        dt: str = None,
    ):
        """Construct WHERE query."""
        wheres = [logic.S(True)]

        # `ids` filter
        if ids is not None:
            if len(ids) == 1:
                wheres.append(
                    logic.V(self.id_column)
                    == pg_funcs.cast(
                        pg_funcs.cast(ids[0], "text"), self.id_column_info.type
                    )
                )
            else:
                w = [
                    logic.V(self.id_column)
                    == logic.S(
                        pg_funcs.cast(
                            pg_funcs.cast(i, "text"), self.id_column_info.type
                        )
                    )
                    for i in ids
                ]
                wheres.append(pg_funcs.OR(*w))

        # `properties filter
        if properties is not None:
            w = []
            for (prop, val) in properties:
                col = self.get_column(prop)
                if not col:
                    raise InvalidPropertyName(f"Invalid property name: {prop}")

                w.append(
                    logic.V(col.name)
                    == logic.S(pg_funcs.cast(pg_funcs.cast(val, "text"), col.type))
                )

            if w:
                wheres.append(pg_funcs.AND(*w))

        # `bbox` filter
        geometry_column = self.get_geometry_column(geom)
        if bbox is not None and geometry_column is not None:
            wheres.append(
                logic.Func(
                    "ST_Intersects",
                    logic.S(bbox_to_wkt(bbox)),
                    logic.V(geometry_column.name),
                )
            )

        # `datetime` filter
        if datetime:
            if not self.datetime_columns:
                raise MissingDatetimeColumn(
                    "Must have timestamp typed column to filter with datetime."
                )

            datetime_column = self.get_datetime_column(dt)
            if not datetime_column:
                raise InvalidDatetimeColumnName(f"Invalid Datetime Column: {dt}.")

            wheres.append(self._datetime_filter_to_sql(datetime, datetime_column.name))

        # `CQL` filter
        if cql is not None:
            wheres.append(to_filter(cql, [p.name for p in self.properties]))

        return clauses.Where(pg_funcs.AND(*wheres))

    def _datetime_filter_to_sql(self, interval: List[str], dt_name: str):
        if len(interval) == 1:
            return logic.V(dt_name) == logic.S(
                pg_funcs.cast(parse_rfc3339(interval[0]), "timestamptz")
            )

        else:
            start = (
                parse_rfc3339(interval[0]) if not interval[0] in ["..", ""] else None
            )
            end = parse_rfc3339(interval[1]) if not interval[1] in ["..", ""] else None

            if start is None and end is None:
                raise InvalidDatetime(
                    "Double open-ended datetime intervals are not allowed."
                )

            if start is not None and end is not None and start > end:
                raise InvalidDatetime("Start datetime cannot be before end datetime.")

            if not start:
                return logic.V(dt_name) <= logic.S(pg_funcs.cast(end, "timestamptz"))

            elif not end:
                return logic.V(dt_name) >= logic.S(pg_funcs.cast(start, "timestamptz"))

            else:
                return pg_funcs.AND(
                    logic.V(dt_name) >= logic.S(pg_funcs.cast(start, "timestamptz")),
                    logic.V(dt_name) < logic.S(pg_funcs.cast(end, "timestamptz")),
                )

    def _sortby(self, sortby: Optional[str]):
        sorts = []
        if sortby:
            for s in sortby.strip().split(","):
                parts = re.match(
                    "^(?P<direction>[+-]?)(?P<column>.*)$", s
                ).groupdict()  # type:ignore

                direction = parts["direction"]
                column = parts["column"].strip()
                if self.get_column(column):
                    if direction == "-":
                        sorts.append(logic.V(column).desc())
                    else:
                        sorts.append(logic.V(column))
                else:
                    raise InvalidPropertyName(f"Property {column} does not exist.")

        else:
            if self.id_column is not None:
                sorts.append(logic.V(self.id_column))
            else:
                sorts.append(logic.V(self.properties[0].name))

        return clauses.OrderBy(*sorts)

    def _features_query(
        self,
        *,
        ids_filter: Optional[List[str]] = None,
        bbox_filter: Optional[List[float]] = None,
        datetime_filter: Optional[List[str]] = None,
        properties_filter: Optional[List[Tuple[str, str]]] = None,
        cql_filter: Optional[AstType] = None,
        sortby: Optional[str] = None,
        properties: Optional[List[str]] = None,
        geom: str = None,
        dt: str = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ):
        """Build Features query."""
        return (
            self._select(properties)
            + self._from()
            + self._where(
                ids=ids_filter,
                datetime=datetime_filter,
                bbox=bbox_filter,
                properties=properties_filter,
                cql=cql_filter,
                geom=geom,
                dt=dt,
            )
            + self._sortby(sortby)
            + clauses.Limit(limit or 10)
            + clauses.Offset(offset or 0)
        )

    async def _features_count(self, pool: asyncpg.BuildPgPool, _from, _where):
        """Build features COUNT query."""
        c = self._select_count() + _from + _where
        q, p = render(":c", c=c)
        async with pool.acquire() as conn:
            count = await conn.fetchval(q, *p)
            return count

    async def _features(
        self,
        pool: asyncpg.BuildPgPool,
        _from,
        _where,
        sortby: Optional[str] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        geometry_column=None,
        bbox_only=None,
        simplify=None,
    ):
        """Build features COUNT query."""
        c = (
            self._select(
                properties=properties,
                geometry_column=geometry_column,
                bbox_only=bbox_only,
                simplify=simplify,
            )
            + _from
            + _where
            + self._sortby(sortby)
            + clauses.Limit(limit or 10)
            + clauses.Offset(offset or 0)
        )
        q, p = render(":c", c=c)
        print(q, p)
        async with pool.acquire() as conn:
            for r in await conn.fetch(q, *p):
                properties = dict(r)
                yield properties

    async def query(
        self,
        pool: asyncpg.BuildPgPool,
        *,
        ids_filter: Optional[List[str]] = None,
        bbox_filter: Optional[List[float]] = None,
        datetime_filter: Optional[List[str]] = None,
        properties_filter: Optional[List[Tuple[str, str]]] = None,
        cql_filter: Optional[AstType] = None,
        sortby: Optional[str] = None,
        properties: Optional[List[str]] = None,
        geom: str = None,
        dt: str = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        bbox_only: Optional[bool] = None,
        simplify: Optional[float] = None,
    ) -> Tuple[FeatureCollection, int]:
        """Build and run Pg query."""
        geom_col = self.get_geometry_column(geom)

        if geom and geom.lower() != "none" and not geom_col:
            raise InvalidGeometryColumnName(f"Invalid Geometry Column: {geom}.")

        if geom_col:
            geom_col_name = geom_col.name
        else:
            geom_col_name = None

        _from = self._from()
        _where = self._where()

        # Get count

        pgfeatures = self._features(
            pool=pool,
            _from=_from,
            _where=_where,
            geom_col_name=geom_col_name,
            id_column=self.id_column,
            sortby=sortby,
            properties=properties,
            limit=limit,
            offset=offset,
        )

        return pgfeatures

    async def features(
        self,
        pool: asyncpg.BuildPgPool,
        ids_filter: Optional[List[str]] = None,
        bbox_filter: Optional[List[float]] = None,
        datetime_filter: Optional[List[str]] = None,
        properties_filter: Optional[List[Tuple[str, str]]] = None,
        cql_filter: Optional[AstType] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ):
        """Return a FeatureCollection and the number of matched items."""
        return await self.query(
            pool=pool,
            ids_filter=ids_filter,
            bbox_filter=bbox_filter,
            datetime_filter=datetime_filter,
            properties_filter=properties_filter,
            cql_filter=cql_filter,
            properties=properties,
            limit=limit,
            offset=offset,
            **kwargs,
        )

    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
        properties: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Return a Feature."""
        feature_collection, _ = await self.query(
            pool=pool,
            ids_filter=[item_id],
            properties=properties,
            **kwargs,
        )
        try:
            return await feature_collection.__anext__()
        except:
            return None

    @property
    def queryables(self) -> Dict:
        """Return the queryables."""
        if self.geometry_columns:
            geoms = {
                col.name: {"$ref": geojson_schema.get(col.geometry_type.upper(), "")}
                for col in self.geometry_columns
            }
        else:
            geoms = {}
        props = {
            col.name: {"name": col.name, "type": col.json_type}
            for col in self.properties
            if col.name not in geoms
        }
        return {**geoms, **props}


class Function(CollectionLayer):
    """Function Reader.

    Attributes:
        id (str): Layer's name.
        bounds (list): Layer's bounds (left, bottom, right, top).
        type (str): Layer's type.
        function_name (str): Name of the SQL function to call. Defaults to `id`.
        sql (str): Valid SQL function which returns Tile data.
        options (list, optional): options available for the SQL function.

    """

    type: str = "Function"
    sql: str
    function_name: Optional[str]
    options: Optional[List[Dict[str, Any]]]

    @root_validator
    def function_name_default(cls, values):
        """Define default function's name to be same as id."""
        function_name = values.get("function_name")
        if function_name is None:
            values["function_name"] = values.get("id")
        return values

    @classmethod
    def from_file(cls, id: str, infile: str, **kwargs: Any):
        """load sql from file"""
        with open(infile) as f:
            sql = f.read()

        return cls(id=id, sql=sql, **kwargs)

    async def features(
        self,
        pool: asyncpg.BuildPgPool,
        ids_filter: Optional[List[str]] = None,
        bbox_filter: Optional[List[float]] = None,
        datetime_filter: Optional[List[str]] = None,
        properties_filter: Optional[List[Tuple[str, str]]] = None,
        cql_filter: Optional[AstType] = None,
        properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[FeatureCollection, int]:
        """Return a FeatureCollection and the number of matched items."""
        # TODO
        pass

    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
        properties: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Feature:
        """Return a Feature."""
        # TODO
        pass

    @property
    def queryables(self) -> Dict:
        """Return the queryables."""
        # TODO
        pass


@dataclass
class FunctionRegistry:
    """function registry"""

    funcs: ClassVar[Dict[str, Function]] = {}

    @classmethod
    def get(cls, key: str):
        """lookup function by name."""
        return cls.funcs.get(key)

    @classmethod
    def register(cls, *args: Function):
        """register function(s)."""
        for func in args:
            cls.funcs[func.id] = func

    @classmethod
    def values(cls):
        """get all values."""
        return cls.funcs.values()
