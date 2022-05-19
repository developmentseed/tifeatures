"""tifeatures.layers."""

import abc
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional

from buildpg import Var as pg_variable
from buildpg import asyncpg, clauses, S
from buildpg import funcs as pg_funcs
from buildpg import render
from geojson_pydantic import Feature, FeatureCollection
from pydantic import BaseModel, Field, root_validator


class Items(FeatureCollection):
    """Custom FeatureCollection Model return by `features` methods."""

    total_count: int


class CollectionLayer(BaseModel, metaclass=abc.ABCMeta):
    """Layer's Abstract BaseClass.

    Attributes:
        id (str): Layer's name.
        bounds (list): Layer's bounds (left, bottom, right, top).
        title (str): Layer's title
        description (str): Layer's description

    """

    id: str
    bounds: List[float] = [-180, -90, 180, 90]
    title: Optional[str]
    description: Optional[str]

    @abc.abstractmethod
    async def features(
        self,
        pool: asyncpg.BuildPgPool,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> Items:
        """Return a FeatureCollection."""
        ...

    @abc.abstractmethod
    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
    ) -> Feature:
        """Return a Feature."""
        ...



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
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs: Any,
    ) -> FeatureCollection:
        """Return a FeatureCollection."""
        # TODO
        pass

    async def feature(
        self,
        pool: asyncpg.BuildPgPool,
        item_id: str,
    ) -> Feature:
        """Return a Feature."""
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
