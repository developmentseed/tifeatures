"""tifeatures dependencies."""

import re
from typing import Callable, List, Literal, Optional

from tifeatures.layer import CollectionLayer
from tifeatures.resources.enums import AcceptType, ResponseType

from fastapi import HTTPException, Path, Query

from starlette.requests import Request

from pygeofilter.parsers import cql2_json, cql2_text

def CollectionParams(
    request: Request,
    collectionId: str = Path(..., description="Collection identifier"),
) -> CollectionLayer:
    """Return Layer Object."""
    # Check function_catalog
    function_catalog = getattr(request.app.state, "function_catalog", {})
    func = function_catalog.get(collectionId)
    if func:
        return func

    # Check table_catalog
    else:
        table_pattern = re.match(  # type: ignore
            r"^(?P<schema>.+)\.(?P<table>.+)$", collectionId
        )
        if not table_pattern:
            raise HTTPException(
                status_code=404, detail=f"Invalid Table format '{collectionId}'."
            )

        assert table_pattern.groupdict()["schema"]
        assert table_pattern.groupdict()["table"]

        table_catalog = getattr(request.app.state, "table_catalog", [])
        for r in table_catalog.tables:
            if r.id == collectionId:
                return r

    raise HTTPException(
        status_code=404, detail=f"Table/Function '{collectionId}' not found."
    )


def bbox_query(
    bbox: Optional[str] = Query(
        None,
        description="Spatial Filter.",
    )
) -> Optional[List[float]]:
    """BBox dependency."""
    if bbox:
        bounds = list(map(float, bbox.split(",")))
        if len(bounds) == 4:
            if abs(bounds[0]) > 180 or abs(bounds[2]) > 180:
                raise ValueError(f"Invalid longitude in bbox: {bounds}")
            if abs(bounds[1]) > 90 or abs(bounds[3]) > 90:
                raise ValueError(f"Invalid latitude in bbox: {bounds}")

        elif len(bounds) == 6:
            if abs(bounds[0]) > 180 or abs(bounds[3]) > 180:
                raise ValueError(f"Invalid longitude in bbox: {bounds}")
            if abs(bounds[1]) > 90 or abs(bounds[4]) > 90:
                raise ValueError(f"Invalid latitude in bbox: {bounds}")
        else:
            raise Exception("Invalid BBOX.")

        return bounds

    return None


def datetime_query(
    datetime: Optional[str] = Query(None, description="Temporal Filter."),
) -> Optional[str]:
    """Datetime dependency."""
    # TODO validation / format
    return datetime

def properties_query(
    properties: Optional[str] = Query(
        None,
        description="Return only specific properties (comma-separated). If PROP-LIST is empty, no properties are returned. If not present, all properties are returned.",
    )
) -> Optional[List[str]]:
    if properties is not None:
        return [p.strip() for p in properties.split(',')]

def OutputType(
    request: Request,
    f: Optional[ResponseType] = Query(None, description="Response MediaType."),
) -> Optional[ResponseType]:
    """Output Response type."""
    if f:
        return f

    accept_header = request.headers.get("accept", "")
    if accept_header in AcceptType.__members__.values():
        return ResponseType[AcceptType(accept_header).name]

    return None

def filter_query(
    filter: Optional[str] = Query(
        None,
        description="CQL2 Filter"
    ),
    filterlang: Optional[Literal['cql2-text','cql2-json']] = Query(
        'cql2-text',
        description="CQL2 Language (cql2-text, cql2-json)",
        alias="filter-lang"
    )
) -> Optional[Callable]:
    if filter is not None:
        if filterlang == 'cql2-json':
            return cql2_json.parse(filter)
        else:
            return cql2_text.parse(filter)
    return None
