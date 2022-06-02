"""tifeatures.factory: router factories."""

import json
import pathlib
from os import PathLike
from typing import Union, List, Any
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from geojson_pydantic.geometries import Polygon

from tifeatures import model
from tifeatures.dependencies import CollectionParams, OutputType, bbox_query
from tifeatures.errors import NotFound
from tifeatures.layer import CollectionLayer
from tifeatures.resources.enums import MediaType, ResponseType
from tifeatures.resources.response import GeoJSONResponse
from tifeatures.settings import APISettings

from fastapi import APIRouter, Depends, Path, Query

from starlette.datastructures import QueryParams
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

import jinja2

try:
    from jinja2 import pass_context
except ImportError:
    # jinja2 < 3.0 fallback
    from jinja2 import contextfilter as pass_context

settings = APISettings()

class SearchPathTemplates(Jinja2Templates):
    """
    templates = DefaultTemplates("templates")
    return templates.TemplateResponse("index.html", {"request": request})
    """

    def __init__(
        self, directory: Union[str, PathLike, List[Union[str, None, PathLike]]], **env_options: Any
    ) -> None:
        assert jinja2 is not None, "jinja2 must be installed to use Jinja2Templates"
        self.env = self._create_env(directory, **env_options)

    def _create_env(
        self, directory: Union[str, PathLike, List[Union[str, None, PathLike]]], **env_options: Any
    ) -> "jinja2.Environment":
        @pass_context
        def url_for(context: dict, name: str, **path_params: Any) -> str:
            request = context["request"]
            return request.url_for(name, **path_params)

        if isinstance(directory, list):
            loader = jinja2.FileSystemLoader([d for d in directory if d])
        else:
            loader = jinja2.FileSystemLoader(directory)
        env_options.setdefault("loader", loader)
        env_options.setdefault("autoescape", True)

        env = jinja2.Environment(**env_options)
        env.globals["url_for"] = url_for
        return env


default_template_dir = str(pathlib.Path(__file__).parent.joinpath("templates"))
templates = SearchPathTemplates(directory=[settings.template_directory, default_template_dir])

def create_html_response(
    request: Request, data: str, template_name: str
) -> HTMLResponse:
    """Create Template response."""
    urlpath = request.url.path
    crumbs = []
    baseurl = str(request.base_url).rstrip("/")
    crumbpath = str(baseurl)
    for crumb in urlpath.split("/"):
        crumbpath = crumbpath.rstrip("/")
        part = crumb
        if part is None or part == "":
            part = "Home"
        crumbpath += f"/{crumb}"
        crumbs.append({"url": crumbpath.rstrip("/"), "part": part.capitalize()})

    return templates.TemplateResponse(
        f"{template_name}.html",
        {
            "request": request,
            "response": json.loads(data),
            "template": {
                "api_root": baseurl,
                "params": request.query_params,
                "title": "",
            },
            "crumbs": crumbs,
            "json_url": str(request.url).replace("f=html", "f=json"),
        },
    )


@dataclass
class Endpoints:
    """Endpoints Factory."""

    # FastAPI router
    router: APIRouter = field(default_factory=APIRouter)

    # collection dependency
    collection_dependency: Callable[..., CollectionLayer] = CollectionParams

    # Router Prefix is needed to find the path for routes when prefixed
    # e.g if you mount the route with `/foo` prefix, set router_prefix to foo
    router_prefix: str = ""

    def __post_init__(self):
        """Post Init: register route and configure specific options."""
        self.register_landing()
        self.register_conformance()
        self.register_collections()

    def url_for(self, request: Request, name: str, **path_params: Any) -> str:
        """Return full url (with prefix) for a specific handler."""
        url_path = self.router.url_path_for(name, **path_params)

        base_url = str(request.base_url)
        if self.router_prefix:
            base_url += self.router_prefix.lstrip("/")

        return url_path.make_absolute_url(base_url=base_url)

    def register_landing(self) -> None:
        """Register landing endpoint."""

        @self.router.get(
            "/",
            response_model=model.Landing,
            response_model_exclude_none=True,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/json": {},
                    }
                },
            },
        )
        def landing(
            request: Request,
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            """Get conformance."""
            data = model.Landing(
                title=settings.name,
                links=[
                    model.Link(
                        title="Landing Page",
                        href=self.url_for(request, "landing"),
                        type=MediaType.json,
                        rel="self",
                    ),
                    model.Link(
                        title="HTML Landing Page",
                        href=self.url_for(request, "landing") + "?f=html",
                        type=MediaType.html,
                        rel="alternate",
                    ),
                    model.Link(
                        title="the API definition",
                        href=request.url_for("openapi"),
                        type=MediaType.openapi30_json,
                        rel="service-desc",
                    ),
                    model.Link(
                        title="the API documentation",
                        href=request.url_for("swagger_ui_html"),
                        type=MediaType.html,
                        rel="service-doc",
                    ),
                    model.Link(
                        title="Conformance",
                        href=self.url_for(request, "conformance"),
                        type=MediaType.json,
                        rel="conformance",
                    ),
                    model.Link(
                        title="List of Collections",
                        href=self.url_for(request, "collections"),
                        type=MediaType.json,
                        rel="data",
                    ),
                    model.Link(
                        title="Collection metadata",
                        href=self.url_for(
                            request,
                            "collection",
                            collectionId="{collectionId}",
                        ),
                        type=MediaType.json,
                        rel="data",
                    ),
                    model.Link(
                        title="Collection Features",
                        href=self.url_for(
                            request, "items", collectionId="{collectionId}"
                        ),
                        type=MediaType.geojson,
                        rel="data",
                    ),
                    model.Link(
                        title="Collection Feature",
                        href=self.url_for(
                            request,
                            "item",
                            collectionId="{collectionId}",
                            itemId="{itemId}",
                        ),
                        type=MediaType.geojson,
                        rel="data",
                    ),
                ],
            )

            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="landing",
                )

            return data

    def register_conformance(self) -> None:
        """Register conformance endpoint."""

        @self.router.get(
            "/conformance",
            response_model=model.Conformance,
            response_model_exclude_none=True,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/json": {},
                    }
                },
            },
        )
        def conformance(
            request: Request,
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            """Get conformance."""
            data = model.Conformance(
                conformsTo=[
                    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
                    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas3",
                    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
                    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/html",
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/landing-page",
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/json",
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/html",
                    "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/oas30",
                    "http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/collections",
                    "http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/simple-query",
                ]
            )
            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="conformance",
                )

            return data

    def register_collections(self):  # noqa
        """Register Collections endpoints."""

        @self.router.get(
            "/collections",
            response_model=model.Collections,
            response_model_exclude_none=True,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/json": {},
                    }
                },
            },
        )
        def collections(
            request: Request,
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            """List of collections."""
            functions = getattr(request.app.state, "function_catalog", {})
            tables = getattr(request.app.state, "table_catalog", [])

            data = model.Collections(
                links=[
                    model.Link(
                        href=self.url_for(request, "landing"),
                        rel="parent",
                        type=MediaType.json,
                    ),
                    model.Link(
                        href=self.url_for(request, "collections"),
                        rel="self",
                        type=MediaType.json,
                    ),
                ],
                collections=[
                    model.Collection(
                        **{
                            **collection.dict(),
                            "links": [
                                model.Link(
                                    href=self.url_for(
                                        request,
                                        "collection",
                                        collectionId=collection.id,
                                    ),
                                    rel="collection",
                                    type=MediaType.json,
                                ),
                                model.Link(
                                    href=self.url_for(
                                        request,
                                        "items",
                                        collectionId=collection.id,
                                    ),
                                    rel="items",
                                    type=MediaType.geojson,
                                ),
                            ],
                        }
                    )
                    for collection in [
                        *tables,
                        *list(functions.values()),
                    ]
                ],
            )

            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="collections",
                )

            return data

        @self.router.get(
            "/collections/{collectionId}",
            response_model=model.Collection,
            response_model_exclude_none=True,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/json": {},
                    }
                },
            },
        )
        def collection(
            request: Request,
            collection=Depends(self.collection_dependency),
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            """Metadata for a feature collection."""
            data = model.Collection(
                **{
                    **collection.dict(),
                    "links": [
                        model.Link(
                            href=self.url_for(
                                request,
                                "collection",
                                collectionId=collection.id,
                            ),
                            rel="self",
                            type=MediaType.json,
                        ),
                        model.Link(
                            href=self.url_for(
                                request, "items", collectionId=collection.id
                            ),
                            rel="items",
                            type=MediaType.geojson,
                        ),
                    ],
                }
            )

            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="collection",
                )

            return data

        @self.router.get(
            "/collections/{collectionId}/items",
            response_model=model.Items,
            response_model_exclude_none=True,
            response_class=GeoJSONResponse,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/geo+json": {},
                    }
                },
            },
        )
        async def items(
            request: Request,
            collection=Depends(self.collection_dependency),
            limit: int = Query(
                10,
                description="Limits the number of features in the response.",
            ),
            offset: Optional[int] = Query(
                None,
                ge=0,
                description="Starts the response at an offset.",
            ),
            intersects: Optional[Polygon] = Depends(bbox_query),
            properties: Optional[str] = Query(
                None,
                description="Return only specific properties (comma-separated). If PROP-LIST is empty, no properties are returned. If not present, all properties are returned.",
            ),
            sortby: Optional[str] = Query(
                None,
                description="Sort the response items by a property (ascending (default) or descending).",
            ),
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            offset = offset or 0

            # req ={}
            # if bbox:
            #     req["filter"]["args"].append(
            #         {
            #             "op": "s_intersects",
            #             "args": [{"property": "geometry"}, bbox.dict()],
            #         }
            #     )
            # # <propname>=val - filter features for a property having a value. Multiple property filters are ANDed together.
            # qs_key_to_remove = ["limit", "offset", "bbox", "properties", "sortby"]
            # propname = [
            #     {"op": "eq", "args": [{"property": key}, value]}
            #     for (key, value) in request.query_params.items()
            #     if key.lower() not in qs_key_to_remove
            # ]
            # if propname:
            #     req["filter"]["args"].append(*propname)
            #
            # # sortby=[+|-]PROP - sort the response items by a property (ascending (default) or descending).
            # if sortby:
            #     sort_expr = []
            #     for s in sortby.split(","):
            #         parts = re.match(
            #             "^(?P<dir>[+-]?)(?P<prop>.*)$", s
            #         ).groupdict()  # type:ignore
            #         sort_expr.append(
            #             {
            #                 "field": f"properties.{parts['prop']}",
            #                 "direction": "desc" if parts["dir"] == "-" else "asc",
            #             }
            #         )
            #     req["sortby"] = sort_expr
            # # properties=PROP-LIST- return only specific properties (comma-separated). If PROP-LIST is empty, no properties are returned. If not present, all properties are returned.
            # if properties is not None:
            #     if properties == "":
            #         req["fields"]["exclude"].append("properties")
            #     else:
            #         required_props = ["type", "geometry", "id", "bbox", "assets"]
            #         req["fields"].update(
            #             {
            #                 "include": required_props
            #                 + [f"properties.{p}" for p in properties.split(",")]
            #             }
            #         )

            # TODO: create filter using bbox/properties by using pygeofilter
            # TODO: use sortby
            items = await collection.features(
                request.app.state.pool,
                limit=limit,
                offset=offset,
            )

            qs = "?" + str(request.query_params) if request.query_params else ""
            links = [
                model.Link(
                    href=self.url_for(
                        request, "collection", collectionId=collection.id
                    ),
                    rel="collection",
                    type=MediaType.json,
                ),
                model.Link(
                    href=self.url_for(request, "items", collectionId=collection.id)
                    + qs,
                    rel="self",
                    type=MediaType.geojson,
                ),
            ]

            items_returned = len(items["features"])
            matched_items = items["total_count"]

            if (matched_items - items_returned) > offset:
                next_offset = offset + items_returned
                query_params = QueryParams(
                    {**request.query_params, "offset": next_offset}
                )
                url = (
                    self.url_for(request, "items", collectionId=collection.id)
                    + f"?{query_params}"
                )
                links.append(
                    model.Link(href=url, rel="next", type=MediaType.geojson),
                )

            if offset:
                query_params = dict(request.query_params)
                query_params.pop("offset")
                prev_offset = max(offset - items_returned, 0)
                if prev_offset:
                    query_params = QueryParams({**query_params, "offset": prev_offset})
                else:
                    query_params = QueryParams({**query_params})

                url = self.url_for(request, "items", collectionId=collection.id)
                if query_params:
                    url += f"?{query_params}"

                links.append(
                    model.Link(href=url, rel="prev", type=MediaType.geojson),
                )

            data = model.Items(
                id=collection.id,
                title=collection.title or collection.id,
                description=collection.description or collection.title or collection.id,
                numberMatched=matched_items,
                numberReturned=items_returned,
                links=links,
                features=[
                    model.Item(
                        **{
                            **feature,
                            "links": [
                                model.Link(
                                    href=self.url_for(
                                        request,
                                        "collection",
                                        collectionId=collection.id,
                                    ),
                                    rel="collection",
                                    type=MediaType.json,
                                ),
                                model.Link(
                                    href=self.url_for(
                                        request,
                                        "item",
                                        collectionId=collection.id,
                                        itemId=feature["properties"][
                                            collection.id_column
                                        ],
                                    ),
                                    rel="item",
                                    type=MediaType.json,
                                ),
                            ],
                        }
                    )
                    for feature in items["features"]
                ],
            )

            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="items",
                )

            return data

        @self.router.get(
            "/collections/{collectionId}/items/{itemId}",
            response_model=model.Item,
            response_model_exclude_none=True,
            response_class=GeoJSONResponse,
            responses={
                200: {
                    "content": {
                        "text/html": {},
                        "application/geo+json": {},
                    }
                },
            },
        )
        async def item(
            request: Request,
            collection=Depends(self.collection_dependency),
            itemId: str = Path(..., description="Item identifier"),
            output_type: Optional[ResponseType] = Depends(OutputType),
        ):
            feature = await collection.feature(
                request.app.state.pool,
                item_id=itemId,
            )

            if not feature:
                raise NotFound(
                    f"Item {itemId} in Collection {collection.id} does not exist."
                )

            data = model.Item(
                **feature,
                links=[
                    model.Link(
                        href=self.url_for(
                            request, "collection", collectionId=collection.id
                        ),
                        rel="collection",
                        type=MediaType.json,
                    ),
                    model.Link(
                        href=self.url_for(
                            request,
                            "item",
                            collectionId=collection.id,
                            itemId=itemId,
                        ),
                        rel="self",
                        type=MediaType.geojson,
                    ),
                ],
            )

            if output_type and output_type == ResponseType.html:
                return create_html_response(
                    request,
                    data.json(exclude_none=True),
                    template_name="item",
                )

            return data
