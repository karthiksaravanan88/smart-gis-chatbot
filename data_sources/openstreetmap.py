"""Dynamic OpenStreetMap access via the Overpass API."""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import Point

from ai_module.query_parser import ENTITY_ALIASES, GeoIntent
from gis_engine.spatial_analysis import SpatialAnalysisEngine
from map_service.map_generator import MapGenerator


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class OpenStreetMapQueryService:
    def __init__(self, spatial_engine: SpatialAnalysisEngine | None = None, map_generator: MapGenerator | None = None) -> None:
        self.spatial_engine = spatial_engine or SpatialAnalysisEngine()
        self.map_generator = map_generator or MapGenerator()

    def run(self, intent: GeoIntent, map_filename: str) -> dict[str, Any]:
        districts = self.spatial_engine.load_tamil_nadu_boundaries()
        state_boundary = self.spatial_engine.build_state_boundary(districts)
        location_frame = self._resolve_location_frame(intent.location, districts, state_boundary)
        bbox = self._bbox(location_frame)
        tag_key, tag_value = self._resolve_tag(intent.entity or "hospital")
        query = self._build_overpass_query(tag_key, tag_value, bbox)
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=(10, 90))
        response.raise_for_status()
        payload = response.json()
        features = self._elements_to_frame(payload.get("elements", []), intent.entity or "feature")

        if features.empty:
            raise ValueError(f"No OpenStreetMap features found for {intent.entity} in {intent.location}.")

        features = gpd.clip(features, location_frame)
        if features.empty:
            raise ValueError(f"No {intent.entity} records remained after clipping to {intent.location}.")

        counts, joined_features = self._district_counts(features, districts)
        map_path = self.map_generator.create_feature_map(
            features=joined_features,
            map_filename=map_filename,
            title=f"{intent.entity.title()} Results",
            districts=districts,
            state_boundary=state_boundary,
            metadata={
                "center": self._center(location_frame),
                "bounds": self.spatial_engine._compute_bounds(location_frame),
                "show_heatmap": intent.analysis == "hotspot",
            },
        )

        top_locations = []
        for row in joined_features.head(10).itertuples():
            top_locations.append(
                {
                    "name": getattr(row, "name", f"{intent.entity.title()} site"),
                    "district": getattr(row, "district", intent.location),
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x),
                }
            )

        metadata = {
            "source": "osm",
            "entity": intent.entity,
            "location": intent.location,
            "analysis": intent.analysis,
            "total_features": int(len(features)),
            "district_counts": counts,
            "top_locations": top_locations,
            "filters": intent.filters,
        }

        return {
            "source": "osm",
            "parsed_query": intent.model_dump(),
            "map_path": str(map_path),
            "map_url": f"/maps/{map_path.name}",
            "metadata": metadata,
        }

    def _resolve_location_frame(
        self,
        location: str,
        districts: gpd.GeoDataFrame,
        state_boundary: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        if location == "Tamil Nadu":
            return state_boundary
        subset = districts[districts["district"].str.lower() == location.lower()]
        return subset if not subset.empty else state_boundary

    def _resolve_tag(self, entity: str) -> tuple[str, str]:
        normalized = (entity or "hospital").lower()
        if normalized in ENTITY_ALIASES:
            return ENTITY_ALIASES[normalized]
        return ("amenity", normalized)

    def _build_overpass_query(self, tag_key: str, tag_value: str, bbox: tuple[float, float, float, float]) -> str:
        south, west, north, east = bbox
        value_clause = "" if tag_value == "*" else f'="{tag_value}"'
        return (
            "[out:json][timeout:25];("
            f'node["{tag_key}"{value_clause}]({south},{west},{north},{east});'
            f'way["{tag_key}"{value_clause}]({south},{west},{north},{east});'
            f'relation["{tag_key}"{value_clause}]({south},{west},{north},{east});'
            ");out center;"
        )

    def _elements_to_frame(self, elements: list[dict[str, Any]], entity: str) -> gpd.GeoDataFrame:
        rows = []
        for element in elements:
            lat = element.get("lat") or element.get("center", {}).get("lat")
            lon = element.get("lon") or element.get("center", {}).get("lon")
            if lat is None or lon is None:
                continue
            tags = element.get("tags", {})
            rows.append(
                {
                    "osm_id": element.get("id"),
                    "name": tags.get("name", f"{entity.title()} {element.get('id')}"),
                    "category": entity,
                    "entity": entity,
                    "raw_tags": json.dumps(tags, ensure_ascii=True),
                    "geometry": Point(float(lon), float(lat)),
                }
            )
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def _district_counts(self, features: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> tuple[list[dict[str, Any]], gpd.GeoDataFrame]:
        joined = gpd.sjoin(features, districts, predicate="within", how="left")
        joined["district"] = joined["district"].fillna("Outside District Boundary")
        counts = joined.groupby("district").size().reset_index(name="record_count")
        counts = counts.sort_values("record_count", ascending=False)
        return (
            [
                {"district": row.district, "record_count": int(row.record_count)}
                for row in counts.itertuples()
            ],
            joined.drop(columns=[col for col in joined.columns if col.startswith("index_")], errors="ignore"),
        )

    def _bbox(self, frame: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
        minx, miny, maxx, maxy = frame.total_bounds
        return float(miny), float(minx), float(maxy), float(maxx)

    def _center(self, frame: gpd.GeoDataFrame) -> list[float]:
        centroid = frame.to_crs(epsg=3857).geometry.unary_union.centroid
        center = gpd.GeoSeries([centroid], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        return [float(center.y), float(center.x)]
