"""Interactive map generation using Folium."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import folium
import geopandas as gpd
from folium.features import GeoJsonTooltip
from folium.plugins import HeatMap, MarkerCluster

from gis_engine.spatial_analysis import AnalysisResult, GENERATED_MAPS_DIR


class MapGenerator:
    def create_map(self, result: AnalysisResult, map_filename: str) -> Path:
        fmap = folium.Map(
            location=result.center,
            zoom_start=7,
            control_scale=True,
            tiles="CartoDB positron",
            max_bounds=True,
            zoom_control=True,
        )
        fmap.fit_bounds(result.bounds)
        fmap.options["maxBounds"] = result.bounds

        self._add_state_boundary(fmap, result)
        self._add_district_layer(fmap, result)
        self._add_incident_layer(fmap, result)
        self._add_heatmap_layer(fmap, result)
        self._add_hotspot_centers(fmap, result)
        self._add_analysis_overlay(fmap, result)

        folium.LayerControl(collapsed=False).add_to(fmap)
        output_path = GENERATED_MAPS_DIR / map_filename
        fmap.save(output_path)
        return output_path

    def create_feature_map(
        self,
        *,
        features: gpd.GeoDataFrame,
        map_filename: str,
        title: str,
        districts: gpd.GeoDataFrame | None = None,
        state_boundary: gpd.GeoDataFrame | None = None,
        metadata: dict | None = None,
    ) -> Path:
        metadata = metadata or {}
        center = metadata.get("center", [11.1271, 78.6569])
        bounds = metadata.get("bounds")
        fmap = folium.Map(
            location=center,
            zoom_start=7,
            control_scale=True,
            tiles="CartoDB positron",
            max_bounds=True,
            zoom_control=True,
        )
        if bounds:
            fmap.fit_bounds(bounds)
            fmap.options["maxBounds"] = bounds

        if state_boundary is not None and not state_boundary.empty:
            folium.GeoJson(
                data=state_boundary.__geo_interface__,
                name="Tamil Nadu Boundary",
                style_function=lambda _: {"color": "#0f172a", "weight": 3, "fillOpacity": 0.02},
                tooltip=GeoJsonTooltip(fields=["state"]),
            ).add_to(fmap)

        if districts is not None and not districts.empty:
            folium.GeoJson(
                data=districts.__geo_interface__,
                name="District Boundaries",
                style_function=lambda _: {"color": "#475569", "weight": 1, "fillOpacity": 0.04},
                tooltip=GeoJsonTooltip(fields=["district"]),
            ).add_to(fmap)

        cluster = MarkerCluster(name=title).add_to(fmap)
        heat_points = []
        for row in features.itertuples():
            if row.geometry.geom_type != "Point":
                continue
            popup_parts = []
            for field in ("name", "category", "entity", "district"):
                value = getattr(row, field, None)
                if value:
                    popup_parts.append(f"{field.title()}: {value}")
            popup = "<br>".join(popup_parts) if popup_parts else title
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#0f766e",
                fill=True,
                fill_color="#2dd4bf",
                fill_opacity=0.8,
                tooltip=getattr(row, "name", getattr(row, "entity", title)),
                popup=popup,
            ).add_to(cluster)
            weight = float(getattr(row, "density_weight", 0.5))
            heat_points.append([row.geometry.y, row.geometry.x, weight])

        if heat_points and metadata.get("show_heatmap"):
            HeatMap(
                heat_points,
                name=f"{title} Heatmap",
                radius=24,
                blur=18,
                min_opacity=0.35,
                max_zoom=11,
            ).add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        output_path = GENERATED_MAPS_DIR / map_filename
        fmap.save(output_path)
        return output_path

    def _add_state_boundary(self, fmap: folium.Map, result: AnalysisResult) -> None:
        folium.GeoJson(
            data=result.state_boundary.__geo_interface__,
            name="Tamil Nadu Boundary",
            style_function=lambda _: {"color": "#0f172a", "weight": 3, "fillOpacity": 0.02},
            tooltip=GeoJsonTooltip(fields=["state"]),
        ).add_to(fmap)

    def _add_district_layer(self, fmap: folium.Map, result: AnalysisResult) -> None:
        def style(feature: Dict) -> Dict[str, object]:
            count = feature["properties"].get("record_count", 0) or 0
            if count >= 4:
                fill = "#ef4444"
            elif count >= 2:
                fill = "#f59e0b"
            elif count > 0:
                fill = "#fde68a"
            else:
                fill = "#e2e8f0"
            return {"fillColor": fill, "color": "#475569", "weight": 1, "fillOpacity": 0.35}

        folium.GeoJson(
            data=result.districts.__geo_interface__,
            name="District Boundaries",
            style_function=style,
            tooltip=GeoJsonTooltip(fields=["district", "record_count", "density_per_sqkm"]),
        ).add_to(fmap)

    def _add_incident_layer(self, fmap: folium.Map, result: AnalysisResult) -> None:
        cluster = MarkerCluster(name="Accident Points").add_to(fmap)
        for row in result.points.itertuples():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="#1d4ed8",
                fill=True,
                fill_color="#60a5fa",
                fill_opacity=0.8,
                tooltip=getattr(row, "name", "Incident"),
                popup=f"{getattr(row, 'name', 'Incident')}<br>{getattr(row, 'category', 'record')}",
            ).add_to(cluster)

    def _add_heatmap_layer(self, fmap: folium.Map, result: AnalysisResult) -> None:
        points = result.metadata.get("heatmap_points", [])
        if points:
            HeatMap(
                points,
                name="Hotspot Heatmap",
                radius=24,
                blur=18,
                min_opacity=0.35,
                max_zoom=11,
            ).add_to(fmap)

    def _add_hotspot_centers(self, fmap: folium.Map, result: AnalysisResult) -> None:
        feature_group = folium.FeatureGroup(name="Hotspot Clusters")
        for row in result.hotspot_centers.itertuples():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                popup=f"Cluster {int(row.cluster_id)}<br>{int(row.incident_count)} incidents",
            ).add_to(feature_group)
        feature_group.add_to(fmap)

    def _add_analysis_overlay(self, fmap: folium.Map, result: AnalysisResult) -> None:
        if result.analysis_type == "buffer":
            folium.GeoJson(
                data=result.analysis_layer.__geo_interface__,
                name="Buffer Zones",
                style_function=lambda _: {"color": "#2563eb", "weight": 2, "fillOpacity": 0.12},
            ).add_to(fmap)
        elif result.analysis_type == "spatial_join":
            folium.GeoJson(
                data=result.analysis_layer.__geo_interface__,
                name="District Density Overlay",
                style_function=lambda _: {"color": "#7c3aed", "weight": 2, "fillOpacity": 0.05},
            ).add_to(fmap)
