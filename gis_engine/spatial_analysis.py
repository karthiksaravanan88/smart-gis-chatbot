"""Spatial analysis engine for Smart GIS Chatbot."""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_SAMPLES_DIR = DATA_DIR / "Data Samples"
SHAPEFILES_DIR = ROOT_DIR / "shapefiles" / "TamilNadu"
GENERATED_MAPS_DIR = ROOT_DIR / "generated_maps"
TN_REPO_ZIP_URL = "https://github.com/datta07/INDIAN-SHAPEFILES/archive/refs/heads/master.zip"
TAMIL_NADU_CENTER = (11.1271, 78.6569)


@dataclass
class AnalysisResult:
    analysis_type: str
    points: gpd.GeoDataFrame
    districts: gpd.GeoDataFrame
    state_boundary: gpd.GeoDataFrame
    analysis_layer: gpd.GeoDataFrame
    hotspot_centers: gpd.GeoDataFrame
    center: Tuple[float, float]
    bounds: list[list[float]]
    summary: str
    metadata: Dict[str, Any]


class SpatialAnalysisEngine:
    def __init__(self) -> None:
        GENERATED_MAPS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        SHAPEFILES_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_sample_data()

    def run(self, parsed_query: Dict[str, Any]) -> AnalysisResult:
        districts = self.load_tamil_nadu_boundaries()
        state_boundary = self.build_state_boundary(districts)
        dataset = self.load_dataset(parsed_query.get("dataset", "accidents"))
        dataset = self.clip_to_state(dataset, state_boundary)
        dataset = self.apply_filters(dataset, parsed_query.get("filters", {}), districts)

        if dataset.empty:
            raise ValueError("No records matched the requested filters inside Tamil Nadu.")

        joined = self.join_points_to_districts(dataset, districts)
        district_stats = self.compute_district_statistics(joined, districts)
        points_with_density = self.enrich_point_density(dataset)
        hotspot_points, hotspot_centers = self.compute_hotspot_clusters(points_with_density)

        analysis = parsed_query.get("analysis", "heatmap")
        if analysis == "buffer":
            analysis_layer = self.create_buffer_layer(dataset, parsed_query.get("buffer_distance_m", 1000))
        elif analysis == "spatial_join":
            analysis_layer = district_stats
        else:
            analysis_layer = hotspot_points

        bounds = self._compute_bounds(state_boundary)
        metadata = self.build_metadata(
            parsed_query=parsed_query,
            dataset=dataset,
            hotspot_points=hotspot_points,
            hotspot_centers=hotspot_centers,
            district_stats=district_stats,
            bounds=bounds,
        )
        summary = self.build_summary(metadata)

        return AnalysisResult(
            analysis_type=analysis,
            points=dataset,
            districts=district_stats,
            state_boundary=state_boundary,
            analysis_layer=analysis_layer,
            hotspot_centers=hotspot_centers,
            center=TAMIL_NADU_CENTER,
            bounds=bounds,
            summary=summary,
            metadata=metadata,
        )

    def load_tamil_nadu_boundaries(self) -> gpd.GeoDataFrame:
        self._extract_local_archives()
        districts = self._find_boundary_dataset()
        if districts is None and self._remote_download_enabled():
            self._download_tamil_nadu_boundaries()
            districts = self._find_boundary_dataset()

        if districts is None:
            fallback = self._create_demo_districts()
            fallback.to_file(SHAPEFILES_DIR / "tamil_nadu_districts.geojson", driver="GeoJSON")
            districts = fallback
        elif isinstance(districts, Path):
            districts = gpd.read_file(districts)

        districts = districts.to_crs(epsg=4326)
        if "district" in districts.columns:
            districts["district"] = districts["district"]
        elif "DISTRICT" in districts.columns:
            districts["district"] = districts["DISTRICT"]
        else:
            name_col = next((col for col in districts.columns if "dist" in col.lower() or "name" in col.lower()), None)
            districts["district"] = districts[name_col] if name_col else [f"District {idx + 1}" for idx in range(len(districts))]
        return districts[["district", "geometry"]].dropna(subset=["geometry"]).reset_index(drop=True)

    def build_state_boundary(self, districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        state_geometry = districts.unary_union
        return gpd.GeoDataFrame([{"state": "Tamil Nadu", "geometry": state_geometry}], crs="EPSG:4326")

    def load_dataset(self, dataset_name: str) -> gpd.GeoDataFrame:
        self._extract_local_archives()
        file_candidates = {
            dataset_name,
            dataset_name.rstrip("s"),
            f"{dataset_name}.geojson",
            f"{dataset_name.rstrip('s')}.geojson",
            "accidents_tn.geojson" if dataset_name == "accidents" else "",
        }
        for path in DATA_SAMPLES_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".geojson", ".json", ".shp"}:
                if path.name.lower() in {candidate.lower() for candidate in file_candidates}:
                    frame = gpd.read_file(path).to_crs(epsg=4326)
                    return self._normalize_dataset(frame, dataset_name)

        root_accident = DATA_DIR / "accidents.geojson"
        if dataset_name == "accidents" and root_accident.exists() and root_accident.stat().st_size > 0:
            return self._normalize_dataset(gpd.read_file(root_accident).to_crs(epsg=4326), dataset_name)

        if dataset_name == "accidents":
            raise FileNotFoundError(
                "Accident dataset not found. Expected `Accidents_TN.geojson` from the provided `Data Samples.zip` under `data/Data Samples/`."
            )

        demo = self._build_demo_points(dataset_name)
        demo.to_file(DATA_SAMPLES_DIR / f"{dataset_name}.geojson", driver="GeoJSON")
        return demo

    def clip_to_state(self, dataset: gpd.GeoDataFrame, state_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        clipped = gpd.sjoin(dataset, state_boundary, predicate="within", how="inner")
        return clipped.drop(columns=[col for col in clipped.columns if col.startswith("index_") or col == "state"], errors="ignore")

    def apply_filters(
        self,
        dataset: gpd.GeoDataFrame,
        filters: Dict[str, Any],
        districts: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        filtered = dataset.copy()
        district_name = filters.get("district")
        if district_name:
            districts_subset = districts[districts["district"].str.lower() == district_name.lower()]
            if not districts_subset.empty:
                filtered = gpd.sjoin(filtered, districts_subset, predicate="within", how="inner")
                filtered = filtered.drop(columns=[col for col in filtered.columns if col.startswith("index_")], errors="ignore")
                filtered = self._coalesce_district_column(filtered)

        keyword = filters.get("keyword")
        if keyword and "category" in filtered.columns:
            filtered = filtered[filtered["category"].str.contains(keyword, case=False, na=False)]
        return filtered.reset_index(drop=True)

    def join_points_to_districts(self, dataset: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        joined = gpd.sjoin(dataset, districts, predicate="within", how="left")
        joined = joined.drop(columns=[col for col in joined.columns if col.startswith("index_")], errors="ignore")
        joined = self._coalesce_district_column(joined)
        joined["district"] = joined["district"].fillna("Outside District Boundary")
        return joined

    def _coalesce_district_column(self, frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        normalized = frame.copy()
        if "district" in normalized.columns:
            return normalized

        district_columns = [col for col in normalized.columns if col.startswith("district")]
        if not district_columns:
            return normalized

        preferred_order = ["district_right", "district_left", *district_columns]
        district_series = None
        for column in preferred_order:
            if column not in normalized.columns:
                continue
            if district_series is None:
                district_series = normalized[column]
            else:
                district_series = district_series.fillna(normalized[column])

        if district_series is not None:
            normalized["district"] = district_series

        cleanup_columns = [col for col in district_columns if col != "district"]
        return normalized.drop(columns=cleanup_columns, errors="ignore")

    def compute_district_statistics(self, joined: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        projected_districts = districts.to_crs(epsg=3857).copy()
        districts = districts.copy()
        districts["area_sqkm"] = projected_districts.geometry.area / 1_000_000
        counts = joined.groupby("district").size().reset_index(name="record_count")
        stats = districts.merge(counts, on="district", how="left").fillna({"record_count": 0})
        stats["record_count"] = stats["record_count"].astype(int)
        stats["density_per_sqkm"] = stats["record_count"] / stats["area_sqkm"].replace(0, np.nan)
        stats["density_per_sqkm"] = stats["density_per_sqkm"].fillna(0).round(4)
        return stats

    def enrich_point_density(self, dataset: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        enriched = dataset.to_crs(epsg=3857).copy()
        coordinates = np.column_stack((enriched.geometry.x, enriched.geometry.y))
        if len(enriched) == 1:
            enriched["density_weight"] = 1.0
            return enriched.to_crs(epsg=4326)

        bandwidth = 25_000 if len(enriched) > 8 else 15_000
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(coordinates)
        scores = kde.score_samples(coordinates)
        normalized = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)
        enriched["density_weight"] = (normalized * 0.9) + 0.1
        return enriched.to_crs(epsg=4326)

    def compute_hotspot_clusters(self, dataset: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        projected = dataset.to_crs(epsg=3857).copy()
        if len(projected) < 2:
            single = dataset.copy()
            single["cluster_id"] = -1
            single["is_hotspot"] = False
            centers = gpd.GeoDataFrame(columns=["cluster_id", "incident_count", "geometry"], crs="EPSG:4326")
            return single, centers

        coords = np.column_stack((projected.geometry.x, projected.geometry.y))
        clustering = DBSCAN(eps=25_000, min_samples=2).fit(coords)
        projected["cluster_id"] = clustering.labels_
        projected["is_hotspot"] = projected["cluster_id"] >= 0

        hotspot_points = projected.to_crs(epsg=4326)
        center_rows = []
        clustered = projected[projected["cluster_id"] >= 0]
        for cluster_id, frame in clustered.groupby("cluster_id"):
            center_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "incident_count": int(len(frame)),
                    "geometry": frame.unary_union.centroid,
                }
            )
        hotspot_centers = gpd.GeoDataFrame(center_rows, crs="EPSG:3857").to_crs(epsg=4326) if center_rows else gpd.GeoDataFrame(
            columns=["cluster_id", "incident_count", "geometry"], crs="EPSG:4326"
        )
        return hotspot_points, hotspot_centers

    def create_buffer_layer(self, dataset: gpd.GeoDataFrame, buffer_distance_m: int) -> gpd.GeoDataFrame:
        projected = dataset.to_crs(epsg=3857).copy()
        projected["geometry"] = projected.geometry.buffer(buffer_distance_m)
        return projected.to_crs(epsg=4326)

    def build_metadata(
        self,
        parsed_query: Dict[str, Any],
        dataset: gpd.GeoDataFrame,
        hotspot_points: gpd.GeoDataFrame,
        hotspot_centers: gpd.GeoDataFrame,
        district_stats: gpd.GeoDataFrame,
        bounds: list[list[float]],
    ) -> Dict[str, Any]:
        top_districts = district_stats.sort_values(["record_count", "density_per_sqkm"], ascending=False).head(5)
        cluster_distribution = []
        if "cluster_id" in hotspot_points.columns:
            counts = hotspot_points[hotspot_points["cluster_id"] >= 0].groupby("cluster_id").size().reset_index(name="incident_count")
            cluster_distribution = [
                {"cluster_id": int(row.cluster_id), "incident_count": int(row.incident_count)}
                for row in counts.itertuples()
            ]

        top_locations = []
        sortable_points = hotspot_points.copy()
        if "density_weight" in sortable_points.columns:
            sortable_points = sortable_points.sort_values("density_weight", ascending=False).head(5)
            top_locations = [
                {
                    "name": getattr(row, "name", f"Location {index + 1}"),
                    "district": getattr(row, "district", "Unknown"),
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x),
                    "density_weight": float(getattr(row, "density_weight", 0.0)),
                }
                for index, row in enumerate(sortable_points.itertuples())
                if row.geometry.geom_type == "Point"
            ]

        return {
            "dataset": parsed_query.get("dataset", "accidents"),
            "analysis": parsed_query.get("analysis", "heatmap"),
            "total_incidents": int(len(dataset)),
            "cluster_count": int(len(hotspot_centers)),
            "hotspot_records": int(hotspot_points.get("is_hotspot", pd.Series(dtype=bool)).sum()) if "is_hotspot" in hotspot_points.columns else 0,
            "district_density": [
                {
                    "district": row.district,
                    "record_count": int(row.record_count),
                    "density_per_sqkm": float(row.density_per_sqkm),
                }
                for row in district_stats.itertuples()
            ],
            "top_districts": [
                {
                    "district": row.district,
                    "record_count": int(row.record_count),
                    "density_per_sqkm": float(row.density_per_sqkm),
                }
                for row in top_districts.itertuples()
            ],
            "cluster_distribution": cluster_distribution,
            "top_locations": top_locations,
            "map_bounds": bounds,
            "filters": parsed_query.get("filters", {}),
            "hotspot_centers": [
                {
                    "cluster_id": int(row.cluster_id),
                    "incident_count": int(row.incident_count),
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x),
                }
                for row in hotspot_centers.itertuples()
            ],
            "heatmap_points": [
                [float(row.geometry.y), float(row.geometry.x), float(getattr(row, "density_weight", 0.5))]
                for row in hotspot_points.itertuples()
                if row.geometry.geom_type == "Point"
            ],
        }

    def build_summary(self, metadata: Dict[str, Any]) -> str:
        top = metadata["top_districts"][:3]
        if not top:
            return "No district-level pattern was detected."
        lead = ", ".join(f"{item['district']} ({item['record_count']})" for item in top)
        return (
            f"Analyzed {metadata['total_incidents']} incidents across Tamil Nadu. "
            f"Leading districts are {lead}, with {metadata['cluster_count']} hotspot cluster(s) detected."
        )

    def _compute_bounds(self, frame: gpd.GeoDataFrame) -> list[list[float]]:
        minx, miny, maxx, maxy = frame.total_bounds
        return [[float(miny), float(minx)], [float(maxy), float(maxx)]]

    def _extract_local_archives(self) -> None:
        for zip_path in [DATA_DIR / "Data Samples.zip", ROOT_DIR / "Data Samples.zip"]:
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(DATA_SAMPLES_DIR)
        nested_zip = DATA_SAMPLES_DIR / "tamilnadu_accident_testdata_shapefile.zip"
        if nested_zip.exists():
            nested_target = DATA_SAMPLES_DIR / "tamilnadu_accident_testdata_shapefile"
            nested_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(nested_zip, "r") as archive:
                archive.extractall(nested_target)

    def _find_boundary_dataset(self) -> Path | gpd.GeoDataFrame | None:
        for path in SHAPEFILES_DIR.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if lowered.endswith((".shp", ".geojson")) and ("district" in lowered or "admin" in lowered):
                return path
        geojson = SHAPEFILES_DIR / "tamil_nadu_districts.geojson"
        if geojson.exists():
            return geojson
        return None

    def _download_tamil_nadu_boundaries(self) -> None:
        archive_path = SHAPEFILES_DIR / "indian_shapefiles_master.zip"
        if archive_path.exists():
            return
        try:
            response = requests.get(TN_REPO_ZIP_URL, timeout=5)
            response.raise_for_status()
            archive_path.write_bytes(response.content)
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(SHAPEFILES_DIR)
            self._copy_tamil_nadu_files()
        except Exception:
            return

    def _remote_download_enabled(self) -> bool:
        value = os.getenv("SMART_GIS_ENABLE_REMOTE_SHAPEFILES", "")
        return value.lower() in {"1", "true", "yes"}

    def _copy_tamil_nadu_files(self) -> None:
        extracted_root = next((path for path in SHAPEFILES_DIR.iterdir() if path.is_dir()), None)
        if extracted_root is None:
            return
        candidates = list(extracted_root.rglob("TAMIL NADU")) + list(extracted_root.rglob("TAMIL_NADU"))
        if not candidates:
            return
        source_dir = candidates[0]
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                destination = SHAPEFILES_DIR / file_path.name
                shutil.copy(file_path, destination)

    def _normalize_dataset(self, frame: gpd.GeoDataFrame, dataset_name: str) -> gpd.GeoDataFrame:
        gdf = frame.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(epsg=4326)
        if "category" not in gdf.columns:
            gdf["category"] = dataset_name
        if "name" not in gdf.columns:
            gdf["name"] = [f"{dataset_name.title()} {idx + 1}" for idx in range(len(gdf))]
        return gdf.dropna(subset=["geometry"]).reset_index(drop=True)

    def _ensure_sample_data(self) -> None:
        if any(DATA_SAMPLES_DIR.glob("*.geojson")):
            return
        for dataset_name in ("accidents", "incidents", "infrastructure"):
            self._build_demo_points(dataset_name).to_file(DATA_SAMPLES_DIR / f"{dataset_name}.geojson", driver="GeoJSON")

    def _build_demo_points(self, dataset_name: str) -> gpd.GeoDataFrame:
        coordinates = {
            "accidents": [
                (80.2707, 13.0827, "industrial"),
                (80.2740, 13.0781, "road"),
                (80.2812, 13.0690, "junction"),
                (80.2564, 13.0451, "road"),
                (76.9558, 11.0168, "junction"),
                (76.9481, 11.0044, "road"),
                (76.9655, 11.0235, "industrial"),
                (78.1198, 9.9252, "road"),
                (78.1321, 9.9311, "road"),
                (78.7047, 10.7905, "industrial"),
                (78.6924, 10.8014, "road"),
                (79.1271, 12.9249, "road"),
                (79.1382, 12.9171, "junction"),
                (78.1462, 11.6643, "road"),
                (78.1559, 11.6738, "road"),
            ],
            "incidents": [
                (80.2470, 13.0475, "public"),
                (80.2514, 13.0542, "public"),
                (77.5946, 11.0168, "service"),
                (78.1460, 9.9391, "public"),
            ],
            "infrastructure": [
                (80.2206, 13.0067, "hospital"),
                (78.1460, 11.6643, "school"),
                (79.8083, 11.9416, "industrial"),
            ],
        }
        rows = coordinates.get(dataset_name, coordinates["accidents"])
        return gpd.GeoDataFrame(
            [
                {"name": f"{dataset_name.title()} {idx + 1}", "category": category, "geometry": Point(lon, lat)}
                for idx, (lon, lat, category) in enumerate(rows)
            ],
            crs="EPSG:4326",
        )

    def _create_demo_districts(self) -> gpd.GeoDataFrame:
        districts = [
            {
                "district": "Chennai",
                "geometry": Polygon([(80.10, 12.95), (80.35, 12.95), (80.35, 13.20), (80.10, 13.20)]),
            },
            {
                "district": "Coimbatore",
                "geometry": Polygon([(76.85, 10.90), (77.10, 10.90), (77.10, 11.12), (76.85, 11.12)]),
            },
            {
                "district": "Madurai",
                "geometry": Polygon([(78.00, 9.80), (78.25, 9.80), (78.25, 10.05), (78.00, 10.05)]),
            },
            {
                "district": "Tiruchirappalli",
                "geometry": Polygon([(78.55, 10.68), (78.84, 10.68), (78.84, 10.92), (78.55, 10.92)]),
            },
            {
                "district": "Vellore",
                "geometry": Polygon([(79.00, 12.80), (79.25, 12.80), (79.25, 13.02), (79.00, 13.02)]),
            },
            {
                "district": "Salem",
                "geometry": Polygon([(78.05, 11.55), (78.25, 11.55), (78.25, 11.78), (78.05, 11.78)]),
            },
        ]
        return gpd.GeoDataFrame(districts, crs="EPSG:4326")
