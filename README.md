# Smart GIS Chatbot

Smart GIS Chatbot is a hackathon-ready end-to-end system that converts natural language questions into GIS analysis, generates interactive Folium maps, and returns plain-language insights. The stack uses Streamlit for the chat UI, FastAPI for orchestration, LangChain for query parsing scaffolding, GeoPandas/Shapely/PySAL for analysis, and Folium for map rendering.

## Architecture

```text
User Query
   |
   v
Streamlit Frontend (`frontend/app.py`)
   |
   v
FastAPI Backend (`backend/main.py`)
   |
   v
LangChain Query Parser (`ai_module/query_parser.py`)
   |
   v
GIS Engine (`gis_engine/spatial_analysis.py`)
   |
   v
Folium Map Generator (`map_service/map_generator.py`)
   |
   v
JSON Response + Interactive Map
```

The backend and frontend communicate with JSON:

```json
{
  "question": "Where are accident hotspots in Tamil Nadu?"
}
```

Example parsed query:

```json
{
  "dataset": "accidents",
  "analysis": "heatmap",
  "filters": {
    "district": "Chennai"
  },
  "buffer_distance_m": 1000,
  "explanation_level": "simple"
}
```

## Repository Structure

```text
smart-gis-chatbot/
├── ai_module/
│   ├── __init__.py
│   └── query_parser.py
├── backend/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── Data Samples/
│   │   ├── accidents.geojson
│   │   ├── incidents.geojson
│   │   └── infrastructure.geojson
│   └── accidents.geojson
├── docs/
│   └── API_SPEC.md
├── frontend/
│   └── app.py
├── generated_maps/
├── gis_engine/
│   ├── __init__.py
│   └── spatial_analysis.py
├── map_service/
│   ├── __init__.py
│   └── map_generator.py
├── shapefiles/
│   └── TamilNadu/
├── Dockerfile
├── README.md
└── requirements.txt
```

## Features

- Accepts natural-language GIS questions.
- Parses intent into structured GIS operations.
- Loads Tamil Nadu administrative boundaries from local shapefiles, extracted zip data, or a GitHub archive fallback.
- Extracts `Data Samples.zip` automatically when present.
- Supports heatmap, buffer, district-wise spatial join, and hotspot cluster workflows.
- Generates interactive Folium HTML maps for the frontend.
- Returns plain-language insights plus machine-readable metadata.

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the backend

```bash
uvicorn backend.main:app --reload
```

Backend docs will be available at `http://localhost:8000/docs`.

### 3. Run the frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

The Streamlit app will call the backend at `http://localhost:8000` by default. To point it elsewhere:

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

## Data Setup

### Tamil Nadu boundaries

Primary source:

`https://github.com/datta07/INDIAN-SHAPEFILES/tree/master/STATES/TAMIL%20NADU`

Runtime behavior:

- If district shapefiles or GeoJSON already exist under `shapefiles/TamilNadu/`, they are used directly.
- If not, the backend attempts to download the GitHub repository zip and copy Tamil Nadu files into `shapefiles/TamilNadu/`.
- If that fails, demo district polygons are generated so the app still runs for hackathon demos.

### Provided `Data Samples.zip`

Runtime behavior:

- If `Data Samples.zip` exists in the repo root or `data/`, it is extracted automatically into `data/Data Samples/`.
- GeoJSON or shapefile layers found there are loaded into GeoPandas.
- If no zip is provided, bundled demo GeoJSON files are used so the pipeline remains runnable.

Recommended drop-in paths:

- `data/Data Samples.zip`
- `Data Samples.zip`

## API

### `POST /ask`

Request:

```json
{
  "question": "Where are accident hotspots near Chennai?"
}
```

Response:

```json
{
  "parsed_query": {
    "dataset": "accidents",
    "analysis": "buffer",
    "filters": {
      "district": "Chennai",
      "near": "Chennai"
    },
    "buffer_distance_m": 1000,
    "explanation_level": "simple"
  },
  "map_path": "generated_maps/abc123.html",
  "map_url": "/maps/abc123.html",
  "insight": "Generated 3 buffer zones with a radius of 1000 meters.",
  "metadata": {
    "buffer_distance_m": 1000
  }
}
```

See [API_SPEC.md](/home/karthik/smart-gis-chatbot/docs/API_SPEC.md) for the full contract.

## Team Git Workflow

For a 4-person hackathon team:

1. Clone the repository.
2. Create a feature branch per owner:
   - `frontend-ui`
   - `backend-api`
   - `ai-parser`
   - `gis-engine`
3. Make focused commits.
4. Push the branch.
5. Open a pull request into `main`.
6. Review, merge, and sync all branches daily.

Commands:

```bash
git clone <repo-url>
cd smart-gis-chatbot
git checkout -b frontend-ui
git add .
git commit -m "Build Streamlit chatbot UI"
git push origin frontend-ui
```

Recommended practice:

- Keep each module contract stable with JSON interfaces.
- Use PR reviews to protect shared files like `backend/main.py` and `README.md`.
- Rebase or merge `main` before opening the PR to reduce conflicts.

## Docker

Build:

```bash
docker build -t smart-gis-chatbot .
```

Run:

```bash
docker run -p 8000:8000 smart-gis-chatbot
```

If you want Streamlit containerized separately for demos, add a second frontend Docker service or use a Procfile-based platform split deployment.

## Cloud Deployment

### Render

Backend:

1. Create a new Web Service from the repository.
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add a persistent disk if you want generated maps or uploaded datasets retained.

Frontend:

1. Create a second service for Streamlit.
2. Set start command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
3. Set `BACKEND_URL` to the Render URL of the FastAPI service.

### Railway

1. Create a new project from GitHub.
2. Add environment variable `BACKEND_URL` for the Streamlit service if splitting services.
3. Use the backend start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

4. For Streamlit, deploy a separate service:

```bash
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
```

### AWS EC2

1. Launch an Ubuntu instance.
2. Install Docker or Python 3.11 + system GIS libraries.
3. Clone the repository and copy your datasets into `data/` and `shapefiles/`.
4. Open security-group ports for `8000` and optionally the Streamlit port.
5. Run either:

```bash
docker build -t smart-gis-chatbot .
docker run -d -p 8000:8000 smart-gis-chatbot
```

Or native services:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

For production, place Nginx in front of both services and use `systemd` units for process management.

## Extension Points

- Replace the rule-based fallback parser with a hosted LLM chain for richer intent extraction.
- Add temporal filters, routing, isochrone analysis, and raster support.
- Persist query history and uploaded datasets in PostgreSQL/PostGIS.
- Add authentication and per-team workspaces for multi-user hackathon demos.
- Move map artifacts to object storage and return signed URLs.

## Notes

- The project is modular by package so frontend, backend, AI parsing, and GIS analysis can be developed independently.
- Errors are surfaced through FastAPI with clear `400` and `500` responses.
- Demo data is bundled so the repository works immediately, even before real shapefiles or zip data are added.
