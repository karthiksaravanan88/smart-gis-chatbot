# Smart GIS Chatbot API Specification

## Base URL

Local:

```text
http://localhost:8000
```

## Endpoints

### `GET /health`

Health probe for local development and cloud deployment.

Response:

```json
{
  "status": "ok"
}
```

### `POST /ask`

Runs the end-to-end GIS chatbot workflow.

Request:

```json
{
  "question": "Where are accident hotspots in Tamil Nadu?"
}
```

Response:

```json
{
  "parsed_query": {
    "dataset": "accidents",
    "analysis": "cluster",
    "filters": {},
    "buffer_distance_m": 1000,
    "explanation_level": "simple"
  },
  "map_path": "generated_maps/8fa0df4f08b44c78aaefe729dc2b27d6.html",
  "map_url": "/maps/8fa0df4f08b44c78aaefe729dc2b27d6.html",
  "insight": "Detected 2 statistically significant hotspot points using local spatial clustering.",
  "metadata": {
    "hotspots": 2
  }
}
```

## Error Handling

- `400`: invalid question or no records after filtering
- `500`: unexpected GIS or map-generation failure

Example:

```json
{
  "detail": "No records matched the requested filters."
}
```
