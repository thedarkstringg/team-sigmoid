# AI Food Analyzer

AI Food Analyzer is a Python project for analyzing meal images, recognizing likely ingredients, looking up nutrition facts, and saving analysis history. It includes a FastAPI HTTP API, a command-line interface, an async nutrition lookup pipeline, SQLite persistence, caching, tests, and Docker support.

## Features

- Image-based food analysis for JPEG and PNG meal images
- Ingredient recognition from uploaded or local images
- Nutrition lookup for recognized ingredients
- Nutrition cache to avoid repeated lookups for the same food data
- FastAPI HTTP API for health checks, analysis, and history
- CLI usage for local image analysis and history viewing
- Async/concurrent nutrition pipeline for faster multi-ingredient lookup
- SQLite persistence through `DATABASE_URL`
- Docker support for containerized runs

## Project Structure

```text
.
|-- ai/                    # AI provider adapters, schemas, and nutrition helpers
|-- data/                  # Sample food images for local testing
|-- scripts/               # Utility scripts and manual checks
|   |-- bench.py           # Offline concurrency benchmark
|   `-- test_cache.py      # Manual nutrition cache check
|-- src/                   # Main application code
|   |-- api.py             # FastAPI application and HTTP routes
|   |-- cli.py             # Command-line interface
|   |-- config.py          # Environment-based settings
|   |-- concurrency/       # Async nutrition pipeline
|   |-- core/              # Analysis orchestration
|   |-- services/          # AI service and nutrition cache
|   `-- storage/           # SQLite repository and persistence layer
|-- tests/                 # Automated test suite
|-- Dockerfile             # Docker image definition
|-- requirements.txt       # Python dependencies
`-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

`DATABASE_URL` controls where analysis history is stored. The nutrition cache is an in-memory TTL cache used during runtime. The default SQLite value is:

```env
DATABASE_URL=sqlite+aiosqlite:///./foodanalyzer.db
```

`USDA_API_KEY` is used by nutrition provider integrations when real USDA lookup is needed. It is optional for some offline tests and scripts, including the manual cache check, because those paths use fake or mocked data instead of making real API calls.

## How to Run the API

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000` by default.

## API Endpoints

- `GET /health` - returns a basic service status response.
- `POST /analyze` - accepts a JPEG or PNG image upload and returns an analysis record with recognized ingredients and nutrition totals.
- `GET /history` - returns recent saved analysis records.

Example analysis request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "image=@data/rice_chicken.png"
```

## How to Run CLI

Analyze an image:

```bash
python -m src.cli analyze path/to/image.png
```

Show saved analysis history:

```bash
python -m src.cli history
```

## Manual Cache Check

Run the manual nutrition cache script:

```bash
python scripts/test_cache.py
```

The script prompts for `DATABASE_URL` and an optional `USDA_API_KEY`. No real USDA API call is required for this script; it inserts and reads fake nutrition data to confirm that the cache works.

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Run tests with coverage:

```bash
python -m pytest --cov=src --cov-report=term -q
```

## Coverage Report

Latest coverage command:

```bash
python -m pytest --cov=src --cov-report=term -q
```

Latest `src/` coverage result:

```text
TOTAL    302    55    82%
```

Coverage is measured against `src/` because `ai/` contains external provider adapters and should not require real network/API calls during offline tests.

## Benchmark

Offline benchmark comparing sequential nutrition lookup with the concurrent pipeline:

```bash
python scripts/bench.py
```

Result from local run:

| mode | items | duration_seconds |
|---|---:|---:|
| sequential | 12 | 0.630 |
| concurrent | 12 | 0.110 |

Speedup: **5.71x**

The benchmark uses a fake USDA provider, so it runs offline and does not require API keys.

## Docker

Build the image:

```bash
docker build -t ai-food-analyzer .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 ai-food-analyzer
```

## Notes / Limitations

- Do not commit real API keys.
- Some provider integrations require API keys.
- Offline tests use fake or mocked data.

## Type checking

Run with:

```bash
mypy src/ --ignore-missing-imports
```

Result: 2 known false positives in `ai_service.py` suppressed with `# type: ignore[call-overload]` — tenacity's `@retry(**dict)` pattern is not resolvable by mypy's overload matching. All other 17 source files pass cleanly.
