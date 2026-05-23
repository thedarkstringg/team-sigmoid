# Architecture — AI Food Analyzer

## Overview

AI Food Analyzer wraps a provided `ai/` module with a software-engineering layer that adds configuration, an HTTP API, a CLI, async nutrition lookup, caching, SQLite persistence, logging, retries, validation, tests, and Docker support.

The main goal of the architecture is to keep entry points, orchestration, provider access, caching, and persistence separated into clear modules.

---

## Module Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                             │
│                                                                 │
│   ┌───────────────────┐          ┌───────────────────────────┐  │
│   │     src/cli.py    │          │       src/api.py          │  │
│   │ python -m src.cli │          │  FastAPI POST /analyze    │  │
│   │ analyze/history   │          │          GET  /health     │  │
│   │                   │          │          GET  /history    │  │
│   └────────┬──────────┘          └────────────┬──────────────┘  │
└────────────┼───────────────────────────────────┼────────────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            │ calls run_analysis(image_path)
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    src/core/analyzer.py                       │
│                                                               │
│  1. identify_ingredients(image_path)   ──► ai_service.py      │
│  2. fetch_nutrition_parallel(ingreds)  ──► pipeline.py        │
│  3. compute_totals(ingreds, facts)     ──► ai_service.py      │
│  4. build IngredientRow objects                               │
│  5. return AnalysisRecord                                     │
└──────┬──────────────────┬──────────────────────┬─────────────┘
       │                  │                      │
       ▼                  ▼                      ▼
┌─────────────┐  ┌─────────────────────┐  ┌────────────────────┐
│ src/services│  │ src/concurrency/    │  │ src/models.py      │
│ ai_service  │  │ pipeline.py         │  │                    │
│ .py         │  │                     │  │ AnalysisRecord     │
│ • retries   │  │ asyncio.gather      │  │ IngredientRow      │
│ • logging   │  │ asyncio.to_thread   │  │                    │
│ • timings   │  │ bounded semaphore   │  │                    │
└──────┬──────┘  └──────────┬──────────┘  └────────────────────┘
       │                    │
       │                    │ one nutrition lookup per ingredient
       ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│                   ai/   provided helper layer                 │
│                                                              │
│  ai.vlm                 ingredient identification             │
│  ai.calculator          nutrition total calculation           │
│  ai.nutrition           USDA nutrition adapter                │
│  ai.schemas             Ingredient, NutritionFacts, Nutrition │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│              External Providers / Integrations               │
│                                                              │
│   VLM providers: Anthropic / OpenAI / Google-style adapters  │
│   Nutrition provider: USDA-style nutrition lookup            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  src/storage/repository.py                   │
│                                                              │
│  AnalysisRepository saves and lists AnalysisRecord data       │
│  using SQLAlchemy async engine and SQLite via aiosqlite.      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  src/config.py  cross-cutting                 │
│                                                              │
│  pydantic-settings reads .env and environment variables       │
│  into one typed Settings object used by application modules.  │
└──────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module Summary

### `src/config.py`

`src/config.py` defines the central `Settings` object using `pydantic-settings`.

It covers:

- AI provider configuration
- API keys
- nutrition provider configuration
- database URL
- cache TTL
- concurrency limit
- server host and port
- logging level
- upload size limit

Application modules import settings from this file instead of scattering configuration logic across the project.

### `src/models.py`

`src/models.py` defines the main software-engineering layer models:

- `IngredientRow`
- `AnalysisRecord`

`IngredientRow` represents one ingredient with estimated grams and calculated nutrition values.

`AnalysisRecord` represents one complete analysis result, including image path, ingredient rows, total nutrition values, timestamp, and meal recognition status.

### `src/services/ai_service.py`

`src/services/ai_service.py` wraps AI-related calls with production concerns.

It handles:

- ingredient identification through the provided AI helper layer
- total nutrition calculation through the provided calculator layer
- retry behavior using `tenacity`
- structured logging
- timing information

This keeps AI-provider-related calls away from the API and CLI layers.

### `src/services/nutrition_cache.py`

`src/services/nutrition_cache.py` provides a TTL-aware in-memory cache keyed by ingredient name.

It supports:

- `get`
- `set`
- `invalidate`
- `clear`
- `size`

The cache avoids repeated nutrition lookups for the same ingredient during runtime. It is in-memory, so it is not persisted to SQLite.

### `src/core/analyzer.py`

`src/core/analyzer.py` is the orchestration layer.

It performs the main analysis flow:

1. Identify ingredients from the image.
2. Fetch nutrition facts for each ingredient.
3. Compute nutrition totals.
4. Build `IngredientRow` objects.
5. Return an `AnalysisRecord`.

It does not contain HTTP logic, CLI parsing, or direct database persistence. Saving the returned record is handled by the API or CLI layer.

If no ingredients are recognized, it returns a structured `AnalysisRecord` with `meal_recognized=False` instead of crashing.

### `src/concurrency/pipeline.py`

`src/concurrency/pipeline.py` performs nutrition lookups concurrently.

It uses:

- `asyncio.gather`
- `asyncio.to_thread`
- a semaphore controlled by `settings.nutrition_concurrency_limit`

The semaphore prevents unbounded concurrent provider calls.

If one ingredient lookup fails, the failure is logged and that ingredient is skipped. Successful lookups are still returned, so one provider failure does not crash the full batch.

### `src/storage/repository.py`

`src/storage/repository.py` contains the persistence layer.

It uses:

- SQLAlchemy async engine
- SQLite through `aiosqlite`
- `MealAnalysis` ORM model
- `AnalysisRepository`

The repository exposes methods such as:

- `init_db`
- `save`
- `list_all`

This keeps SQL and database-specific logic isolated from the API, CLI, and analyzer layers.

### `src/api.py`

`src/api.py` defines the FastAPI application.

Main endpoints:

- `GET /health`
- `POST /analyze`
- `GET /history`

The API validates uploaded image files before analysis. It checks:

- content type
- empty files
- file size
- image magic bytes

Then it calls `run_analysis`, saves the result through `AnalysisRepository`, and returns a JSON response.

### `src/cli.py`

`src/cli.py` defines an argparse-based command-line interface.

Main commands:

```bash
python -m src.cli analyze path/to/image.png
python -m src.cli history
```

The CLI allows local analysis and history viewing from the terminal. It uses the same analyzer and repository layers as the API.

---

## Data Flow Across Module Boundaries

Most cross-module data uses typed models. Nutrition lookup results are passed as `dict[str, NutritionFacts]` because each ingredient name maps to its nutrition facts.

```text
image_path: str
    │
    ▼ ai_service.py
list[Ingredient]              ai.schemas.Ingredient
    │
    ▼ pipeline.py
dict[str, NutritionFacts]     ai.schemas.NutritionFacts
    │
    ▼ ai_service.py
Nutrition                     ai.schemas.Nutrition
    │
    ▼ analyzer.py
AnalysisRecord                src.models.AnalysisRecord
    │
    ▼ api.py / cli.py
AnalysisRepository.save()
    │
    ▼ repository.py
MealAnalysis ORM row
```

The main `ai/` boundary is handled through `ai_service.py` for ingredient identification and total calculation. The nutrition pipeline also uses the provided nutrition adapter for ingredient nutrition facts.

---

## Key Design Decisions

### Why use `asyncio.to_thread`?

The nutrition provider call is treated as a blocking operation. `asyncio.to_thread` allows the project to run those blocking calls in worker threads without blocking the event loop.

This lets multiple ingredient lookups run concurrently while keeping the rest of the application responsive.

### Why use SQLite?

SQLite is simple to run locally and does not require extra infrastructure for development or grading.

The repository pattern keeps database access isolated. A future switch to another SQL database would mainly require changing `DATABASE_URL` and database configuration rather than rewriting API or CLI logic.

### Why keep API, CLI, analyzer, and repository separate?

Separation makes the system easier to test and maintain.

- API handles HTTP validation and responses.
- CLI handles terminal arguments and output.
- Analyzer handles meal-analysis orchestration.
- Pipeline handles concurrent nutrition lookup.
- Repository handles persistence.
- Config handles environment settings.

This prevents one layer from becoming responsible for everything.

### Why use fake or mocked data in tests?

Many provider integrations require network access or API keys. Offline tests use fake or mocked data so the test suite can run reliably without secrets or external services.

---

## Testing Strategy

The project includes tests for:

- CLI behavior
- API validation and endpoints
- storage repository
- nutrition cache
- concurrency failure degradation
- failure injection
- Dockerfile structure
- logging setup
- coverage reporting

Most tests run offline and do not require real API keys.

---

## Docker Runtime

The Dockerfile builds a container for the FastAPI app.

The container starts the API with Uvicorn on port `8000`.

```bash
docker build -t ai-food-analyzer .
docker run --env-file .env -p 8000:8000 ai-food-analyzer
```

---

## Notes

- Real API keys must not be committed.
- Offline tests do not require real USDA or AI provider keys.
- Provider integrations may require keys when used for real external calls.
- The nutrition cache is in-memory and runtime-only.
- SQLite persistence stores analysis history.
