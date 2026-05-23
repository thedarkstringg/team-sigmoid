# Project Name

Analysis of food images using AI.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: `cp .env.example .env`
3. Run CLI: `python -m foodanalyzer analyze`
4. Run API: `uvicorn src.api:app`

## Benchmark

Offline benchmark comparing sequential nutrition lookup with the concurrent pipeline.

Command: `python scripts/bench.py`

Result from local run:

| mode | items | duration_seconds |
|---|---:|---:|
| sequential | 12 | 0.630 |
| concurrent | 12 | 0.110 |

Speedup: **5.71x**

The benchmark uses a fake USDA provider, so it runs offline and does not require API keys.

<!-- coverage-report:start -->
## Test coverage

Latest coverage command:

python -m pytest --cov=src --cov-report=term -q

Latest src/ coverage result:

TOTAL    302    55    82%

Coverage is measured against src/ because ai/ contains external provider adapters that are not safe to exercise with real network/API calls in offline tests.

<!-- coverage-report:end -->
