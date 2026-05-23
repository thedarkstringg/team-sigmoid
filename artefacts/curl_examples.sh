#!/bin/bash
# Start server first: uvicorn src.api:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Analyze a meal image
curl -X POST http://localhost:8000/analyze \
  -F "image=@data/rice_chicken_broccoli.png"

# Analyze unknown meal (graceful degradation)
curl -X POST http://localhost:8000/analyze \
  -F "image=@data/no_meal_blue.png"

# View history
curl http://localhost:8000/history