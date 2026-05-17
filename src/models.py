from pydantic import BaseModel
from datetime import datetime


class IngredientRow(BaseModel):
    name: str
    estimated_grams: float
    confidence: float
    kcal: float
    protein: float
    carbs: float
    fat: float


class AnalysisRecord(BaseModel):
    id: int | None = None
    timestamp: datetime
    image_path: str
    ingredients: list[IngredientRow]
    total_kcal: float
    total_protein: float
    total_carbs: float
    total_fat: float
    meal_recognized: bool