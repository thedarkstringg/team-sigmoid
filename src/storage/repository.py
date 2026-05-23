import json
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Integer, String, Float, Boolean, DateTime, Text, select

from src.config import settings
from src.models import AnalysisRecord, IngredientRow
from src.storage.base import AbstractAnalysisRepository

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class MealAnalysis(Base):
    __tablename__ = "meal_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    image_path: Mapped[str] = mapped_column(String(512))
    ingredients_json: Mapped[str] = mapped_column(Text)
    total_kcal: Mapped[float] = mapped_column(Float)
    total_protein: Mapped[float] = mapped_column(Float)
    total_carbs: Mapped[float] = mapped_column(Float)
    total_fat: Mapped[float] = mapped_column(Float)
    meal_recognized: Mapped[bool] = mapped_column(Boolean)


class AnalysisRepository(AbstractAnalysisRepository):
    def __init__(self) -> None:
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database.initialized")

    async def save(self, record: AnalysisRecord) -> AnalysisRecord:
        async with self._session_factory() as session:
            row = MealAnalysis(
                timestamp=datetime.now(timezone.utc),
                image_path=record.image_path,
                ingredients_json=json.dumps(
                    [i.model_dump() for i in record.ingredients]
                ),
                total_kcal=record.total_kcal,
                total_protein=record.total_protein,
                total_carbs=record.total_carbs,
                total_fat=record.total_fat,
                meal_recognized=record.meal_recognized,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            logger.info("storage.saved", extra={"id": row.id, "image": record.image_path})
            record.id = row.id
            return record

    async def list_all(self, limit: int = 50) -> list[AnalysisRecord]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(MealAnalysis)
                .order_by(MealAnalysis.timestamp.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
            return [self._to_model(r) for r in rows]

    def _to_model(self, row: MealAnalysis) -> AnalysisRecord:
        return AnalysisRecord(
            id=row.id,
            timestamp=row.timestamp,
            image_path=row.image_path,
            ingredients=[
                IngredientRow(**i)
                for i in json.loads(row.ingredients_json)
            ],
            total_kcal=row.total_kcal,
            total_protein=row.total_protein,
            total_carbs=row.total_carbs,
            total_fat=row.total_fat,
            meal_recognized=row.meal_recognized,
        )