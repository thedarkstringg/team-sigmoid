import abc
from src.models import AnalysisRecord


class AbstractAnalysisRepository(abc.ABC):
    """Abstract base for analysis storage backends."""

    @abc.abstractmethod
    async def init_db(self) -> None:
        """Initialize the storage backend."""
        raise NotImplementedError

    @abc.abstractmethod
    async def save(self, record: AnalysisRecord) -> AnalysisRecord:
        """Persist an analysis record and return it with an assigned id."""
        raise NotImplementedError

    @abc.abstractmethod
    async def list_all(self, limit: int = 50) -> list[AnalysisRecord]:
        """Return the most recent analysis records."""
        raise NotImplementedError