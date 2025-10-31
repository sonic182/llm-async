from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..models.event import Event


class PubSubBackend(ABC):
    """Abstract base class for pub/sub backends."""

    @abstractmethod
    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[Event]:
        """Subscribe to events from a topic."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        pass
