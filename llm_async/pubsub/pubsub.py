from collections.abc import AsyncIterator
from datetime import datetime

from ..models.event import Event
from .base import PubSubBackend


class PubSub:
    """Main pub/sub interface."""

    def __init__(self, backend: PubSubBackend):
        self._backend = backend

    async def publish(self, topic: str, payload: dict) -> None:
        """Publish an event to a topic."""
        event = Event(topic=topic, payload=payload, timestamp=datetime.now())
        await self._backend.publish(topic, event)

    async def subscribe(self, topic: str) -> AsyncIterator[Event]:
        """Subscribe to events from a topic."""
        async for event in self._backend.subscribe(topic):
            yield event

    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        await self._backend.close()
