import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

from ..models.event import Event
from .base import PubSubBackend


class LocalQueueBackend(PubSubBackend):
    """Local in-memory pub/sub backend using asyncio queues."""

    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._closed = False

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""
        if self._closed:
            raise RuntimeError("Backend is closed")

        # Create event with timestamp if not provided
        if not hasattr(event, "timestamp") or event.timestamp is None:
            event.timestamp = datetime.now()

        # Find matching queues (support wildcards)
        for pattern, queues in self._queues.items():
            if self._match_topic(topic, pattern):
                for queue in queues:
                    await queue.put(event)

    async def subscribe(self, topic: str) -> AsyncIterator[Event]:
        """Subscribe to events from a topic."""
        if self._closed:
            raise RuntimeError("Backend is closed")

        queue = asyncio.Queue()
        if topic not in self._queues:
            self._queues[topic] = []
        self._queues[topic].append(queue)

        try:
            while not self._closed:
                event = await queue.get()
                yield event
        finally:
            # Clean up queue on exit
            if topic in self._queues and queue in self._queues[topic]:
                self._queues[topic].remove(queue)
                if not self._queues[topic]:
                    del self._queues[topic]

    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        self._closed = True
        self._queues.clear()

    def _match_topic(self, topic: str, pattern: str) -> bool:
        """Simple wildcard matching for topics."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return topic.startswith(prefix)
        return topic == pattern
