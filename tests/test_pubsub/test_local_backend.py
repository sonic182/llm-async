import asyncio
from datetime import datetime

import pytest

from llm_async.models.event import Event
from llm_async.pubsub import LocalQueueBackend, PubSub


@pytest.mark.asyncio
async def test_local_backend_publish_and_subscribe():
    """Test basic publish and subscribe functionality."""
    backend = LocalQueueBackend()

    # Subscribe and receive the event
    received_events = []

    async def collect_events():
        async for event in backend.subscribe("tools.test.start"):
            received_events.append(event)
            if len(received_events) >= 1:
                break

    task = asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)  # Give time for subscription

    # Publish an event
    event = Event(
        topic="tools.test.start",
        payload={"tool_name": "calculator", "status": "started"},
        timestamp=datetime.now(),
    )
    await backend.publish("tools.test.start", event)

    await asyncio.wait_for(task, timeout=1.0)
    await backend.close()

    assert len(received_events) == 1
    assert received_events[0].payload["tool_name"] == "calculator"


@pytest.mark.asyncio
async def test_local_backend_wildcard_subscription():
    """Test wildcard subscription support."""
    backend = LocalQueueBackend()

    received_events = []

    async def collect_events():
        async for event in backend.subscribe("tools.test.*"):
            received_events.append(event)
            if len(received_events) >= 2:
                break

    task = asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Publish to different topics matching the pattern
    event1 = Event(
        topic="tools.test.start", payload={"status": "started"}, timestamp=datetime.now()
    )
    event2 = Event(
        topic="tools.test.complete", payload={"status": "completed"}, timestamp=datetime.now()
    )

    await backend.publish("tools.test.start", event1)
    await asyncio.sleep(0.05)
    await backend.publish("tools.test.complete", event2)

    await asyncio.wait_for(task, timeout=1.0)
    await backend.close()

    assert len(received_events) == 2
    assert received_events[0].topic == "tools.test.start"
    assert received_events[1].topic == "tools.test.complete"


@pytest.mark.asyncio
async def test_local_backend_multiple_subscribers():
    """Test multiple subscribers receiving the same event."""
    backend = LocalQueueBackend()

    received_events_1 = []
    received_events_2 = []

    async def collect_events_1():
        async for event in backend.subscribe("tools.test.*"):
            received_events_1.append(event)
            if len(received_events_1) >= 1:
                break

    async def collect_events_2():
        async for event in backend.subscribe("tools.test.*"):
            received_events_2.append(event)
            if len(received_events_2) >= 1:
                break

    task1 = asyncio.create_task(collect_events_1())
    task2 = asyncio.create_task(collect_events_2())
    await asyncio.sleep(0.1)

    event = Event(
        topic="tools.test.start", payload={"tool_name": "calculator"}, timestamp=datetime.now()
    )
    await backend.publish("tools.test.start", event)

    await asyncio.wait_for(asyncio.gather(task1, task2), timeout=1.0)
    await backend.close()

    assert len(received_events_1) == 1
    assert len(received_events_2) == 1
    assert received_events_1[0].payload == received_events_2[0].payload


@pytest.mark.asyncio
async def test_pubsub_interface():
    """Test PubSub interface."""
    backend = LocalQueueBackend()
    pubsub = PubSub(backend)

    received_events = []

    async def collect_events():
        async for event in pubsub.subscribe("tools.calculator.complete"):
            received_events.append(event)
            if len(received_events) >= 1:
                break

    task = asyncio.create_task(collect_events())
    await asyncio.sleep(0.1)

    # Use PubSub interface to publish
    await pubsub.publish("tools.calculator.complete", {"result": 42})

    await asyncio.wait_for(task, timeout=1.0)
    await pubsub.close()

    assert len(received_events) == 1
    assert received_events[0].payload["result"] == 42
    assert received_events[0].topic == "tools.calculator.complete"


@pytest.mark.asyncio
async def test_local_backend_closed_state():
    """Test that backend raises error when closed."""
    backend = LocalQueueBackend()
    await backend.close()

    event = Event(
        topic="tools.test.start", payload={"status": "started"}, timestamp=datetime.now()
    )

    with pytest.raises(RuntimeError, match="Backend is closed"):
        await backend.publish("tools.test.start", event)

    with pytest.raises(RuntimeError, match="Backend is closed"):
        async for _ in backend.subscribe("tools.test.*"):
            pass
