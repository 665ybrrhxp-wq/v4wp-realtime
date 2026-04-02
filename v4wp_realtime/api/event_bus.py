"""In-memory pub/sub 이벤트 버스 (SSE 브로드캐스트용)

사용법:
    # 이벤트 발행 (store.py에서 호출)
    from v4wp_realtime.api.event_bus import publish
    publish("scores_updated", {"tickers": ["NVDA", "TSLA"], "date": "2026-03-24"})

    # SSE 엔드포인트에서 구독
    from v4wp_realtime.api.event_bus import subscribe, unsubscribe
    queue = subscribe()
    try:
        event = await asyncio.wait_for(queue.get(), timeout=30)
    finally:
        unsubscribe(queue)
"""
import asyncio
import json
import threading
import time
from typing import Any

# 구독자 목록: set of asyncio.Queue
_subscribers: set[asyncio.Queue] = set()
_lock = threading.Lock()


def subscribe() -> asyncio.Queue:
    """새 구독자 등록. asyncio.Queue 반환."""
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    with _lock:
        _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """구독 해제."""
    with _lock:
        _subscribers.discard(q)


def publish(event_type: str, data: Any = None) -> int:
    """모든 구독자에게 이벤트 발행.

    동기 함수에서 호출 가능 (asyncio loop 유무 자동 감지).
    Returns: 발행된 구독자 수.
    """
    payload = {
        "type": event_type,
        "data": data,
        "ts": time.time(),
    }

    delivered = 0
    dead = []

    with _lock:
        for q in list(_subscribers):
            try:
                q.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                    delivered += 1
                except Exception:
                    dead.append(q)
            except Exception:
                dead.append(q)

        for q in dead:
            _subscribers.discard(q)

    return delivered


def subscriber_count() -> int:
    """현재 구독자 수."""
    with _lock:
        return len(_subscribers)
