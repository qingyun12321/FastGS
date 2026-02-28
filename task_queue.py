"""Single-worker task queue with request status tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any, Callable


@dataclass
class RequestRecord:
    status: str
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] = field(default_factory=dict)


class SingleWorkerTaskQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: deque[tuple[str, Any]] = deque()
        self._current_request_id: str | None = None
        self._records: dict[str, RequestRecord] = {}
        self._worker: threading.Thread | None = None
        self._running = False

    def start(self, handler: Callable[[str, Any], dict[str, Any] | None]) -> None:
        with self._cond:
            if self._worker and self._worker.is_alive():
                return
            self._running = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                args=(handler,),
                daemon=True,
                name="fastgs-run-queue-worker",
            )
            self._worker.start()

    def enqueue(
        self,
        payload: Any,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        rid = (request_id or "").strip() or str(uuid.uuid4())
        initial_meta = dict(metadata or {})

        with self._cond:
            if rid in self._records and self._records[rid].status in {"pending", "processing"}:
                raise ValueError("request_id already exists in queue")

            self._records[rid] = RequestRecord(status="pending", result=initial_meta)
            self._pending.append((rid, payload))
            position = self._pending_position_unlocked(rid)
            self._cond.notify()
            return rid, position

    def get_queue_status(self, request_id: str | None = None) -> dict[str, Any]:
        with self._cond:
            payload: dict[str, Any] = {
                "processing": self._current_request_id is not None,
                "pending": len(self._pending),
                "current_request_id": self._current_request_id or "",
            }

            if request_id is not None:
                rid = request_id.strip()
                payload["status"] = self._records[rid].status if rid in self._records else "unknown"
                payload["position"] = self._position_for_request_unlocked(rid)

            return payload

    def get_request_status(self, request_id: str) -> dict[str, Any] | None:
        rid = request_id.strip()
        with self._cond:
            record = self._records.get(rid)
            if not record:
                return None

            result = dict(record.result)
            payload: dict[str, Any] = {
                "request_id": rid,
                "status": record.status,
                "error": record.error,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "result": result,
            }
            for key, value in result.items():
                if key not in payload:
                    payload[key] = value
            return payload

    def _worker_loop(self, handler: Callable[[str, Any], dict[str, Any] | None]) -> None:
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait()
                if not self._running:
                    return

                request_id, payload = self._pending.popleft()
                self._current_request_id = request_id
                record = self._records[request_id]
                record.status = "processing"
                record.error = ""
                record.started_at = time.time()
                record.finished_at = None

            try:
                result = handler(request_id, payload) or {}
                with self._cond:
                    record = self._records[request_id]
                    record.status = "completed"
                    record.error = ""
                    record.finished_at = time.time()
                    if isinstance(result, dict):
                        merged = dict(record.result)
                        merged.update(result)
                        record.result = merged
            except Exception as exc:  # pragma: no cover - best-effort worker guard
                with self._cond:
                    record = self._records[request_id]
                    record.status = "failed"
                    record.error = str(exc)
                    record.finished_at = time.time()
            finally:
                with self._cond:
                    if self._current_request_id == request_id:
                        self._current_request_id = None

    def _position_for_request_unlocked(self, request_id: str) -> int:
        if not request_id:
            return -1
        if request_id == self._current_request_id:
            return 0
        return self._pending_position_unlocked(request_id)

    def _pending_position_unlocked(self, request_id: str) -> int:
        for index, (rid, _) in enumerate(self._pending):
            if rid == request_id:
                return index + 1
        return -1
