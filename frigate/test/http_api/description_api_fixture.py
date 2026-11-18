"""Shared real HTTP/SQLite fixtures for description API regressions."""

import socket
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import uvicorn
import zmq
from fastapi import FastAPI
from peewee import SqliteDatabase

from frigate.api.event import router as event_router
from frigate.api.review import router as review_router
from frigate.comms.embeddings_updater import EmbeddingsRequestor
from frigate.embeddings import EmbeddingsContext
from frigate.models import Event, ReviewSegment


class _ReadyServer(uvicorn.Server):
    def __init__(self, app, ready):
        super().__init__(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="off")
        )
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self.ready.set()


class DescriptionApiTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "events.db"
        self.previous_database = Event._meta.database
        self.db = SqliteDatabase(self.path, timeout=5, pragmas={"journal_mode": "wal"})
        previous_review_database = ReviewSegment._meta.database
        Event.bind(self.db)
        ReviewSegment.bind(self.db)
        self.stack.callback(ReviewSegment.bind, previous_review_database)
        self.stack.callback(Event.bind, self.previous_database)
        self.stack.callback(self.db.close)
        self.db.create_tables([Event, ReviewSegment])
        self.seed_rows()
        self.context = Mock(spec=EmbeddingsContext)
        self.context.db = Mock()
        self.app = FastAPI()
        self.app.embeddings = self.context
        self.app.frigate_config = SimpleNamespace(
            auth=SimpleNamespace(roles={"admin": None, "viewer": None}),
            proxy=SimpleNamespace(separator=","),
            cameras={
                "synthetic-camera": SimpleNamespace(
                    review=SimpleNamespace(genai=SimpleNamespace(enabled=True))
                )
            },
            semantic_search=SimpleNamespace(enabled=True),
        )
        self.app.genai_manager = SimpleNamespace(description_client=object())
        self.app.include_router(event_router)
        self.app.include_router(review_router)

        @self.app.get("/responsiveness-probe")
        async def probe():
            return {"responsive": True}

        listener = self.stack.enter_context(socket.socket())
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        self.url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        ready = threading.Event()
        self.server = _ReadyServer(self.app, ready)
        self.server_thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [listener]}
        )
        self.server_thread.start()
        self.stack.callback(self.stop_server)
        self.assertTrue(ready.wait(5), "test HTTP server did not start")
        self.client = self.stack.enter_context(
            httpx.Client(
                base_url=self.url,
                headers={"remote-user": "synthetic-admin", "remote-role": "admin"},
                timeout=5,
                trust_env=False,
            )
        )

    def seed_rows(self):
        Event.create(
            id="synthetic-event",
            label="person",
            camera="synthetic-camera",
            start_time=datetime.now(),
            end_time=datetime.now(),
            top_score=0.5,
            score=0.5,
            false_positive=False,
            zones=[],
            thumbnail="",
            region=[],
            box=[],
            area=0,
            plus_id="",
            model_hash="",
            detector_type="synthetic",
            model_type="synthetic",
            data={"description": "before", "unchanged": True},
        )
        ReviewSegment.create(
            id="synthetic-review",
            camera="synthetic-camera",
            start_time=datetime.now(),
            end_time=datetime.now(),
            severity="alert",
            thumb_path="synthetic-thumb",
            data={},
        )

    def stop_server(self):
        self.server.should_exit = True
        self.server_thread.join(10)
        self.assertFalse(self.server_thread.is_alive(), "test server did not stop")

    def post(self, description="after", event_id="synthetic-event", **kwargs):
        return self.client.post(
            f"/events/{event_id}/description",
            json={"description": description},
            **kwargs,
        )

    def stored_description(self):
        return Event.get_by_id("synthetic-event").data

    def assert_responsive_while_blocked(self, entered, release, description="after"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.post, description)
            try:
                self.assertTrue(entered.wait(3), "description did not enter blocked IO")
                response = self.client.get("/responsiveness-probe", timeout=0.5)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"responsive": True})
                self.assertFalse(
                    pending.done(), "description returned before IO finished"
                )
            finally:
                release.set()
            return pending.result(timeout=5)

    @contextmanager
    def native_rpc_context(self):
        entered, release, ready = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        endpoint, errors, messages = [], [], []

        def peer():
            with zmq.Context() as context:
                with context.socket(zmq.REP) as responder:
                    responder.rcvtimeo = 4000
                    port = responder.bind_to_random_port("tcp://127.0.0.1")
                    endpoint.append(f"tcp://127.0.0.1:{port}")
                    ready.set()
                    try:
                        messages.append(responder.recv_json())
                        entered.set()
                        if not release.wait(4):
                            raise TimeoutError(
                                "test embedding response was not released"
                            )
                        responder.send_json([])
                    except (zmq.ZMQError, TimeoutError) as error:
                        errors.append(error)

        thread = threading.Thread(target=peer)
        thread.start()
        try:
            self.assertTrue(ready.wait(3), "embedding peer did not start")
            with patch("frigate.comms.embeddings_updater.SOCKET_REP_REQ", endpoint[0]):
                requestor = EmbeddingsRequestor()
            context = object.__new__(EmbeddingsContext)
            context.requestor = requestor
            self.app.embeddings = context
            try:
                yield entered, release, messages
            finally:
                requestor.stop()
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive(), "embedding peer did not stop")
        self.assertEqual(errors, [])
