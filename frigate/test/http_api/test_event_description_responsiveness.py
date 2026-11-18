"""Keep description writes and embedding waits off the API event loop."""

import asyncio
import socket
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, suppress
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException, Request
from peewee import SqliteDatabase

from frigate.api.event import router, set_description
from frigate.comms.embeddings_updater import EmbeddingsRequestor
from frigate.embeddings import EmbeddingsContext
from frigate.models import Event


class _ReadyServer(uvicorn.Server):
    def __init__(self, app, ready):
        super().__init__(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="off")
        )
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self.ready.set()


class TestEventDescriptionResponsiveness(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "events.db"
        self.previous_database = Event._meta.database
        self.db = SqliteDatabase(self.path, timeout=5, pragmas={"journal_mode": "wal"})
        Event.bind(self.db)
        self.stack.callback(Event.bind, self.previous_database)
        self.stack.callback(self.db.close)
        self.db.create_tables([Event])
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
        self.context = Mock(spec=EmbeddingsContext)
        self.context.db = Mock()
        self.app = FastAPI()
        self.app.embeddings = self.context
        self.app.frigate_config = SimpleNamespace(
            auth=SimpleNamespace(roles={"admin": None, "viewer": None}),
            proxy=SimpleNamespace(separator=","),
            cameras={"synthetic-camera": None},
            semantic_search=SimpleNamespace(enabled=True),
        )
        self.app.include_router(router)

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

    def test_nonblank_description_is_saved_before_embedding(self):
        self.context.update_description.side_effect = lambda *args: self.assertEqual(
            self.stored_description(), {"description": "after", "unchanged": True}
        )
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "message": "Event synthetic-event description set to after",
            },
        )
        self.context.update_description.assert_called_once_with(
            "synthetic-event", "after"
        )

    def test_blank_description_cleans_embeddings_when_search_disabled(self):
        self.app.frigate_config.semantic_search.enabled = False
        response = self.post("")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stored_description()["description"], "")
        self.assertEqual(
            response.json()["message"], "Event synthetic-event description is now blank"
        )
        self.context.db.delete_embeddings_description.assert_called_once_with(
            event_ids=["synthetic-event"]
        )
        self.context.update_description.assert_not_called()

    def test_no_embedding_context_still_persists_description(self):
        self.app.embeddings = None
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.stored_description()["description"], "after")
        self.context.update_description.assert_not_called()

    def test_disabled_search_does_not_send_embedding_request(self):
        self.app.frigate_config.semantic_search.enabled = False
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.stored_description()["description"], "after")
        self.context.update_description.assert_not_called()

    def test_missing_event_keeps_404_without_embedding_request(self):
        response = self.post(event_id="synthetic-missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["success"], False)
        self.assertEqual(self.stored_description()["description"], "before")
        self.context.update_description.assert_not_called()

    def test_non_admin_cannot_change_description(self):
        response = self.post(headers={"remote-role": "viewer"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.stored_description()["description"], "before")
        self.context.update_description.assert_not_called()

    def test_camera_access_is_awaited_before_save(self):
        denial = AsyncMock(side_effect=HTTPException(status_code=403))
        with patch("frigate.api.event.require_camera_access", denial):
            response = self.post()
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.stored_description()["description"], "before")
        denial.assert_awaited_once()
        self.context.update_description.assert_not_called()

    def test_slow_event_lookup_does_not_block_other_requests(self):
        entered, release = threading.Event(), threading.Event()
        original = self.db.execute_sql

        def delayed_lookup(sql, *args, **kwargs):
            if sql.startswith("SELECT"):
                entered.set()
                if not release.wait(4):
                    raise TimeoutError("test lookup was not released")
            return original(sql, *args, **kwargs)

        with patch.object(self.db, "execute_sql", delayed_lookup):
            response = self.assert_responsive_while_blocked(entered, release)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stored_description()["description"], "after")

    def test_locked_sqlite_write_does_not_block_other_requests(self):
        entered, release = threading.Event(), threading.Event()
        original = self.db.execute_sql

        def signal_write(sql, *args, **kwargs):
            if sql.startswith("UPDATE"):
                entered.set()
                if not release.wait(4):
                    raise TimeoutError("test database write was not released")
            return original(sql, *args, **kwargs)

        with closing(sqlite3.connect(self.path)) as locker:
            locker.execute("BEGIN IMMEDIATE")
            self.assertTrue(locker.in_transaction)
            with patch.object(self.db, "execute_sql", signal_write):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(self.post)
                    try:
                        self.assertTrue(
                            entered.wait(3), "no native UPDATE was attempted"
                        )
                        self.assertEqual(
                            self.client.get(
                                "/responsiveness-probe", timeout=0.5
                            ).status_code,
                            200,
                        )
                        self.assertFalse(pending.done())
                    finally:
                        locker.rollback()
                        release.set()
                    self.assertEqual(pending.result(timeout=5).status_code, 200)
        self.assertEqual(self.stored_description()["description"], "after")

    def test_native_embedding_rpc_does_not_block_other_requests(self):
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
            # Keep the production context and requestor methods; only the peer is synthetic.
            context = object.__new__(EmbeddingsContext)
            context.requestor = requestor
            self.app.embeddings = context
            try:
                response = self.assert_responsive_while_blocked(entered, release)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.stored_description()["description"], "after")
            finally:
                requestor.stop()
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive(), "embedding peer did not stop")
        self.assertEqual(errors, [])
        self.assertEqual(
            messages,
            [["embed_description", {"id": "synthetic-event", "description": "after"}]],
        )

    def test_slow_blank_cleanup_does_not_block_other_requests(self):
        entered, release = threading.Event(), threading.Event()

        def cleanup(**kwargs):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test cleanup was not released")

        self.context.db.delete_embeddings_description.side_effect = cleanup
        response = self.assert_responsive_while_blocked(entered, release, "")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.stored_description()["description"], "")

    def test_embedding_exception_does_not_report_success(self):
        self.context.update_description.side_effect = ValueError("synthetic failure")
        self.assertEqual(self.post().status_code, 500)
        self.assertEqual(self.stored_description()["description"], "after")
        self.assertEqual(self.client.get("/responsiveness-probe").status_code, 200)

    def test_cancelled_wait_does_not_repeat_the_saved_work(self):
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def embedding(*args):
            entered.set()
            if not release.wait(4):
                raise TimeoutError("test cancelled embedding was not released")
            finished.set()

        self.context.update_description.side_effect = embedding
        scope = {
            "type": "http",
            "app": self.app,
            "headers": [
                (b"remote-user", b"synthetic-admin"),
                (b"remote-role", b"admin"),
            ],
        }

        async def invoke():
            from frigate.api.defs.request.events_body import EventsDescriptionBody

            task = asyncio.create_task(
                set_description(
                    Request(scope),
                    "synthetic-event",
                    EventsDescriptionBody(description="after"),
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                self.assertTrue(task.cancel())
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(
                    self.client.get("/responsiveness-probe", timeout=0.5).status_code,
                    200,
                )
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            self.assertTrue(await asyncio.to_thread(finished.wait, 3))

        asyncio.run(invoke())
        self.assertEqual(self.stored_description()["description"], "after")
        self.context.update_description.assert_called_once_with(
            "synthetic-event", "after"
        )
