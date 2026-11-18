"""Keep description writes and embedding waits off the API event loop."""

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request

from frigate.api.event import set_description
from frigate.test.http_api.description_api_fixture import DescriptionApiTestCase


class TestEventDescriptionResponsiveness(DescriptionApiTestCase):
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
        with self.native_rpc_context() as (entered, release, messages):
            response = self.assert_responsive_while_blocked(entered, release)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.stored_description()["description"], "after")
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
