"""Keep review regeneration lookup and embedding waits off the API event loop."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from frigate.test.http_api.description_api_fixture import DescriptionApiTestCase


class TestReviewDescriptionResponsiveness(DescriptionApiTestCase):
    def post(self, description=None, review_id="synthetic-review", **kwargs):
        return self.client.put(f"/review/{review_id}/regenerate_description", **kwargs)

    def test_existing_review_keeps_accepted_response(self):
        response = self.post()
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "message": "Review synthetic-review description generation has been requested",
            },
        )
        self.context.regenerate_review_description.assert_called_once_with(
            "synthetic-review"
        )

    def test_missing_review_keeps_404(self):
        response = self.post(review_id="synthetic-missing")
        self.assertEqual(response.status_code, 404)
        self.context.regenerate_review_description.assert_not_called()

    def test_non_admin_cannot_regenerate(self):
        self.assertEqual(self.post(headers={"remote-role": "viewer"}).status_code, 403)
        self.context.regenerate_review_description.assert_not_called()

    def test_camera_access_is_awaited_before_regeneration(self):
        denial = AsyncMock(side_effect=HTTPException(status_code=403))
        with patch("frigate.api.review.require_camera_access", denial):
            self.assertEqual(self.post().status_code, 403)
        denial.assert_awaited_once()
        self.context.regenerate_review_description.assert_not_called()

    def test_ongoing_review_keeps_400(self):
        row = SimpleNamespace(camera="synthetic-camera", end_time=None)
        with patch("frigate.api.review.ReviewSegment.get", return_value=row):
            self.assertEqual(self.post().status_code, 400)
        self.context.regenerate_review_description.assert_not_called()

    def test_disabled_camera_keeps_400(self):
        self.app.frigate_config.cameras["synthetic-camera"].review.genai.enabled = False
        self.assertEqual(self.post().status_code, 400)
        self.context.regenerate_review_description.assert_not_called()

    def test_missing_camera_keeps_400(self):
        self.app.frigate_config.cameras.clear()
        with patch("frigate.api.review.require_camera_access", AsyncMock()):
            self.assertEqual(self.post().status_code, 400)
        self.context.regenerate_review_description.assert_not_called()

    def test_missing_provider_keeps_400(self):
        self.app.genai_manager.description_client = None
        self.assertEqual(self.post().status_code, 400)
        self.context.regenerate_review_description.assert_not_called()

    def test_embedding_exception_does_not_report_acceptance(self):
        self.context.regenerate_review_description.side_effect = ValueError("synthetic")
        self.assertEqual(self.post().status_code, 500)
        self.assertEqual(self.client.get("/responsiveness-probe").status_code, 200)

    def test_slow_review_lookup_does_not_block_other_requests(self):
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
        self.assertEqual(response.status_code, 202)

    def test_native_review_rpc_does_not_block_other_requests(self):
        with self.native_rpc_context() as (entered, release, messages):
            response = self.assert_responsive_while_blocked(entered, release)
            self.assertEqual(response.status_code, 202)
        self.assertEqual(
            messages,
            [["regenerate_review_description", {"review_id": "synthetic-review"}]],
        )
