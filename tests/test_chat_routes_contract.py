import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.routes import router  # noqa: E402


class ChatRoutesContractTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_generate_text_forwards_response_length(self) -> None:
        request_id = "req-route-123"

        with patch(
            "api.routes.generate_text",
            new=AsyncMock(
                return_value={
                    "status": "success",
                    "response": "ok",
                    "session_id": "sid-1",
                    "session_mode": "persistent",
                }
            ),
        ) as mock_generate:
            response = self.client.post(
                "/ai/generate-text",
                json={
                    "prompt": "Hello",
                    "model": "test-model",
                    "session_id": "sid-1",
                    "session_mode": "persistent",
                    "response_length": "long",
                },
                headers={"X-Request-ID": request_id},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["response"], "ok")
        self.assertEqual(payload["request_id"], request_id)
        self.assertEqual(response.headers.get("X-Request-ID"), request_id)

        mock_generate.assert_awaited_once_with(
            "Hello",
            model="test-model",
            session_id="sid-1",
            session_mode="persistent",
            response_length="long",
            request_id=request_id,
        )

    def test_generate_text_error_maps_to_http_500(self) -> None:
        with patch(
            "api.routes.generate_text",
            new=AsyncMock(return_value={"status": "error", "message": "boom"}),
        ):
            response = self.client.post(
                "/ai/generate-text",
                json={"prompt": "Hello"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json().get("detail"), "boom")

    def test_ask_route_brain_init_failure_maps_to_http_400(self) -> None:
        with patch("api.routes._use_docsum_proxy", return_value=False), patch(
            "api.routes._get_local_brain",
            side_effect=RuntimeError("brain init failed"),
        ):
            response = self.client.post(
                "/ai/ask",
                json={"paper_id": 123, "question": "What is this paper about?"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "brain init failed")

    def test_upload_route_brain_init_failure_maps_to_http_503(self) -> None:
        with patch("api.routes._use_docsum_proxy", return_value=False), patch(
            "api.routes._get_local_brain",
            side_effect=RuntimeError("NVIDIA_API_KEY is required for answer generation"),
        ):
            response = self.client.post(
                "/ai/upload",
                files={"file": ("paper.pdf", b"dummy pdf bytes", "application/pdf")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("NVIDIA_API_KEY", response.json().get("detail", ""))

    def test_stream_route_forwards_response_length_and_formats_events(self) -> None:
        request_id = "req-stream-789"

        with patch("api.routes.stream_text", return_value=iter(["Hello ", "world"])) as mock_stream:
            response = self.client.post(
                "/ai/generate-text/stream",
                json={
                    "prompt": "Hello",
                    "session_id": "sid-2",
                    "session_mode": "persistent",
                    "response_length": "long",
                },
                headers={"X-Request-ID": request_id},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        self.assertEqual(response.headers.get("X-Request-ID"), request_id)

        events = []
        for line in response.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[-1].get("type"), "done")
        self.assertEqual(events[-1].get("session_mode"), "persistent")
        self.assertEqual(events[-1].get("request_id"), request_id)

        combined_chunks = "".join(event.get("content", "") for event in events if event.get("type") == "chunk")
        self.assertEqual(combined_chunks, "Hello world")

        mock_stream.assert_called_once_with(
            "Hello",
            model=None,
            session_id="sid-2",
            session_mode="persistent",
            response_length="long",
            request_id=request_id,
        )


if __name__ == "__main__":
    unittest.main()
