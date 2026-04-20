import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.routes import router  # noqa: E402


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("utf-8")


class ExecuteOutputDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_execute_decodes_base64_stdout_and_stderr(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": _b64("4\n"),
            "stderr": "",
            "compile_output": "",
            "message": "",
            "status": {"id": 3},
            "exit_code": 0,
            "time": "0.02",
            "memory": 1024,
        }

        with patch("api.routes.requests.post", return_value=mock_response):
            response = self.client.post(
                "/ai/execute",
                json={
                    "language_id": 71,
                    "source_code": _b64("print(2 + 2)"),
                    "stdin": "",
                    "cpu_time_limit": 10,
                    "memory_limit": 128000,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "judge0")
        self.assertEqual(payload["stdout"], "4\n")
        self.assertEqual(payload["stderr"], "")
        self.assertEqual(payload["exitCode"], 0)

    def test_execute_preserves_plain_text_stdout_when_not_base64(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": "plain output",
            "stderr": "",
            "compile_output": "",
            "message": "",
            "status": {"id": 3},
            "exit_code": 0,
            "time": "0.01",
        }

        with patch("api.routes.requests.post", return_value=mock_response):
            response = self.client.post(
                "/ai/execute",
                json={
                    "language_id": 63,
                    "source_code": _b64("console.log('ok')"),
                    "stdin": "",
                    "cpu_time_limit": 10,
                    "memory_limit": 128000,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stdout"], "plain output")
        self.assertEqual(payload["stderr"], "")

    def test_execute_encodes_plain_stdin_for_judge0_base64_mode(self) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stdout": _b64("12\n"),
            "stderr": "",
            "compile_output": "",
            "message": "",
            "status": {"id": 3},
            "exit_code": 0,
            "time": "0.01",
        }

        with patch("api.routes.requests.post", return_value=mock_response) as mock_post:
            response = self.client.post(
                "/ai/execute",
                json={
                    "language_id": 62,
                    "source_code": _b64(
                        "import java.util.*;\n"
                        "public class Main {\n"
                        "  public static void main(String[] args) {\n"
                        "    Scanner sc = new Scanner(System.in);\n"
                        "    int a = sc.nextInt();\n"
                        "    int b = sc.nextInt();\n"
                        "    System.out.println(a + b);\n"
                        "  }\n"
                        "}"
                    ),
                    "stdin": "5 7",
                    "cpu_time_limit": 10,
                    "memory_limit": 128000,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stdout"], "12\n")
        posted_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(posted_payload["stdin"], _b64("5 7"))


if __name__ == "__main__":
    unittest.main()
