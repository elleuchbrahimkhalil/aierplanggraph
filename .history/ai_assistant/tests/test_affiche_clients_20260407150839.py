import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant.langgraph_skeleton import run_once


class _MockResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class TestAfficheClientsFlow(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        os.environ["ERP_ENDPOINTS_JSON"] = (
            "C:/Users/brahim/OneDrive/Bureau/example erp stage pfe/"
            "aierpjava/test/src/main/resources/endpoints.json"
        )
        os.environ["ERP_API_BASE_URL"] = "http://localhost:5006"
        os.environ["ERP_LOAD_SWAGGER_ENDPOINTS"] = "0"
        os.environ["USE_OLLAMA"] = "0"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    @patch("ai_assistant.langgraph_skeleton.requests.get")
    def test_affiche_les_clients_calls_endpoint_and_returns_data(self, mock_get):
        def _side_effect(url, headers=None, timeout=60):
            if url.endswith("/swagger/v1/swagger.json"):
                return _MockResponse(
                    200,
                    {
                        "paths": {
                            "/api/Client/GetAllClients": {"get": {}},
                            "/odata/ClientOdata": {"get": {}},
                        }
                    },
                )
            if url.startswith("http://localhost:5006/api/Client/GetAllClients"):
                return _MockResponse(
                    200,
                    [
                        {"code": "4110001", "rais_soc_clt": "Client Test 1"},
                        {"code": "4110002", "rais_soc_clt": "Client Test 2"},
                    ],
                )
            return _MockResponse(404, {"error": "not found"})

        mock_get.side_effect = _side_effect

        result = run_once("affiche les clients")

        self.assertEqual(result.get("selected_endpoint", {}).get("id"), "get_clients")
        self.assertEqual(result.get("errors", []), [])

        api_result_path = Path(result.get("api_result_path", ""))
        self.assertTrue(api_result_path.exists())

        payload = json.loads(api_result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("statusCode"), 200)
        self.assertEqual(payload.get("resolvedUrl"), "/api/Client/GetAllClients")
        self.assertEqual(len(payload.get("data", [])), 2)


if __name__ == "__main__":
    unittest.main()
