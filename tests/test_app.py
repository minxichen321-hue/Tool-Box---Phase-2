import unittest
from unittest.mock import call, patch

from fastapi.testclient import TestClient
from fastmcp import Client

from app import app, mcp


class HttpAppTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_mcp_endpoint_starts_without_server_error(self) -> None:
        with TestClient(app) as client:
            response = client.get("/mcp/")

        self.assertNotEqual(response.status_code, 500)


class McpContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_advertises_required_tools_and_aliases(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        self.assertEqual(
            [tool.name for tool in tools],
            [
                "recall_study_passages",
                "search",
                "retrieve",
                "next_route_node",
            ],
        )
        recall_schema = tools[0].inputSchema
        self.assertEqual(recall_schema["required"], ["question"])
        self.assertEqual(recall_schema["properties"]["question"]["type"], "string")

        for alias in tools[1:3]:
            self.assertEqual(alias.inputSchema["required"], ["query"])
            self.assertEqual(alias.inputSchema["properties"]["query"]["type"], "string")

    async def test_observed_retrieval_aliases_return_passages(self) -> None:
        with patch(
            "app.recall_study_passages", return_value=["source evidence"]
        ) as recall:
            async with Client(mcp) as client:
                search_result = await client.call_tool(
                    "search", {"query": "first question"}
                )
                retrieve_result = await client.call_tool(
                    "retrieve", {"query": "second question"}
                )

        self.assertEqual(search_result.structured_content, {"result": ["source evidence"]})
        self.assertEqual(
            retrieve_result.structured_content, {"result": ["source evidence"]}
        )
        recall.assert_has_calls([call("first question"), call("second question")])


if __name__ == "__main__":
    unittest.main()
