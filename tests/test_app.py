import unittest

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
    async def test_server_advertises_only_the_two_required_tools(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        self.assertEqual(
            [tool.name for tool in tools],
            ["recall_study_passages", "next_route_node"],
        )
        recall_schema = tools[0].inputSchema
        self.assertEqual(recall_schema["required"], ["question"])
        self.assertEqual(recall_schema["properties"]["question"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
