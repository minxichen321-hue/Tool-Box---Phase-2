"""HTTP entry point for the Tool-Box Phase 2 MCP server."""

import os

from fastapi import FastAPI
from fastmcp import FastMCP
import uvicorn

from tool_logic import next_route_node, recall_study_passages


mcp = FastMCP("Tool-Box Phase 2")


@mcp.tool(
    name="next_route_node",
    description=(
        "CALL THIS for every 'How can I get from START to DESTINATION?' question, "
        "and call it again after every move until arrival. Return the node from "
        "this tool as the next move. Pass map_id, the node where the android is "
        "currently standing, and the requested destination. If the request says "
        "there are N hops left, also pass hops_left=N; the current move counts as "
        "one of those hops."
    ),
)
def next_route_node_tool(
    map_id: str,
    current_node: str,
    destination: str,
    hops_left: int | None = None,
) -> str:
    return next_route_node(
        map_id=map_id,
        current_node=current_node,
        destination=destination,
        hops_remaining=hops_left,
    )


@mcp.tool(
    name="search",
    description=(
        "Search the assigned study materials for a recall or school-trip "
        "question. Pass the complete question as query. Returns relevant source "
        "passages within the combined 900-token limit."
    ),
)
def search_tool(query: str) -> list[str]:
    return recall_study_passages(query)


@mcp.tool(
    name="retrieve",
    description=(
        "Retrieve source evidence from the assigned study materials. Pass the "
        "complete question as query. This is a compatibility alias for search "
        "and returns the same 900-token-limited list of passages."
    ),
)
def retrieve_tool(query: str) -> list[str]:
    return recall_study_passages(query)


@mcp.tool(
    name="recall_study_passages",
    description=(
        "Use for every question about the assigned study materials, including "
        "school-trip questions that ask which STOP_ destination matches a place. "
        "Pass the complete question once. It returns a list of source passages "
        "within the 900-token recall limit; read them and answer from their facts."
    ),
)
def recall_study_passages_tool(question: str) -> list[str]:
    return recall_study_passages(question)


mcp_app = mcp.http_app(path="/")
app = FastAPI(title="Tool-Box Phase 2", lifespan=mcp_app.lifespan)


@app.get("/")
@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "Tool-Box Phase 2",
        "mcp": "/mcp",
        "tools": [
            "next_route_node",
            "search",
            "retrieve",
            "recall_study_passages",
        ],
    }


app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
