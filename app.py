"""HTTP entry point for the Tool-Box Phase 2 MCP server."""

import os

from fastapi import FastAPI
from fastmcp import FastMCP
import uvicorn

from tool_logic import next_route_node, recall_study_passages


mcp = FastMCP("Tool-Box Phase 2")


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


@mcp.tool(
    name="next_route_node",
    description=(
        "Use at every journey step. Returns exactly the adjacent next node on a "
        "least-total-cost directed route, where cost is edge weights plus each "
        "entered node's toll. Pass the opaque map_id, your current node, and the "
        "required destination. If the prompt gives hops left, pass that number as "
        "hops_remaining; it includes the move being requested. Optionally pass "
        "previously visited nodes in avoid_nodes."
    ),
)
def next_route_node_tool(
    map_id: str,
    current_node: str,
    destination: str,
    hops_remaining: int | None = None,
    avoid_nodes: list[str] | None = None,
) -> str:
    return next_route_node(
        map_id=map_id,
        current_node=current_node,
        destination=destination,
        hops_remaining=hops_remaining,
        avoid_nodes=avoid_nodes,
    )


mcp_app = mcp.http_app(path="/")
app = FastAPI(title="Tool-Box Phase 2", lifespan=mcp_app.lifespan)


@app.get("/")
@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "Tool-Box Phase 2",
        "mcp": "/mcp",
        "tools": ["recall_study_passages", "next_route_node"],
    }


app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
