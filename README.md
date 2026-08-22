# Tool-Box Phase 2

FastMCP solution for **Stage 2 — School Days**.

## MCP tools

- `recall_study_passages(question)` downloads and indexes the five assigned
  documents, then returns the most relevant source passages as `list[str]`.
  The passages are counted with `o200k_base` and never exceed the stage's
  combined 900-token recall limit.
- `next_route_node(map_id, current_node, destination, hops_remaining,
  avoid_nodes)` fetches the opaque directed map and returns the adjacent next
  node on a least-cost route. It includes both edge weights and destination
  entry tolls and supports the optional remaining-hop constraint.

These two tools also compose for school-trip questions: recall the `STOP_XX`
destination first, then traverse to that exact destination one node at a time.

The Streamable HTTP MCP endpoint is exposed at `/mcp` (and `/mcp/`). A health
endpoint is available at `/health`.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Run all tests:

```bash
python -m unittest discover -s tests -v
```

The service defaults to the supplied challenge API. Override it only when
running against a local fixture:

```bash
TOOLBOX_API_BASE=http://127.0.0.1:9000 python3 app.py
```

## Deployment

`render.yaml` and `Procfile` are included. On Render, create the service from
this repository; the service binds to `0.0.0.0` and uses the platform's `PORT`.
Register the deployed base URL with the challenge. The evaluator will try
`{teamUrl}/mcp/` first and then `{teamUrl}/mcp`.
