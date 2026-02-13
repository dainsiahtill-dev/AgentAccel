from __future__ import annotations

from accel.mcp_server import SERVER_NAME, create_server


def test_create_mcp_server_smoke() -> None:
    server = create_server()
    assert server is not None
    assert SERVER_NAME == "agent-accel-mcp"
    assert hasattr(server, "run")
