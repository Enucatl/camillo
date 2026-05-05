from fastapi.testclient import TestClient

from camillo.main import app


def test_mcp_get_returns_405() -> None:
    """Protect the MCP endpoint from misleading GET 404s.

    Streamable HTTP clients may probe the MCP URL with GET. Camillo does not
    serve SSE here, so the server should answer 405 instead of looking missing.
    """
    client = TestClient(app)

    response = client.get("/mcp")

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


def test_mcp_slash_get_returns_405() -> None:
    """Keep the trailing-slash probe consistent with the bare MCP path."""
    client = TestClient(app)

    response = client.get("/mcp/")

    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
