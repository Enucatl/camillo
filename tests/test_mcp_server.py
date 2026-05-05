def test_mcp_server_module_imports_successfully() -> None:
    """Protect MCP tool registration from import-time dependency drift."""
    import camillo.mcp_server.server as server

    assert server.mcp is not None


def test_mcp_allowed_hosts_include_localhost_defaults(monkeypatch) -> None:
    """Keep direct local MCP runs working when proxy hosts are unset."""
    from camillo.mcp_server.server import _mcp_allowed_hosts

    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)

    assert _mcp_allowed_hosts() == ["127.0.0.1:*", "localhost:*", "[::1]:*"]


def test_mcp_allowed_hosts_add_configured_proxy_hosts(monkeypatch) -> None:
    """Allow reverse proxy host headers without disabling DNS rebinding checks."""
    from camillo.mcp_server.server import _mcp_allowed_hosts

    monkeypatch.setenv(
        "MCP_ALLOWED_HOSTS",
        "camillo.docker.home.arpa, camillo.internal:8000",
    )

    assert _mcp_allowed_hosts() == [
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "camillo.docker.home.arpa",
        "camillo.internal:8000",
    ]
