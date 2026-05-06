import pytest


def test_mcp_server_module_imports_successfully() -> None:
    """Protect MCP tool registration from import-time dependency drift."""
    import camillo.mcp_server.server as server

    assert server.mcp is not None


def test_mcp_allowed_hosts_include_localhost_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep direct local MCP runs working when proxy hosts are unset."""
    from camillo.mcp_server.server import _mcp_allowed_hosts

    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)

    assert _mcp_allowed_hosts() == [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
        "camillo",
        "camillo:*",
    ]


def test_mcp_allowed_hosts_add_configured_proxy_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow reverse proxy host headers without disabling DNS rebinding checks."""
    from camillo.mcp_server.server import _mcp_allowed_hosts

    monkeypatch.setenv(
        "MCP_ALLOWED_HOSTS",
        "camillo.docker.home.arpa, camillo.internal:8000",
    )

    assert _mcp_allowed_hosts() == [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
        "camillo",
        "camillo:*",
        "camillo.docker.home.arpa",
        "camillo.docker.home.arpa:*",
        "camillo.internal:8000",
    ]


@pytest.mark.asyncio
async def test_recall_memory_metadata_is_adaptive_without_count_output() -> None:
    """Expose MCP recall as adaptive while hiding bookkeeping details."""
    from camillo.mcp_server.server import mcp

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    recall_tool = tools["recall_memory"]
    dumped = recall_tool.model_dump(by_alias=True)

    assert dumped["annotations"]["readOnlyHint"] is False
    assert dumped["annotations"]["destructiveHint"] is False
    assert dumped["annotations"]["idempotentHint"] is False
    assert "bookkeeping" not in dumped["description"]
    assert dumped["_meta"]["side_effects"]
    top_k_schema = dumped["inputSchema"]["properties"]["top_k"]
    integer_schema = next(
        schema for schema in top_k_schema["anyOf"] if schema.get("type") == "integer"
    )
    assert top_k_schema["default"] is None
    assert integer_schema["minimum"] == 1
    assert "maximum" not in integer_schema
    assert "access_count" not in dumped["outputSchema"]["$defs"]["McpRecalledMemory"]["properties"]


@pytest.mark.asyncio
async def test_record_interaction_metadata_is_mutating() -> None:
    """Keep write tools clearly separate from read-only MCP tools."""
    from camillo.mcp_server.server import mcp

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    record_tool = tools["record_interaction"].model_dump(by_alias=True)

    assert record_tool["annotations"]["readOnlyHint"] is False
    assert record_tool["annotations"]["destructiveHint"] is False
    assert record_tool["_meta"]["side_effects"]
