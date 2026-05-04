def test_mcp_server_module_imports_successfully() -> None:
    """Protect MCP tool registration from import-time dependency drift."""
    import camillo.mcp_server.server as server

    assert server.mcp is not None
