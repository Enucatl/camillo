VALID_MEMORY_SCOPES = {"local", "shared", "global"}


def default_scope_for_memory_type(memory_type: str) -> str:
    """Choose the initial reuse scope for a memory type.

    The policy is intentionally small: raw episodes stay local, reusable
    procedures can cross namespaces, and only core memories become global by
    default.

    Args:
        memory_type: Memory type requested by ingestion or reconciliation.

    Returns:
        One of `local`, `shared`, or `global`.
    """
    if memory_type == "episodic":
        return "local"
    if memory_type == "procedural":
        return "shared"
    if memory_type == "core":
        return "global"
    if memory_type == "preference":
        return "shared"
    return "local"


def normalize_memory_scope(scope: str | None, memory_type: str) -> str:
    """Validate a caller-supplied scope or derive the memory-type default.

    Args:
        scope: Optional explicit scope.
        memory_type: Memory type used when scope is omitted or invalid.

    Returns:
        Valid memory scope.
    """
    if scope in VALID_MEMORY_SCOPES:
        return scope
    return default_scope_for_memory_type(memory_type)


def scope_affinity(memory_namespace: str, memory_scope: str, query_namespace: str) -> float:
    """Score how appropriate a memory scope is for a query namespace.

    Args:
        memory_namespace: Namespace where the memory originated.
        memory_scope: Memory reuse scope.
        query_namespace: Namespace used by the recall request.

    Returns:
        Normalized affinity for final recall scoring.
    """
    if memory_namespace == query_namespace:
        return 1.0
    if memory_scope == "global":
        return 0.85
    if memory_scope == "shared":
        return 0.75
    return 0.0
