from importlib import import_module


def test_legacy_memory_mapping_covers_representative_rows() -> None:
    """Protect the data-preserving type mapping before destructive column drops."""
    migration = import_module("migrate.versions.0006_single_user_memory")
    assert migration.LEGACY_TYPE_MAP == {
        "episodic": "episode",
        "semantic": "fact",
        "relationship": "fact",
        "profile": "fact",
        "core": "fact",
        "preference": "preference",
        "procedural": "procedure",
    }
