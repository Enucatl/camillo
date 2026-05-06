from camillo.worker import build_parser


def test_worker_parser_accepts_once_and_loop_modes() -> None:
    """Ensure worker CLI keeps documented execution modes importable."""
    parser = build_parser()

    once = parser.parse_args(["--once", "--namespace", "repo:backend"])
    loop = parser.parse_args(["--loop", "--namespace", "repo:backend", "--dry-run"])

    assert once.once is True
    assert once.loop is False
    assert once.namespace == "repo:backend"
    assert loop.loop is True
    assert loop.dry_run is True
