import sys
import types

import pytest

from camillo import tracing_config


def test_phoenix_tracing_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave Phoenix tracing disabled when the feature flag is off."""
    monkeypatch.setattr(tracing_config.settings, "phoenix_tracing_enabled", False)
    monkeypatch.setattr(tracing_config, "_configured", False)

    assert tracing_config.configure_phoenix_tracing() is False


def test_phoenix_tracing_registers_and_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register Phoenix once and instrument LiteLLM when enabled."""
    calls = {}

    phoenix_module = types.ModuleType("phoenix")
    phoenix_otel_module = types.ModuleType("phoenix.otel")
    openinference_module = types.ModuleType("openinference")
    instrumentation_module = types.ModuleType("openinference.instrumentation")
    litellm_module = types.ModuleType("openinference.instrumentation.litellm")

    def register(**kwargs: object) -> None:
        """Capture Phoenix registration arguments for the assertion."""
        calls["register"] = kwargs

    class LiteLLMInstrumentor:
        """Stand in for the optional LiteLLM instrumentor."""

        def instrument(self) -> None:
            """Record that instrumentation was attempted."""
            calls["instrumented"] = True

    phoenix_otel_module.register = register
    litellm_module.LiteLLMInstrumentor = LiteLLMInstrumentor

    monkeypatch.setitem(sys.modules, "phoenix", phoenix_module)
    monkeypatch.setitem(sys.modules, "phoenix.otel", phoenix_otel_module)
    monkeypatch.setitem(sys.modules, "openinference", openinference_module)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation", instrumentation_module)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.litellm", litellm_module)
    monkeypatch.setattr(tracing_config.settings, "phoenix_tracing_enabled", True)
    monkeypatch.setattr(
        tracing_config.settings, "phoenix_collector_endpoint", "http://phoenix:6006"
    )
    monkeypatch.setattr(tracing_config.settings, "phoenix_project_name", "camillo")
    monkeypatch.setattr(tracing_config, "_configured", False)

    assert tracing_config.configure_phoenix_tracing() is True
    assert calls["register"] == {
        "endpoint": "http://phoenix:6006",
        "project_name": "camillo",
        "auto_instrument": False,
    }
    assert calls["instrumented"] is True
