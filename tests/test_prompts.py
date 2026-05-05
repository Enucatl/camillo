from camillo.ai.prompts import render_relationship_prompt


def test_relationship_prompt_renders_template_values() -> None:
    """Protect the reconciliation prompt from drifting back into inline strings."""
    prompt = render_relationship_prompt(
        intent="correct",
        new_content="Camillo now uses Jinja templates for prompts.",
        numbered_memories="0. Camillo used inline prompts before.",
    )

    assert "Camillo now uses Jinja templates for prompts." in prompt
    assert "0. Camillo used inline prompts before." in prompt
    assert "<intent>" in prompt
    assert "</intent>" in prompt
    assert "<new_content>" in prompt
    assert "<existing_memories>" in prompt
