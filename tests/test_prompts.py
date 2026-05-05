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


def test_valence_prompt_separates_user_and_assistant_turns() -> None:
    """Protect the interaction scorer from flattening speaker boundaries."""
    from camillo.ai.prompts import render_valence_prompt

    prompt = render_valence_prompt("User asks for a cache change.", "Assistant agrees.")

    assert "<interaction>" in prompt
    assert "<user>" in prompt
    assert "</user>" in prompt
    assert "<assistant>" in prompt
    assert "</assistant>" in prompt
    assert "User asks for a cache change." in prompt
    assert "Assistant agrees." in prompt
