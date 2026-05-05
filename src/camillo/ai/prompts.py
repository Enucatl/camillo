from functools import lru_cache

from jinja2 import Environment, PackageLoader, select_autoescape


@lru_cache
def _template_environment() -> Environment:
    """Create the package template environment once per process.

    Jinja keeps prompt wording in versioned template files, which makes prompt
    changes reviewable without mixing natural language into Python control flow.
    """
    return Environment(
        loader=PackageLoader("camillo.ai", "templates"),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_valence_prompt(raw_content: str) -> str:
    """Render the memory-importance scoring prompt.

    Args:
        raw_content: The interaction text to score.

    Returns:
        A LiteLLM-ready prompt that asks for a continuous importance score.
    """
    return (
        _template_environment().get_template("valence_score.jinja").render(raw_content=raw_content)
    )


def render_relationship_prompt(
    intent: str,
    new_content: str,
    numbered_memories: str,
) -> str:
    """Render the contradiction-aware reconciliation prompt.

    Args:
        intent: Caller intent for the memory submission.
        new_content: Candidate memory text to compare.
        numbered_memories: Existing memories formatted as an indexed list.

    Returns:
        A LiteLLM-ready prompt that asks for strict JSON relationship output.
    """
    return (
        _template_environment()
        .get_template("relationship_resolution.jinja")
        .render(
            intent=intent,
            new_content=new_content,
            numbered_memories=numbered_memories,
        )
    )
