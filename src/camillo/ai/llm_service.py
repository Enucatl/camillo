import json
import re

import litellm
from loguru import logger

from camillo.ai.prompts import render_relationship_prompt, render_valence_prompt
from camillo.db.models import Memory
from camillo.interfaces import CompletionProvider, EmbeddingProvider, Reranker
from camillo.schemas.submit_memory import MemoryRelationshipClassification
from camillo.settings import settings

OPENROUTER_RERANK_API_BASE = "https://openrouter.ai/api/v1/rerank"


class LiteLLMService(CompletionProvider, EmbeddingProvider, Reranker):
    """LiteLLM-backed implementation of the AI provider interfaces."""

    async def score_valence(self, user_msg: str, ai_msg: str) -> float:
        """Score whether an interaction is worth retaining long-term.

        Args:
            user_msg: The user-side turn content to classify.
            ai_msg: The assistant-side turn content to classify.

        Returns:
            A clamped continuous score from 0.0 to 1.0, with 0.5 as a neutral
            fallback when the provider response cannot be parsed.
        """
        try:
            response = await litellm.acompletion(
                model=settings.litellm_completion_model,
                messages=[
                    {
                        "role": "user",
                        "content": render_valence_prompt(user_msg, ai_msg),
                    }
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            score = float(content.strip())
        except Exception:
            logger.exception("Failed to score memory valence; using default")
            return 0.5

        return max(0.0, min(score, 1.0))

    async def get_embedding(self, text: str) -> list[float]:
        """Embed text using the configured LiteLLM embedding model.

        Args:
            text: The text to embed.

        Returns:
            A list of floats matching the configured embedding dimension.
        """
        response = await litellm.aembedding(model=settings.litellm_embedding_model, input=[text])
        embedding = response.data[0]["embedding"]
        return [float(value) for value in embedding]

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        """Use reranking as an optional relevance signal, not a hard dependency.

        Recall should remain available when no rerank model is configured or a
        provider returns a shape LiteLLM does not normalize consistently.

        Args:
            query: The recall query.
            documents: Candidate memory texts.

        Returns:
            One relevance score per document, with order-preserving fallback.
        """
        if not documents:
            return []

        fallback = [1.0 - (index / max(len(documents), 1)) * 0.2 for index in range(len(documents))]
        rerank_model = settings.litellm_rerank_model
        if not rerank_model:
            return fallback

        try:
            rerank_kwargs = _rerank_provider_kwargs(rerank_model)
            response = await litellm.arerank(
                model=rerank_kwargs.pop("model"),
                query=query,
                documents=documents,
                **rerank_kwargs,
            )
            results = _response_value(response, "results") or []
            scores = [0.0] * len(documents)

            for item in results:
                index = _response_value(item, "index")
                score = _response_value(item, "relevance_score")
                if score is None:
                    score = _response_value(item, "score")
                if index is None:
                    continue
                index = int(index)
                if 0 <= index < len(scores):
                    scores[index] = float(score or 0.0)

            return scores
        except Exception:
            logger.exception("Failed to rerank recall candidates; using fallback")
            return fallback

    async def classify_memory_relationships(
        self,
        intent: str,
        new_content: str,
        existing_memories: list[Memory],
    ) -> list[MemoryRelationshipClassification]:
        """Classify new durable memory against related active memories.

        Args:
            intent: Caller intent passed through to the classifier.
            new_content: Proposed durable memory text.
            existing_memories: Recalled memories to compare with the candidate.

        Returns:
            One defensive classification per existing memory.
        """
        if not existing_memories:
            return []

        fallback = _fallback_relationships(len(existing_memories))
        numbered_memories = "\n".join(
            f"{index}. {memory.raw_content}" for index, memory in enumerate(existing_memories)
        )
        prompt = render_relationship_prompt(intent, new_content, numbered_memories)

        try:
            response = await litellm.acompletion(
                model=settings.litellm_completion_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content.strip())
            if not isinstance(parsed, list):
                return fallback

            classifications_by_index: dict[int, MemoryRelationshipClassification] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                raw_index = item.get("index")
                if (
                    not isinstance(raw_index, int)
                    or raw_index < 0
                    or raw_index >= len(existing_memories)
                ):
                    continue
                if "confidence" in item:
                    item["confidence"] = max(0.0, min(float(item["confidence"]), 1.0))
                classification = MemoryRelationshipClassification.model_validate(item)
                classifications_by_index[classification.index] = classification

            return [
                classifications_by_index.get(index, fallback[index])
                for index in range(len(existing_memories))
            ]
        except Exception:
            logger.exception("Failed to classify memory relationships; using fallback")
            return fallback

    async def synthesize_dream(
        self,
        cluster_memories: list[str],
        *,
        namespace: str,
    ) -> dict:
        """Ask LiteLLM to promote a cluster into durable memory candidates.

        Args:
            cluster_memories: Raw episodic memory text in evidence-index order.
            namespace: Partition used to keep synthesis context scoped.

        Returns:
            Parsed JSON dream output, or a conservative no-op fallback.
        """
        if not cluster_memories:
            return _fallback_dream()

        numbered_memories = "\n".join(
            f"{index}. {memory}" for index, memory in enumerate(cluster_memories)
        )
        prompt = _render_dream_prompt(namespace, numbered_memories)
        model = settings.dreaming_model or settings.litellm_completion_model
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(_strip_json_fence(content))
            if not isinstance(parsed, dict):
                return _fallback_dream()
            memories = parsed.get("memories")
            if not isinstance(memories, list):
                parsed["memories"] = []
            parsed["should_create_memory"] = bool(parsed.get("should_create_memory"))
            parsed["summary"] = str(parsed.get("summary") or "")
            return parsed
        except Exception:
            logger.exception("Failed to synthesize dream; using fallback")
            return _fallback_dream()


def _response_value(item: object, key: str) -> object | None:
    """Handle provider response shape drift behind one defensive accessor.

    Args:
        item: Dict-like or attribute-based LiteLLM response object.
        key: Field name to read.

    Returns:
        The field value when present, otherwise `None`.
    """
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _rerank_provider_kwargs(model: str) -> dict[str, str]:
    """Route OpenRouter-backed rerank models through LiteLLM's proxy-compatible path.

    LiteLLM does not expose an OpenRouter rerank provider, but its generic
    proxy route can post Cohere-shaped rerank requests to OpenRouter's rerank
    endpoint while preserving normalized response parsing.

    Args:
        model: Configured rerank model name.

    Returns:
        Keyword arguments for `litellm.arerank`.
    """
    if settings.openrouter_api_key and (
        model.startswith("openrouter/") or model.startswith("cohere/rerank")
    ):
        return {
            "model": model.removeprefix("openrouter/"),
            "custom_llm_provider": "litellm_proxy",
            "api_base": OPENROUTER_RERANK_API_BASE,
            "api_key": settings.openrouter_api_key,
        }
    return {"model": model}


def _fallback_relationships(count: int) -> list[MemoryRelationshipClassification]:
    """Build index-complete unavailable classifications.

    Args:
        count: Number of existing memories being classified.

    Returns:
        Unrelated classifications that keep reconciliation conservative.
    """
    return [
        MemoryRelationshipClassification(
            index=index,
            relation="unrelated",
            confidence=0.0,
            contradiction_type="none",
            resolution="keep_both",
            rationale="classification unavailable",
            old_memory_refinement=None,
            new_memory_refinement=None,
        )
        for index in range(count)
    ]


def _strip_json_fence(content: str) -> str:
    """Remove common markdown code fences from provider JSON responses.

    Args:
        content: Raw completion content.

    Returns:
        A string intended for `json.loads`.
    """
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _fallback_dream() -> dict:
    """Return the conservative no-op dream shape.

    Returns:
        JSON-compatible dictionary matching dream synthesis expectations.
    """
    return {
        "should_create_memory": False,
        "summary": "Dream synthesis unavailable.",
        "memories": [],
    }


def _render_dream_prompt(namespace: str, numbered_memories: str) -> str:
    """Render the dreaming prompt without introducing a template dependency.

    Args:
        namespace: Memory partition being consolidated.
        numbered_memories: Evidence-indexed source memory text.

    Returns:
        Provider prompt that asks for strict JSON.
    """
    return f"""You are consolidating a graph-connected cluster of raw episodic memories.

These memories are connected through Hebbian edges, meaning they were adjacent,
co-accessed, or repeatedly associated. Your job is to decide whether this
cluster contains durable knowledge worth promoting into long-term semantic memory.

Extract only stable, reusable information:
- durable project decisions
- user preferences
- architectural constraints
- recurring implementation patterns
- procedural rules
- relationship facts
- long-lived goals or commitments

Do not extract:
- temporary chatter
- one-off status updates
- uncertain guesses
- secrets or credentials
- facts not supported by the cluster
- overly broad claims that are only true in a narrow context

If the cluster does not contain durable knowledge, return should_create_memory=false.

When creating memories:
- Be compact.
- Be specific.
- Preserve context.
- Preserve uncertainty.
- Do not invent facts.
- Prefer one high-quality semantic memory over many weak memories.
- Include evidence_indices pointing to the source memories that support the claim.

Namespace:
{namespace}

Cluster memories:
{numbered_memories}

Return only valid JSON in this shape:
{{
  "should_create_memory": true,
  "summary": "Short description of the durable pattern, or why none exists.",
  "memories": [
    {{
      "content": "Compact durable memory.",
      "memory_type": "semantic",
      "confidence": 0.9,
      "evidence_indices": [0, 1],
      "rationale": "Why this is durable and supported."
    }}
  ]
}}

Allowed memory_type values:
- semantic
- preference
- procedural
- relationship
- profile
- core

If nothing durable should be created:
{{
  "should_create_memory": false,
  "summary": "No durable memory found.",
  "memories": []
}}"""
