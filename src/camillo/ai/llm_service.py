import logging

from litellm import acompletion, aembedding

from camillo.ai.prompts import VALENCE_PROMPT
from camillo.settings import settings

logger = logging.getLogger(__name__)


class LiteLLMService:
    async def score_valence(self, raw_content: str) -> float:
        try:
            response = await acompletion(
                model=settings.litellm_completion_model,
                messages=[
                    {"role": "user", "content": VALENCE_PROMPT.format(raw_content=raw_content)}
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            score = float(content.strip())
        except Exception:
            logger.exception("Failed to score memory valence; using default")
            return 0.5

        return max(0.1, min(score, 1.0))

    async def get_embedding(self, text: str) -> list[float]:
        response = await aembedding(model=settings.litellm_embedding_model, input=[text])
        embedding = response.data[0]["embedding"]
        return [float(value) for value in embedding]

    async def rerank_results(self, query: str, documents: list[str]) -> list[float]:
        # TODO(Phase 2): route through LiteLLM reranking when the provider is configured.
        if not documents:
            return []
        max_len = max(len(document) for document in documents) or 1
        return [max(0.1, min(len(document) / max_len, 1.0)) for document in documents]
