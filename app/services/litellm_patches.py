"""Small runtime fixes for LiteLLM bugs, applied once at gateway startup."""

import functools
import importlib
import logging

import tiktoken

logger = logging.getLogger(__name__)

# LiteLLM's Gemini batch-embedding path counts prompt tokens *after* the
# provider has returned the vectors, via its module-local `token_counter`
# (imported with `from litellm.utils import token_counter`). tiktoken raises
# ValueError on text containing special tokens such as '<|endoftext|>', which
# turns a successful embedding into a 500. Wrapping that module-level name
# is enough: process_response looks it up as a global at call time.
_BATCH_EMBED_MODULE = "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation"
_FALLBACK_ENCODING = "cl100k_base"


def _count_as_plain_text(text) -> int:
    encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
    texts = text if isinstance(text, list) else [text]
    return sum(len(encoding.encode(str(t), disallowed_special=())) for t in texts)


def _special_token_safe(token_counter):
    @functools.wraps(token_counter)
    def safe_token_counter(*args, **kwargs):
        try:
            return token_counter(*args, **kwargs)
        except ValueError:
            text = kwargs.get("text")
            if text is None:
                raise
            return _count_as_plain_text(text)

    safe_token_counter._gateway_special_token_safe = True
    return safe_token_counter


def apply_litellm_patches() -> None:
    """Idempotent; a no-op if the LiteLLM module layout differs."""
    try:
        module = importlib.import_module(_BATCH_EMBED_MODULE)
    except ImportError:
        logger.warning(f"{_BATCH_EMBED_MODULE} not found; special-token patch skipped")
        return
    original = getattr(module, "token_counter", None)
    if original is None:
        logger.warning(f"{_BATCH_EMBED_MODULE}.token_counter not found; patch skipped")
        return
    if getattr(original, "_gateway_special_token_safe", False):
        return
    module.token_counter = _special_token_safe(original)
