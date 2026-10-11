"""CPU-only record and duck-typed API contracts for the Topic 04 notebook."""
from typing import Any, Literal, Mapping, Protocol, Sequence, TypedDict

TaskLabel = Literal['billing', 'access', 'bug']
TaskStatus = Literal['complete', 'partial', 'failed']


class PredictionRow(TypedDict):
    """One parser-normalized attempt; invalid/incomplete outputs have no label."""

    case_id: str
    expected: TaskLabel
    content: str
    predicted: TaskLabel | None
    status: TaskStatus
    finish_reason: str | None


class PredictionScores(TypedDict):
    """All-attempt counts/rates and a 3-by-4 true-row/predicted-column matrix."""

    attempts: int
    correct: int
    valid: int
    complete: int
    accuracy: float | None
    valid_rate: float | None
    completion_rate: float | None
    confusion: list[list[int]]


class ClientLike(Protocol):
    """OpenAI-compatible client; concrete SDK is imported only in GPU work."""

    chat: Any


class TokenizerLike(Protocol):
    """Tokenizer subset used by the visible calibration callback."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode one string as token IDs.

        Parameters
        ----------
        text : str
            Input string, not a pre-rendered batch.
        add_special_tokens : bool
            Whether to add tokenizer-specific special tokens.

        Returns
        -------
        list[int]
            One-dimensional token ID sequence of length S.
        """
        ...

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]], *, tokenize: bool,
                            add_generation_prompt: bool) -> str | list[int] | Mapping[str, list[int]]:
        """Render or tokenize one chat with the pinned tokenizer's template.

        Parameters
        ----------
        messages : sequence of mappings
            Ordered role/content records, not a batch of chats.
        tokenize : bool
            Return token IDs rather than rendered text when True.
        add_generation_prompt : bool
            Append the assistant prefix before generation.

        Returns
        -------
        str or list[int] or mapping
            Rendered chat or one input-ID sequence; some versions wrap it.
        """
        ...
