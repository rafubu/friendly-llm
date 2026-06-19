class LitertError(Exception):
    """Base exception for all SDK errors."""


class ConnectionError(LitertError):
    """Connection to signaling or node failed."""


class AuthError(LitertError):
    """Authentication failed (invalid JWT, API key, etc)."""


class ModelNotFoundError(LitertError):
    """Requested model is not available."""


class RoomCreationError(LitertError):
    """Could not create a room (node at capacity)."""


class TimeoutError(LitertError):
    """P2P connection or inference timed out."""


class ContextOverflowError(LitertError):
    """The input messages exceed the model's context window.

    Attributes:
        estimated_tokens: Estimated input token count.
        context_limit: Maximum tokens the model supports.
        overflow_by: How many tokens over the limit.
        suggestion: Action recommended by the server.
    """

    def __init__(
        self,
        estimated_tokens: int,
        context_limit: int,
        overflow_by: int,
        suggestion: str = "",
        model: str = "",
    ) -> None:
        self.estimated_tokens = estimated_tokens
        self.context_limit = context_limit
        self.overflow_by = overflow_by
        self.suggestion = suggestion
        self.model = model
        msg = (
            f"Context overflow for model '{model}': "
            f"{estimated_tokens} tokens > {context_limit} limit "
            f"(exceeded by {overflow_by}). {suggestion}"
        )
        super().__init__(msg)
