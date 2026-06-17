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
