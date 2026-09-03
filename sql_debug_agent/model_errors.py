class ModelConfigurationError(RuntimeError):
    """Raised when required model configuration is missing."""


class ModelAPIError(RuntimeError):
    """Raised when a model request fails or returns an invalid payload."""
