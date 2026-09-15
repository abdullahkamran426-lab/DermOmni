from __future__ import annotations


class SkinResearchError(RuntimeError):
    """Base class for expected, user-facing skin research failures."""


class VisionAnalysisError(SkinResearchError):
    """Safe, user-facing failure from the visual-analysis step."""

    def __init__(self, message: str, *, code: str = "vision_provider_error") -> None:
        super().__init__(message)
        self.code = code


class SearchError(SkinResearchError):
    """Expected failure while querying an external research provider."""

    def __init__(self, message: str, *, code: str = "search_provider_error") -> None:
        super().__init__(message)
        self.code = code


class ProviderError(RuntimeError):
    """Base class for typed downstream provider failures (no substring matching)."""

    def __init__(self, message: str, *, code: str = "provider_error") -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(ProviderError):
    """Missing/invalid server-side config (e.g. API key). Maps to 500."""

    def __init__(self, message: str, *, code: str = "missing_api_key") -> None:
        super().__init__(message, code=code)


class TranscriptionError(ProviderError):
    """Speech-to-text failure. Maps to 502."""

    def __init__(self, message: str, *, code: str = "stt_provider_error") -> None:
        super().__init__(message, code=code)


class SpeechSynthesisError(ProviderError):
    """Text-to-speech failure. Maps to 502."""

    def __init__(self, message: str, *, code: str = "tts_provider_error") -> None:
        super().__init__(message, code=code)


class GuidanceError(ProviderError):
    """Guidance/vision LLM failure for the consult path. Maps to 502."""

    def __init__(self, message: str, *, code: str = "guidance_provider_error") -> None:
        super().__init__(message, code=code)
