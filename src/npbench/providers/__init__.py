"""Provider adapters. Every call returns a ``ProviderResponse`` with full metadata; the fake provider is
unmistakably typed as a mock (``is_mock=True``) and can never satisfy a live-study validator."""

from .base import ProviderError, ProviderRequest, ProviderResponse, build_provider

__all__ = ["ProviderError", "ProviderRequest", "ProviderResponse", "build_provider"]
