"""Typed exception hierarchy for KingoGPT errors.

Using typed exceptions instead of bare ``RuntimeError`` lets callers handle
specific failure modes with ``isinstance`` checks rather than substring
matching on error messages.
"""


class KingoGPTError(Exception):
    """Base exception for all KingoGPT errors."""


class TokenMissingError(KingoGPTError):
    """Access token is missing from cache and CLI arguments."""


class TokenExpiredError(KingoGPTError):
    """Access token is expired or about to expire."""


class TokenCacheCorruptError(KingoGPTError):
    """Token cache JSON file exists but cannot be parsed."""


class AuthenticationError(KingoGPTError):
    """HTTP 401/403 from the KingoGPT API."""


class BackendError(KingoGPTError):
    """KingoGPT backend returned a server-side error (e.g. @list_sort)."""


class UpstreamTimeoutError(KingoGPTError):
    """KingoGPT API timed out."""


class UserProfileError(KingoGPTError):
    """Failed to fetch or parse the user profile from the identity API."""
