from __future__ import annotations


class ServiceValidationError(ValueError):
    """Raised when service input is invalid."""


class ServiceAuthError(ServiceValidationError):
    """Raised when authenticated user context is required but missing."""


class ServiceRuntimeError(RuntimeError):
    """Raised for non-validation runtime failures in service layer."""
