from typing import Optional

from fastapi import HTTPException, Request

from auth.models import AuthenticatedUser
from auth.service import AuthService
from config.config import settings


def _extract_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return None
    prefix = "bearer "
    if auth_header.lower().startswith(prefix):
        return auth_header[len(prefix):].strip()
    return None


def get_current_user_optional(request: Request) -> Optional[AuthenticatedUser]:
    if not settings.auth_enabled:
        return None
    token = _extract_bearer_token(request)
    if not token:
        return None
    return AuthService.get_user_from_token(token)


def get_current_user_required(request: Request) -> AuthenticatedUser:
    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


def enforce_auth_for_request(request: Request) -> Optional[AuthenticatedUser]:
    if not settings.auth_enabled:
        return None
    user = get_current_user_required(request)
    request.state.current_user = user
    return user
