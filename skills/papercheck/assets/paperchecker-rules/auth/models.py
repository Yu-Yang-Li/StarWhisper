from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AuthenticatedUser(BaseModel):
    user_id: int
    phone: str
    username: Optional[str] = None
    role: str = "human_user"


class SendCodeRequest(BaseModel):
    phone: str = Field(..., description="11位中国大陆手机号")
    purpose: str = Field(..., description="register/login/reset_password")


class PhoneOnlyRequest(BaseModel):
    phone: str = Field(..., description="11位中国大陆手机号")


class SendCodeResponse(BaseModel):
    message: str
    dev_code: Optional[str] = None


class RegisterRequest(BaseModel):
    phone: str
    code: Optional[str] = None
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    phone: str
    password: str


class LoginByCodeRequest(BaseModel):
    phone: str
    code: str


class ResetPasswordRequest(BaseModel):
    phone: str
    code: Optional[str] = None
    new_password: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: AuthenticatedUser


class UserRecord(BaseModel):
    id: int
    phone: str
    password_hash: str
    username: Optional[str] = None
    role: str = "human_user"
    is_active: int = 1
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
