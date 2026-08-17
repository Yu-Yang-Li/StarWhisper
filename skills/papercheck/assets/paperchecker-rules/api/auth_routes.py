from fastapi import APIRouter, Request

from auth.dependencies import get_current_user_required
from auth.models import (
    LoginByCodeRequest,
    LoginRequest,
    PhoneOnlyRequest,
    RegisterRequest,
    ResetPasswordRequest,
    SendCodeRequest,
)
from auth.service import AuthService


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/send-code")
async def send_code(payload: SendCodeRequest, request: Request):
    client_ip = request.client.host if request.client else None
    return AuthService.send_code(phone=payload.phone, purpose=payload.purpose, request_ip=client_ip)


@router.post("/register")
async def register(payload: RegisterRequest):
    return AuthService.register(
        phone=payload.phone,
        code=payload.code,
        password=payload.password,
        username=payload.username,
    )


@router.post("/login")
async def login(payload: LoginRequest):
    return AuthService.login_with_password(phone=payload.phone, password=payload.password)


@router.post("/login-by-code")
async def login_by_code(payload: LoginByCodeRequest):
    return AuthService.login_with_code(phone=payload.phone, code=payload.code)


@router.post("/forgot-password/send-code")
async def forgot_password_send_code(payload: PhoneOnlyRequest, request: Request):
    # 固定为 reset_password，避免前端传错 purpose
    client_ip = request.client.host if request.client else None
    return AuthService.send_code(phone=payload.phone, purpose="reset_password", request_ip=client_ip)


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    return AuthService.reset_password(
        phone=payload.phone,
        code=payload.code,
        new_password=payload.new_password,
    )


@router.get("/me")
async def me(request: Request):
    user = get_current_user_required(request)
    return {"user": user.model_dump()}


@router.post("/logout")
async def logout():
    # 当前为无状态 JWT，前端删除本地 token 即可。
    return {"message": "已退出登录"}
