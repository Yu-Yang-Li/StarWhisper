import random
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from fastapi import HTTPException

from auth.database import auth_db_conn, ensure_auth_schema
from auth.models import AuthenticatedUser
from auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    hash_verification_code,
    verify_password,
)
from auth.smsbao import send_sms_code
from config.config import settings


PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")
CODE_PATTERN = re.compile(r"^\d{6}$")
SUPPORTED_PURPOSES = {"register", "login", "reset_password"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _generate_code() -> str:
    return str(random.randint(100000, 999999))


def _validate_phone(phone: str) -> str:
    phone = (phone or "").strip()
    if not PHONE_PATTERN.match(phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确")
    return phone


def _validate_password(password: str, field_name: str = "密码") -> str:
    value = (password or "").strip()
    if len(value) < 6:
        raise HTTPException(status_code=400, detail=f"{field_name}长度至少6位")
    if len(value) > 128:
        raise HTTPException(status_code=400, detail=f"{field_name}长度不能超过128位")
    return value


def _validate_code(code: str) -> str:
    value = (code or "").strip()
    if not CODE_PATTERN.match(value):
        raise HTTPException(status_code=400, detail="验证码格式不正确")
    return value


def _fetch_user_by_phone(phone: str) -> Optional[Dict]:
    with auth_db_conn() as conn:
        row = conn.execute(
            "SELECT id, phone, password_hash, username, role, is_active, created_at, updated_at, last_login_at "
            "FROM users WHERE phone = ? LIMIT 1",
            (phone,),
        ).fetchone()
        return dict(row) if row else None


def _fetch_user_by_id(user_id: int) -> Optional[Dict]:
    with auth_db_conn() as conn:
        row = conn.execute(
            "SELECT id, phone, password_hash, username, role, is_active, created_at, updated_at, last_login_at "
            "FROM users WHERE id = ? LIMIT 1",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def _mark_latest_code_used(phone: str, purpose: str, code_hash: str) -> None:
    now = _utc_now_iso()
    with auth_db_conn() as conn:
        row = conn.execute(
            "SELECT id FROM verification_codes "
            "WHERE phone = ? AND purpose = ? AND code_hash = ? AND used_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (phone, purpose, code_hash),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE verification_codes SET used_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            conn.commit()


class AuthService:
    @staticmethod
    def bootstrap() -> None:
        ensure_auth_schema()

    @staticmethod
    def _code_required() -> bool:
        return bool(settings.auth_require_sms_code)

    @staticmethod
    def _issue_token(user_row: Dict) -> Tuple[str, int, AuthenticatedUser]:
        ttl_seconds = int(settings.auth_access_token_ttl_minutes) * 60
        user_payload = {
            "sub": user_row["id"],
            "phone": user_row["phone"],
            "username": user_row.get("username"),
            "role": user_row.get("role", "human_user"),
        }
        token = create_access_token(
            secret=settings.auth_jwt_secret,
            payload=user_payload,
            expires_in_seconds=ttl_seconds,
        )
        auth_user = AuthenticatedUser(
            user_id=user_row["id"],
            phone=user_row["phone"],
            username=user_row.get("username"),
            role=user_row.get("role", "human_user"),
        )
        return token, ttl_seconds, auth_user

    @staticmethod
    def send_code(phone: str, purpose: str, request_ip: Optional[str] = None) -> Dict:
        AuthService.bootstrap()
        if not AuthService._code_required():
            return {"message": "内测模式：验证码链路已关闭"}

        phone = _validate_phone(phone)
        purpose = (purpose or "").strip().lower()
        if purpose not in SUPPORTED_PURPOSES:
            raise HTTPException(status_code=400, detail="purpose 必须是 register/login/reset_password")

        user = _fetch_user_by_phone(phone)
        if purpose == "register" and user:
            raise HTTPException(status_code=400, detail="该手机号已注册")
        if purpose in {"login", "reset_password"} and not user:
            raise HTTPException(status_code=400, detail="该手机号尚未注册")

        now = _utc_now()
        cooldown_start = now - timedelta(seconds=int(settings.auth_code_cooldown_seconds))
        with auth_db_conn() as conn:
            recent = conn.execute(
                "SELECT id FROM verification_codes "
                "WHERE phone = ? AND purpose = ? AND created_at > ? "
                "ORDER BY created_at DESC LIMIT 1",
                (phone, purpose, cooldown_start.isoformat()),
            ).fetchone()
            if recent:
                raise HTTPException(status_code=429, detail="验证码发送过于频繁，请稍后再试")

        code = _generate_code()
        code_hash = hash_verification_code(settings.auth_jwt_secret, phone, purpose, code)
        expires_at = (now + timedelta(minutes=int(settings.auth_code_expire_minutes))).isoformat()

        sms_sent = False
        sms_message = "验证码发送成功"
        if settings.smsbao_username and settings.smsbao_password:
            sms_sent, sms_message = send_sms_code(phone, code)
            if not sms_sent:
                raise HTTPException(status_code=500, detail=sms_message)
        elif not settings.auth_dev_mode:
            raise HTTPException(status_code=500, detail="短信宝未配置，且未开启开发模式")

        with auth_db_conn() as conn:
            conn.execute(
                "INSERT INTO verification_codes(phone, code_hash, purpose, expires_at, request_ip, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (phone, code_hash, purpose, expires_at, request_ip, now.isoformat()),
            )
            conn.commit()

        response = {"message": sms_message if sms_sent else "验证码生成成功（开发模式）"}
        if settings.auth_dev_mode:
            response["dev_code"] = code
        return response

    @staticmethod
    def _verify_code(phone: str, purpose: str, code: Optional[str]) -> str:
        phone = _validate_phone(phone)
        code = _validate_code(code)
        purpose = (purpose or "").strip().lower()
        if purpose not in SUPPORTED_PURPOSES:
            raise HTTPException(status_code=400, detail="验证码用途不合法")

        code_hash = hash_verification_code(settings.auth_jwt_secret, phone, purpose, code)
        now = _utc_now().isoformat()

        with auth_db_conn() as conn:
            row = conn.execute(
                "SELECT id FROM verification_codes "
                "WHERE phone = ? AND purpose = ? AND code_hash = ? "
                "AND used_at IS NULL AND expires_at > ? "
                "ORDER BY created_at DESC LIMIT 1",
                (phone, purpose, code_hash, now),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="验证码错误或已过期")
        return code_hash

    @staticmethod
    def register(phone: str, code: Optional[str], password: str, username: Optional[str] = None) -> Dict:
        AuthService.bootstrap()
        phone = _validate_phone(phone)
        password = _validate_password(password)
        code_hash = None
        if AuthService._code_required():
            code_hash = AuthService._verify_code(phone, "register", code)

        if _fetch_user_by_phone(phone):
            raise HTTPException(status_code=400, detail="该手机号已注册")

        now = _utc_now_iso()
        password_hash = hash_password(password)
        safe_username = (username or phone).strip()[:50]

        with auth_db_conn() as conn:
            conn.execute(
                "INSERT INTO users(phone, password_hash, username, role, is_active, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (phone, password_hash, safe_username, "human_user", now, now),
            )
            conn.commit()

        if code_hash:
            _mark_latest_code_used(phone, "register", code_hash)
        user_row = _fetch_user_by_phone(phone)
        token, ttl_seconds, user = AuthService._issue_token(user_row)
        return {
            "message": "注册成功",
            "access_token": token,
            "token_type": "bearer",
            "expires_in_seconds": ttl_seconds,
            "user": user.model_dump(),
        }

    @staticmethod
    def login_with_password(phone: str, password: str) -> Dict:
        AuthService.bootstrap()
        phone = _validate_phone(phone)
        password = _validate_password(password)
        user_row = _fetch_user_by_phone(phone)
        if not user_row or not verify_password(password, user_row["password_hash"]):
            raise HTTPException(status_code=401, detail="手机号或密码错误")

        with auth_db_conn() as conn:
            conn.execute("UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?", (_utc_now_iso(), _utc_now_iso(), user_row["id"]))
            conn.commit()

        token, ttl_seconds, user = AuthService._issue_token(user_row)
        return {
            "message": "登录成功",
            "access_token": token,
            "token_type": "bearer",
            "expires_in_seconds": ttl_seconds,
            "user": user.model_dump(),
        }

    @staticmethod
    def login_with_code(phone: str, code: str) -> Dict:
        AuthService.bootstrap()
        if not AuthService._code_required():
            raise HTTPException(status_code=400, detail="内测模式已关闭验证码登录，请使用密码登录")
        phone = _validate_phone(phone)
        code_hash = AuthService._verify_code(phone, "login", code)
        user_row = _fetch_user_by_phone(phone)
        if not user_row:
            raise HTTPException(status_code=401, detail="该手机号尚未注册")

        _mark_latest_code_used(phone, "login", code_hash)
        with auth_db_conn() as conn:
            conn.execute("UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?", (_utc_now_iso(), _utc_now_iso(), user_row["id"]))
            conn.commit()

        token, ttl_seconds, user = AuthService._issue_token(user_row)
        return {
            "message": "登录成功",
            "access_token": token,
            "token_type": "bearer",
            "expires_in_seconds": ttl_seconds,
            "user": user.model_dump(),
        }

    @staticmethod
    def reset_password(phone: str, code: Optional[str], new_password: str) -> Dict:
        AuthService.bootstrap()
        phone = _validate_phone(phone)
        new_password = _validate_password(new_password, field_name="新密码")
        code_hash = None
        if AuthService._code_required():
            code_hash = AuthService._verify_code(phone, "reset_password", code)

        user_row = _fetch_user_by_phone(phone)
        if not user_row:
            raise HTTPException(status_code=404, detail="用户不存在")

        password_hash = hash_password(new_password)
        now = _utc_now_iso()
        with auth_db_conn() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, now, user_row["id"]),
            )
            conn.commit()

        if code_hash:
            _mark_latest_code_used(phone, "reset_password", code_hash)
        return {"message": "密码重置成功"}

    @staticmethod
    def get_user_from_token(token: str) -> Optional[AuthenticatedUser]:
        AuthService.bootstrap()
        payload = decode_access_token(settings.auth_jwt_secret, token)
        if not payload:
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None
        user_row = _fetch_user_by_id(int(user_id))
        if not user_row or int(user_row.get("is_active", 0)) != 1:
            return None
        return AuthenticatedUser(
            user_id=user_row["id"],
            phone=user_row["phone"],
            username=user_row.get("username"),
            role=user_row.get("role", "human_user"),
        )

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[AuthenticatedUser]:
        user_row = _fetch_user_by_id(int(user_id))
        if not user_row:
            return None
        return AuthenticatedUser(
            user_id=user_row["id"],
            phone=user_row["phone"],
            username=user_row.get("username"),
            role=user_row.get("role", "human_user"),
        )
