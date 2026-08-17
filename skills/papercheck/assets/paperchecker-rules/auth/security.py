import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padded = raw + "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def hash_password(password: str, iterations: int = 210_000) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        _b64url_encode(salt),
        _b64url_encode(digest),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iter_text, salt_text, digest_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iter_text)
        salt = _b64url_decode(salt_text)
        expected_digest = _b64url_decode(digest_text)
        actual_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual_digest, expected_digest)
    except Exception:
        return False


def hash_verification_code(secret: str, phone: str, purpose: str, code: str) -> str:
    payload = f"{phone}:{purpose}:{code}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def create_access_token(
    secret: str,
    payload: Dict[str, Any],
    expires_in_seconds: int,
) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now_ts = int(time.time())
    token_payload = dict(payload)
    token_payload["iat"] = now_ts
    token_payload["exp"] = now_ts + int(expires_in_seconds)
    header_encoded = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_encoded = _b64url_encode(json.dumps(token_payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_encoded}.{payload_encoded}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_encoded}.{payload_encoded}.{_b64url_encode(signature)}"


def decode_access_token(secret: str, token: str) -> Optional[Dict[str, Any]]:
    try:
        header_encoded, payload_encoded, signature_encoded = token.split(".", 2)
        signing_input = f"{header_encoded}.{payload_encoded}".encode("utf-8")
        expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        provided_signature = _b64url_decode(signature_encoded)
        if not hmac.compare_digest(expected_signature, provided_signature):
            return None

        payload = json.loads(_b64url_decode(payload_encoded).decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if exp <= int(time.time()):
            return None
        return payload
    except Exception:
        return None
