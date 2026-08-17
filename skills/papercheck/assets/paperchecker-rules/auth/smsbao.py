from typing import Tuple
from urllib.parse import quote

import requests

from config.config import settings


SMSBAO_API = "https://api.smsbao.com/sms"

SMSBAO_ERROR_MESSAGES = {
    "30": "短信宝密码错误",
    "40": "短信宝账号不存在",
    "41": "短信宝余额不足",
    "43": "短信宝IP地址限制",
    "50": "短信内容含敏感词",
    "51": "手机号格式不正确",
}


def send_sms_code(phone: str, code: str) -> Tuple[bool, str]:
    username = settings.smsbao_username
    password = settings.smsbao_password

    if not username or not password:
        return False, "短信宝配置未完成"

    message = f"{settings.smsbao_sign}您的验证码是{code}"
    encoded_message = quote(message)
    url = f"{SMSBAO_API}?u={username}&p={password}&m={phone}&c={encoded_message}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        result = response.text.strip()
        if result == "0":
            return True, "验证码发送成功"
        return False, SMSBAO_ERROR_MESSAGES.get(result, f"短信发送失败，错误码: {result}")
    except requests.RequestException as exc:
        return False, f"短信发送失败: {exc}"
