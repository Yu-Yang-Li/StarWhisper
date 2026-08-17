import os
from typing import Optional
try:
    from pydantic import BaseSettings
except ImportError:
    from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务器配置
    server_host: str = "0.0.0.0"
    server_port: int = 8002
    server_reload: bool = False
    
    # 临时文件目录
    temp_dir: str = "temp_uploads"
    
    # 最大文件上传大小 (bytes) - 10MB
    max_upload_size: int = 10 * 1024 * 1024
    
    # API配置
    api_prefix: str = "/api"

    # 账号系统配置
    auth_enabled: bool = False
    auth_db_path: str = "data/paperchecker_auth.db"
    auth_jwt_secret: str = "paperchecker-change-me"
    auth_access_token_ttl_minutes: int = 60 * 24  # 1天
    auth_code_expire_minutes: int = 5
    auth_code_cooldown_seconds: int = 60
    auth_dev_mode: bool = True
    auth_require_sms_code: bool = False

    # 短信宝配置
    smsbao_username: Optional[str] = None
    smsbao_password: Optional[str] = None
    smsbao_sign: str = "【短信宝】"
    
    class Config:
        env_file = ".env"

settings = Settings()
