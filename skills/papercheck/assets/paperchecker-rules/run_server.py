import uvicorn
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # 使用app目录下的main.py中的app实例
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level="info"
    )