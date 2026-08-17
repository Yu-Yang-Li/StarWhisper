"""
配置管理器的简化版本，用于独立测试程序
"""
import json
import os

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件，如果不存在则使用默认配置"""
        default_config = {
            "model": "qwen",
            "api_key": "your-api-key",
            "model_name": "qwen-plus",
            "api_url": None,
            "download_timeout": 60,
            "max_retries": 3,
            "retry_delay_min": 4,
            "retry_delay_max": 10,
            "analysis_mode": "subjective",
            "use_advanced_extraction": True
        }
        
        # 检查配置文件路径，如果不存在则尝试从config目录查找
        config_path = self.config_path
        if not os.path.exists(config_path):
            # 尝试在config目录中查找
            alt_path = os.path.join("config", self.config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                # 尝试相对路径到项目根目录的config目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                alt_path = os.path.join(project_root, "config", self.config_path)
                if os.path.exists(alt_path):
                    config_path = alt_path
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {config_path}: {e}")
        
        return default_config
    
    def get_config(self):
        """获取配置"""
        return self.config
    
    def update_config(self, new_config):
        """更新配置"""
        self.config.update(new_config)
    
    def save_config(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)