import docx

class Reviewer:
    def __init__(self, doc_path: str, config_path: str = "config.json"):
        self.doc_path = doc_path
        self.doc = docx.Document(doc_path)
        self.full_text_str = []

        # 读取配置文件
        try:
            from config.config_manager import ConfigManager
            # 确保配置路径正确，处理相对路径和绝对路径

            original_config_path = config_path  # 保存原始路径用于错误消息
            config_manager_initialized = False
            
            if not os.path.isabs(config_path):
                # 如果是相对路径，尝试在当前目录和config目录中查找
                if not os.path.exists(config_path):
                    # 尝试在项目根目录的config子目录中查找
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    config_in_config_dir = os.path.join(project_root, "config", config_path)
                    if os.path.exists(config_in_config_dir):
                        config_path = config_in_config_dir
                    else:
                        # 如果配置文件不存在，使用默认配置
                        print(f"配置文件不存在: {original_config_path}, 使用默认配置")
                        self.config = self._get_default_config(config_path)
                        self.config_manager = None
                        config_manager_initialized = False
                else:
                    config_manager_initialized = True
                    self.config_manager = ConfigManager(config_path)
                    self.config = self.config_manager.get_config()
                    
                # 如果在config目录中找到了文件
                if os.path.exists(config_path) and not config_manager_initialized:
                    config_manager_initialized = True
                    self.config_manager = ConfigManager(config_path)
                    self.config = self.config_manager.get_config()
            else:
                # 绝对路径
                config_manager_initialized = True
                self.config_manager = ConfigManager(config_path)
                self.config = self.config_manager.get_config()
                
            # 如果config_manager没有初始化，使用默认配置
            if not config_manager_initialized:
                self.config = self._get_default_config(original_config_path)
                self.config_manager = None
        except Exception as e:
            print(f"配置管理器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果config_manager不可用，使用默认配置
            self.config = self._get_default_config(config_path)
            self.config_manager = None

        # 初始化LLM
        self.llm = None
        self.model_type = None
        self._initialize_llm()
        
        # 配置文献获取参数
        self.download_timeout = self.config.get("download_timeout", 60)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay_min = self.config.get("retry_delay_min", 4)
        self.retry_delay_max = self.config.get("retry_delay_max", 10)
        
        # 配置分析模式
        self.analysis_mode = self.config.get("analysis_mode", "full")  # "full" or "quick" or "subjective"
        
        # 初始化进度跟踪
        self.total_citations = 0
        self.processed_citations = 0
        self.progress_file = "analysis_progress.json"

    # 提取文本
    def core(self) -> list:
        for para in self.doc.paragraphs:
            self.full_text_str.append(para.text)
        return self.full_text_str
    
    # 初始化大语言模型
    def _initialize_llm(self):
        """初始化语言模型 - 使用新的AI服务模块"""
        # 从ai模块导入AIClient
        try:
            from ai.ai_client import AIClient
            # 根据配置决定使用哪种AI服务
            if self.config["model"] == "gpt":
                if not self.config.get("api_key") or self.config.get("api_key") == "your-api-key":
                    print("警告：未设置有效的GPT_API_KEY，将跳过AI相关性分析")
                    self.ai_client = None
                else:
                    self.ai_client = AIClient(
                        provider_type="openai",
                        api_key=self.config.get("api_key"),
                        model=self.config.get("model_name", "gpt-3.5-turbo"),
                        base_url=self.config.get("api_url", "https://api.openai.com/v1")
                    )
                    self.model_type = "gpt"
                    self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
            elif self.config["model"] == "qwen":
                if not self.config.get("api_key") or self.config.get("api_key") == "your-api-key":
                    print("警告：未设置有效的QWEN_API_KEY，将跳过AI相关性分析")
                    self.ai_client = None
                else:
                    self.ai_client = AIClient(
                        provider_type="dashscope",
                        api_key=self.config.get("api_key"),
                        model=self.config.get("model_name", "qwen-plus")
                    )
                    self.model_type = "qwen"
                    self.model_name = self.config.get("model_name", "qwen-plus")
        except ImportError:
            print("警告：无法导入AI服务模块，将跳过AI相关性分析")
            self.ai_client = None
        except Exception as e:
            print(f"警告：AI客户端初始化失败: {e}")
            self.ai_client = None
