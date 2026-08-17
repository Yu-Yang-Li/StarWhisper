"""
增强版AI客户端，具有更好的错误处理和备用方案
"""
import os
import json
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import requests

# 尝试导入不同的AI服务库
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: 未找到dashscope库，将无法使用通义千问服务")

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    print("警告: 未找到langchain-openai库，将无法使用OpenAI服务")

try:
    from langchain_community.llms import Tongyi
    LANGCHAIN_TONGYI_AVAILABLE = True
except ImportError:
    LANGCHAIN_TONGYI_AVAILABLE = False
    print("警告: 未找到langchain-community库，将无法使用通义千问LangChain服务")


class AIProvider(ABC):
    """AI服务提供者抽象基类"""

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """调用AI服务"""
        pass


class DashScopeAI(AIProvider):
    """通义千问AI服务提供者"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-max"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("DashScope API密钥未设置")
        dashscope.api_key = self.api_key

    def call(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """调用通义千问服务"""
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("DashScope库不可用")

        try:
            from dashscope import Generation
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                **kwargs
            )
            if response.status_code == 200:
                return response.output.text
            else:
                raise Exception(f"AI请求失败: {response.code}, {response.message}")
        except Exception as e:
            raise e


class OpenAIAI(AIProvider):
    """OpenAI服务提供者"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")

    def call(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """调用OpenAI服务"""
        if not LANGCHAIN_OPENAI_AVAILABLE:
            raise ImportError("langchain-openai库不可用")

        try:
            llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url
            )

            # 可以根据需要添加更多参数
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            raise e


class CustomAPIAI(AIProvider):
    """自定义API服务提供者"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str = "custom"):
        self.api_key = api_key or os.getenv("CUSTOM_API_KEY")
        self.api_url = api_url or os.getenv("CUSTOM_API_URL")
        self.model = model
        if not self.api_key:
            raise ValueError("Custom API密钥未设置")
        if not self.api_url:
            raise ValueError("Custom API URL未设置")

    def call(self, prompt: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """调用自定义API服务"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        # 添加其他可能的参数
        data.update(kwargs)

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                try:
                    result = response.json()
                    # 检查响应是否为HTML（错误页面）
                    if isinstance(result, str) and '<!DOCTYPE' in result:
                        raise Exception("API返回了HTML页面，可能是错误页面或配置问题")
                    
                    # 根据API响应格式提取内容
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0].get("message", {}).get("content", "")
                    elif "data" in result:
                        return result.get("data", "")
                    elif "content" in result:
                        return result.get("content", "")
                    else:
                        return str(result)
                except json.JSONDecodeError:
                    # 检查响应是否为HTML
                    response_text = response.text
                    if '<!DOCTYPE' in response_text or '<html' in response_text:
                        raise Exception(f"API返回了HTML页面，状态码: {response.status_code}")
                    # 如果响应不是JSON格式，返回原始文本
                    return response_text
            else:
                raise Exception(f"API请求失败: {response.status_code}, {response.text}")
        except Exception as e:
            raise e


class EnhancedAIClient:
    """增强版AI客户端，提供统一的AI服务接口，具有更好的错误处理"""

    def __init__(self, provider_type: str = "dashscope", **config):
        """
        初始化AI客户端

        Args:
            provider_type: AI服务提供商类型 ("dashscope", "openai", "custom")
            **config: 配置参数，如api_key, model等
        """
        self.provider_type = provider_type.lower()
        self.config = config

        if self.provider_type == "dashscope":
            if not DASHSCOPE_AVAILABLE:
                raise ImportError("DashScope库不可用，无法使用dashscope服务")
            self.provider = DashScopeAI(
                api_key=config.get("api_key"),
                model=config.get("model", "qwen-max")
            )
        elif self.provider_type == "openai":
            if not LANGCHAIN_OPENAI_AVAILABLE:
                raise ImportError("langchain-openai库不可用，无法使用openai服务")
            self.provider = OpenAIAI(
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-3.5-turbo"),
                base_url=config.get("base_url")
            )
        elif self.provider_type == "custom":
            self.provider = CustomAPIAI(
                api_key=config.get("api_key"),
                api_url=config.get("api_url"),
                model=config.get("model", "custom")
            )
        else:
            raise ValueError(f"不支持的AI服务提供商: {provider_type}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成AI响应

        Args:
            prompt: 提示词内容
            **kwargs: 其他参数，会传递给底层AI服务

        Returns:
            str: AI生成的响应内容
        """
        try:
            response = self.provider.call(prompt, **kwargs)
            return response if response is not None else "AI服务返回空响应"
        except Exception as e:
            error_msg = f"AI调用错误: {str(e)}"
            print(error_msg)
            return error_msg

    def extract_info(self, text: str, instruction: str) -> str:
        """
        从文本中提取特定信息

        Args:
            text: 要分析的文本
            instruction: 提取指令

        Returns:
            str: 提取的信息
        """
        prompt = f"请根据以下要求从文本中提取信息：{instruction}\n\n文本内容：\n{text}"
        return self.generate(prompt)

    def classify_text(self, text: str, categories: list) -> str:
        """
        对文本进行分类

        Args:
            text: 要分类的文本
            categories: 分类列表

        Returns:
            str: 分类结果
        """
        prompt = f"请将以下文本分类到最合适的类别中：\n类别：{', '.join(categories)}\n\n文本：{text}"
        return self.generate(prompt)


class EnhancedSimpleAIClient:
    """增强版简化AI客户端，只接受提示词，使用默认配置，具有备用方案"""

    def __init__(self):
        """使用环境变量或配置文件初始化AI客户端"""
        self.ai_client = None
        self._initialize_client()

    def _initialize_client(self):
        """根据环境变量或配置文件初始化AI客户端"""
        # 优先使用环境变量中的API密钥
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        custom_api_key = os.getenv("CUSTOM_API_KEY")
        custom_api_url = os.getenv("CUSTOM_API_URL")

        if dashscope_api_key:
            try:
                # 检查是否是有效的DashScope密钥格式
                if dashscope_api_key.startswith("sk-") and len(dashscope_api_key) > 10:
                    self.ai_client = EnhancedAIClient(
                        provider_type="dashscope",
                        api_key=dashscope_api_key
                    )
                    print("使用环境变量中的DashScope API密钥")
                    return
            except Exception as e:
                print(f"DashScope初始化失败: {e}")

        # 如果环境变量中没有DashScope密钥，尝试OpenAI环境变量
        if openai_api_key:
            try:
                self.ai_client = EnhancedAIClient(
                    provider_type="openai",
                    api_key=openai_api_key
                )
                print("使用环境变量中的OpenAI API密钥")
                return
            except Exception as e:
                print(f"OpenAI初始化失败: {e}")

        # 如果环境变量中没有OpenAI密钥，尝试自定义API环境变量
        if custom_api_key and custom_api_url:
            try:
                self.ai_client = EnhancedAIClient(
                    provider_type="custom",
                    api_key=custom_api_key,
                    api_url=custom_api_url
                )
                print("使用环境变量中的自定义API配置")
                return
            except Exception as e:
                print(f"自定义API初始化失败: {e}")

        # 如果环境变量不可用，尝试从配置文件读取
        try:
            # 从当前文件向上三级到达项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, "config", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 尝试使用配置文件中的api_key
                config_api_key = config.get("api_key")
                config_api_url = config.get("api_url")
                config_model = config.get("model", "qwen")

                if config_api_key and config_api_key != "your-api-key":  # 检查是否为默认值
                    # 如果有api_url，使用自定义API
                    if config_api_url and config_api_url != "your-api-url":
                        try:
                            self.ai_client = EnhancedAIClient(
                                provider_type="custom",
                                api_key=config_api_key,
                                api_url=config_api_url,
                                model=config.get("model_name", "custom")
                            )
                            print("使用配置文件中的自定义API配置")
                            return
                        except Exception as e:
                            print(f"自定义API初始化失败: {e}")
                            # 如果自定义API失败，尝试根据model_name判断使用哪种服务
                            model_name = config.get("model_name", "qwen").lower()

                            if "qwen" in model_name:
                                # 使用DashScope服务
                                try:
                                    self.ai_client = EnhancedAIClient(
                                        provider_type="dashscope",
                                        api_key=config_api_key
                                    )
                                    print("使用配置文件中的DashScope配置")
                                    return
                                except Exception as e2:
                                    print(f"DashScope服务初始化失败: {e2}")
                            else:
                                # 尝试OpenAI服务
                                try:
                                    self.ai_client = EnhancedAIClient(
                                        provider_type="openai",
                                        api_key=config_api_key
                                    )
                                    print("使用配置文件中的OpenAI配置")
                                    return
                                except Exception as e3:
                                    print(f"OpenAI服务初始化失败: {e3}")
                    # 检查API URL是否为标准的API端点而不是网页
                    elif config_api_url and "dmxapi.cn" in config_api_url:
                        print("检测到非标准API URL，可能无法正常工作")
                        # 尝试使用DashScope的标准API端点
                        standard_dashscope_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        try:
                            self.ai_client = EnhancedAIClient(
                                provider_type="custom",
                                api_key=config_api_key,
                                api_url=standard_dashscope_url,
                                model=config.get("model_name", "qwen-plus")
                            )
                            print("使用配置文件中的API密钥和标准DashScope端点")
                            return
                        except Exception as e:
                            print(f"使用标准DashScope端点失败: {e}")
                    else:
                        # 根据model_name判断使用哪种服务
                        model_name = config.get("model_name", "qwen").lower()

                        if "qwen" in model_name:
                            # 使用DashScope服务
                            try:
                                self.ai_client = EnhancedAIClient(
                                    provider_type="dashscope",
                                    api_key=config_api_key
                                )
                                print("使用配置文件中的DashScope配置")
                                return
                            except Exception as e:
                                print(f"DashScope服务初始化失败: {e}")
                        else:
                            # 默认使用OpenAI服务
                            try:
                                self.ai_client = EnhancedAIClient(
                                    provider_type="openai",
                                    api_key=config_api_key
                                )
                                print("使用配置文件中的OpenAI配置")
                                return
                            except Exception as e:
                                print(f"OpenAI服务初始化失败: {e}")
        except Exception as e:
            print(f"从配置文件初始化AI客户端失败: {e}")

        print("警告：未配置有效的AI服务，AI功能将不可用")

    def generate(self, prompt: str) -> str:
        """
        生成AI响应

        Args:
            prompt: 提示词内容（用户只需要提供这个）

        Returns:
            str: AI生成的响应内容
        """
        if not self.ai_client:
            return "错误：AI服务未配置有效密钥"

        try:
            result = self.ai_client.generate(prompt)
            # 检查结果是否为HTML页面
            if result and ("<html" in result or "<!DOCTYPE" in result):
                return "错误：AI服务返回了HTML页面，可能是API配置问题"
            return result if result is not None else "AI服务返回空响应"
        except Exception as e:
            return f"AI调用错误: {str(e)}"


# 创建全局增强版AI客户端实例（延迟初始化）
def get_global_enhanced_ai_client():
    """获取全局增强版AI客户端实例"""
    if not hasattr(get_global_enhanced_ai_client, '_instance'):
        get_global_enhanced_ai_client._instance = EnhancedSimpleAIClient()
    return get_global_enhanced_ai_client._instance


def enhanced_ai_generate(prompt: str) -> str:
    """
    便捷函数：使用增强版AI生成响应

    Args:
        prompt: 提示词内容

    Returns:
        str: AI生成的响应内容
    """
    client = get_global_enhanced_ai_client()
    return client.generate(prompt)