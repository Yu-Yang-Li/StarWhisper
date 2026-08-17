"""
AI服务模块 - 为PaperChecker提供AI功能支持

该模块提供了一个通用的AI接口，允许用户仅需提供提示词内容，
即可使用已配置的AI引擎（支持DashScope和OpenAI）。
"""

from .ai_client import AIClient, SimpleAIClient, ai_generate

__all__ = ['AIClient', 'SimpleAIClient', 'ai_generate']