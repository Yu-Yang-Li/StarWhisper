# AI服务模块使用指南

PaperChecker的AI服务模块为开发者提供了一个统一、简便的AI接口，只需要定义提示词内容，即可使用已配置的AI引擎。

## 模块结构

```
ai/
├── ai_client.py     # AI客户端实现
└── __init__.py      # 模块初始化
```

## 快速开始

### 1. 简单调用（推荐）

对于基本使用，可以直接使用简化版客户端，只需提供提示词：

```python
from ai.ai_client import ai_generate

# 直接调用AI，只需提供提示词
result = ai_generate("请总结以下文本的要点...")
print(result)
```

### 2. 完整功能调用

如果需要更多控制，可以使用完整的AI客户端：

```python
from ai.ai_client import AIClient

# 初始化AI客户端
client = AIClient(
    provider_type="dashscope",  # 或 "openai"
    api_key="your-api-key",
    model="qwen-plus"           # 或其他模型名称
)

# 生成响应
response = client.generate("请分析以下学术文献的相关性...")

# 使用特定功能
info = client.extract_info("文本内容", "提取作者和年份信息")
category = client.classify_text("文本内容", ["学术", "新闻", "其他"])
```

## 环境变量配置

AI客户端会自动从环境变量读取API密钥：

```bash
# DashScope (通义千问)
export DASHSCOPE_API_KEY=your_dashscope_api_key

# OpenAI
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_API_BASE=https://api.openai.com/v1  # 可选，自定义API端点
```

## API提供商支持

### 1. DashScope (通义千问)

支持以下模型：
- `qwen-turbo`
- `qwen-plus`
- `qwen-max`
- `qwen-7b-chat`
- 其他通义系列模型

### 2. OpenAI

支持以下模型：
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`
- 以及其他GPT系列模型

## 使用场景示例

### 场景1：引用格式优化

```python
from ai.ai_client import ai_generate

def optimize_citation_format(raw_citation):
    prompt = f"""
    请将以下引用格式标准化为作者(年份)格式：
    
    原始引用：{raw_citation}
    
    请返回标准格式的引用，格式为：
    - 中文：张三（2024）
    - 英文：Smith（2024）
    """
    
    return ai_generate(prompt)

# 使用示例
result = optimize_citation_format("邹林林, 姚恩建, 潘龙.面向停取自平衡的共享单车补贴方案定价优化方法[J].交通工程, 2023")
print(result)  # 输出：邹林林（2023）
```

### 场景2：引文相关性分析

```python
from ai.ai_client import ai_generate

def analyze_citation_relevance(citation, context, reference):
    prompt = f"""
    请分析以下学术论文中的引用是否与上下文相关：
    
    引用编号：{citation}
    参考文献条目：{reference}
    引用上下文：{context}
    
    请严格按照以下格式回答：
    1. 相关性判断：相关 / 不相关
    2. 分析理由：详细说明引用与上下文的相关性
    3. 问题说明：如果相关则写"无"，如果不相关则指出问题
    """
    
    return ai_generate(prompt)

# 使用示例
relevance = analyze_citation_relevance(
    "邹林林（2023）", 
    "本文研究共享单车的运营模式", 
    "邹林林等关于共享单车补贴方案的研究"
)
print(relevance)
```

### 场景3：文献内容提取

```python
from ai.ai_client import SimpleAIClient

def extract_key_info(text):
    client = SimpleAIClient()  # 自动检测可用的AI服务
    
    prompt = f"""
    请从以下学术文献中提取以下信息：
    1. 作者姓名
    2. 发表年份
    3. 主要研究主题
    4. 研究方法
    
    文献内容：
    {text}
    """
    
    return client.generate(prompt)
```

## 高级功能

### 自定义参数

AI客户端支持传递额外的参数到底层模型：

```python
from ai.ai_client import AIClient

client = AIClient(provider_type="dashscope", api_key="your-key")

# 传递模型特定参数
response = client.generate(
    prompt="你的提示词",
    temperature=0.7,      # 创造性参数
    max_tokens=1000,      # 最大输出长度
    top_p=0.9             # 核采样参数
)
```

### 错误处理

```python
from ai.ai_client import ai_generate

def safe_ai_call(prompt):
    try:
        result = ai_generate(prompt)
        if "错误：" in result or "无法" in result:
            # AI调用失败，使用备用方案
            return fallback_solution(prompt)
        return result
    except Exception as e:
        print(f"AI调用出错: {e}")
        return fallback_solution(prompt)

def fallback_solution(prompt):
    # 实现备用逻辑
    return "AI服务不可用，使用备用方案处理"
```

## 最佳实践

1. **使用简化接口**：对于简单任务，优先使用 `ai_generate()` 函数
2. **错误处理**：始终包含错误处理逻辑
3. **提示词优化**：使用结构化和明确的提示词格式
4. **环境变量**：使用环境变量管理API密钥
5. **成本控制**：注意API调用次数和成本

## 集成到现有项目

要在现有代码中集成AI服务，只需导入并替换原有的AI调用：

```python
# 旧代码
# import dashscope
# response = dashscope.Generation.call(...)

# 新代码
from ai.ai_client import ai_generate
response = ai_generate("你的提示词")
```

这样，你的项目中就有了一个可扩展的AI服务模块，其他开发者只需要提供提示词内容即可使用AI功能。