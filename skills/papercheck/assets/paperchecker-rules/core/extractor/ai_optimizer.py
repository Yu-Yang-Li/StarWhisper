import re
import os
import sys
import json

# 使用新的AI服务模块
try:
    from core.ai.ai_client import AIClient
    AI_CLIENT_AVAILABLE = True
except ImportError:
    AI_CLIENT_AVAILABLE = False
    print("警告: 未找到AI客户端模块，将使用基于规则的优化")

def optimize_citations_with_ai(citations_list, config=None):
    """
    使用AI优化引用列表，过滤非引用内容并提取核心作者年份信息
    
    Args:
        citations_list (list): 提取到的引用列表
        config (dict): 配置参数，包含API密钥等信息
        
    Returns:
        list: 优化后的引用列表
    """
    if not citations_list:
        return []
    
    # 从配置中获取API密钥和模型信息
    api_key = config.get("api_key") if config else None
    model_name = config.get("model_name", "qwen-plus") if config else "qwen-plus"
    model_type = config.get("model", "qwen")  # 支持 qwen 或 gpt
    
    # 如果AI服务可用且有有效的API密钥，则使用AI优化
    if AI_CLIENT_AVAILABLE and api_key and api_key != "your-api-key":
        try:
            # 创建AI客户端
            if model_type == "gpt":
                ai_client = AIClient(
                    provider_type="openai",
                    api_key=api_key,
                    model=model_name
                )
            else:  # 默认使用qwen
                ai_client = AIClient(
                    provider_type="dashscope",
                    api_key=api_key,
                    model=model_name
                )
            
            return _optimize_citations_with_ai_client(citations_list, ai_client)
        except Exception as e:
            print(f"AI优化失败: {e}")
            # 如果AI优化失败，回退到基于规则的优化
            return _optimize_citations_with_rules(citations_list)
    else:
        # 如果AI不可用，使用基于规则的优化
        return _optimize_citations_with_rules(citations_list)

def _optimize_citations_with_ai_client(citations_list, ai_client):
    """
    使用新的AI客户端优化引用列表
    """
    try:
        # 构造提示词，将所有引用打包处理
        citations_json = json.dumps(citations_list, ensure_ascii=False)
        prompt = f"""
请从以下JSON格式的引用列表中，对每个引用进行优化处理。

要求：
1. 提取标准的作者年份格式引用,去掉无关的内容
2. 如果文本中包含多个引用，请只提取最核心的一个
3. 请严格按照以下格式回答:
[作者（年份）, 作者（年份）, ...]
4. 麻烦你确保引用的个数不要发生改变，不要去重

输入JSON引用列表: {citations_json}

请严格按照以下格式回答:
[作者（年份）, 作者（年份）, ...]

例如:
["张三（2024）", "Smith（2024）", "Johnson & Smith（2024）", "无效引用"]
"""
        # 使用AI客户端调用API
        result = ai_client.generate(
            prompt=prompt,
            max_tokens=10000,  # 增加max_tokens以适应更多内容
            temperature=0.0   # 使用较低的温度以获得更确定的结果
        )
        
        # 打印AI返回的原始结果，用于调试
        print("AI返回的原始结果:")
        print(result)
        print("---")
        
        # 尝试解析返回的JSON列表
        try:
            # 提取方括号内的内容，更灵活地匹配
            match = re.search(r'\[(.*?)\]', result, re.DOTALL)
            if match:
                json_str = match.group(0)
                print(f"提取到的JSON字符串: {json_str}")
                
                # 尝试修复可能导致解析失败的常见问题
                # 替换中文引号为英文引号
                json_str = json_str.replace('""', '"')
                json_str = json_str.replace('"', '"')
                json_str = json_str.replace("'", '"')
                
                # 尝试解析JSON
                try:
                    result_list = json.loads(json_str)
                    print(f"成功解析JSON，结果类型: {type(result_list)}, 长度: {len(result_list) if isinstance(result_list, list) else 'N/A'}")
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试预处理字符串
                    print("直接JSON解析失败，尝试预处理字符串...")
                    # 移除可能的问题字符
                    import re as regex_module  # 使用别名避免与局部变量冲突
                    
                    # 首先尝试使用正则表达式从看起来像列表的字符串中提取内容
                    # 匹配引号内的内容和没有引号的内容
                    # 例如: [A（2023）, B（2024）] 或 ["A（2023）", "B（2024）"]
                    items = []
                    # 匹配带引号的项目
                    quoted_items = regex_module.findall(r'"([^"]*?)"', json_str)
                    if quoted_items:
                        # 如果找到了带引号的项目，直接使用
                        items = quoted_items
                    else:
                        # 如果没有找到带引号的项目，尝试提取不带引号的项目
                        # 识别格式为 "文本（年份）" 或 "文本 等（年份）" 的模式
                        unquoted_items = regex_module.findall(r'[^,\[\]]*?[^,\[\]]*?（\d{4}）|[^,\[\]]*?[^,\[\]]*?等（\d{4}）|[^,\[\]]*?[^,\[\]]*?et al\.?（\d{4}）', json_str)
                        if unquoted_items:
                            items = [item.strip() for item in unquoted_items if item.strip()]
                    
                    if items and len(items) == len(citations_list):
                        print(f"通过正则表达式提取到 {len(items)} 个项目")
                        result_list = items
                    else:
                        # 如果正则表达式方法失败，使用预处理方法
                        processed_str = regex_module.sub(r',\s*,', ',', json_str)  # 处理连续的逗号
                        processed_str = regex_module.sub(r',\s*\]', ']', processed_str)  # 处理末尾的逗号
                        processed_str = regex_module.sub(r'\[\s*,', '[', processed_str)  # 处理开头的逗号
                        
                        print(f"预处理后的字符串: {processed_str}")
                        
                        # 再次尝试解析
                        try:
                            result_list = json.loads(processed_str)
                            print(f"预处理后成功解析JSON")
                        except json.JSONDecodeError:
                            print(f"预处理后仍然JSON解析失败: {processed_str[:200]}...")
                            print("回退到逐个处理")
                            return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
                
                # 确保返回的是列表
                if not isinstance(result_list, list):
                    print(f"返回的不是列表格式，而是 {type(result_list)}: {result_list}")
                    print("回退到逐个处理")
                    return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
                    
                # 确保返回的列表长度与输入列表长度一致
                if len(result_list) == len(citations_list):
                    print(f"返回列表长度匹配，开始处理 {len(result_list)} 个结果")
                    optimized_citations = []
                    for i, citation in enumerate(result_list):
                        # 将非字符串类型的元素转换为字符串
                        if not isinstance(citation, str):
                            citation = str(citation)
                        
                        print(f"  处理结果[{i}]: {citation}")
                        
                        # 检查结果是否为有效的引用格式或"无效引用"
                        if citation == "无效引用":
                            # 对于无效引用，保留原始引用
                            optimized_citations.append(citations_list[i])
                            print(f"    无效引用，保留原始: {citations_list[i]}")
                        elif re.search(r'[^（(]+[（(]\d{4}[)）]', citation):
                            optimized_citations.append(citation)
                            print(f"    有效引用格式")
                        else:
                            # 如果格式不正确，保留原始引用
                            optimized_citations.append(citations_list[i])
                            print(f"    格式不正确，保留原始: {citations_list[i]}")

                    print(f"最终优化结果: {optimized_citations}")
                    return optimized_citations
                else:
                    # 如果长度不匹配，回退到逐个处理
                    print(f"返回的引用列表长度不匹配: 期望 {len(citations_list)}, 实际 {len(result_list)}，回退到逐个处理")
                    return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
            else:
                # 如果没有找到JSON格式的列表，回退到逐个处理
                print(f"未找到JSON格式的列表，AI返回的内容: {result[:200]}...")
                print("回退到逐个处理")
                return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，回退到逐个处理
            print(f"JSON解析失败: {e}")
            print(f"尝试解析的内容: {result[:500]}...")
            print("回退到逐个处理")
            return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
        except Exception as e:
            # 捕获其他可能的异常
            print(f"处理AI响应时出错: {e}")
            print(f"错误类型: {type(e).__name__}")
            print("回退到逐个处理")
            return _optimize_citations_with_ai_client_individual(citations_list, ai_client)
            
    except Exception as e:
        print(f"批量处理引用时出错: {e}")
        # 如果出错，回退到逐个处理
        return _optimize_citations_with_ai_client_individual(citations_list, ai_client)

def _optimize_citations_with_ai_client_individual(citations_list, ai_client):
    """
    使用新的AI客户端逐个优化引用列表（回退方案）
    """
    optimized_citations = []
    
    for citation in citations_list:
        try:
            # 构造提示词
            prompt = f"""
请从以下文本中提取标准的作者年份格式引用。如果文本中包含多个引用，请只提取最核心的一个。
如果文本不是有效的引用，请返回"无效引用"。

输入文本: {citation}

请严格按照以下格式回答:
作者（年份）

例如:
张三（2024）
Smith（2024）
Johnson & Smith（2024）
"""
            # 使用AI客户端调用API
            result = ai_client.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.0   # 使用较低的温度以获得更确定的结果
            )
            
            result = result.strip()
            print(f"逐个处理的AI结果: {result}")  # 添加调试信息
            
            # 检查结果是否为有效的引用格式
            if result != "无效引用" and re.search(r'[^（(]+[（(]\d{4}[)）]', result):
                optimized_citations.append(result)
            else:
                # 如果是无效引用或格式不正确，保留原始引用
                optimized_citations.append(citation)
                
        except Exception as e:
            print(f"处理引用 '{citation}' 时出错: {e}")
            # 如果出错，添加原始引用
            optimized_citations.append(citation)
    
    # 去重
    unique_citations = list(set(optimized_citations))
    
    # 按年份排序
    try:
        unique_citations.sort(key=lambda x: int(re.search(r'（(\d{4})）', x).group(1)))
    except:
        # 如果排序失败，保持原有顺序
        pass
    
    return unique_citations

def _optimize_citations_with_rules(citations_list):
    """
    使用严格的规则优化引用列表，过滤非引用内容
    这个版本更严格地过滤，模拟AI优化的效果
    
    Args:
        citations_list (list): 提取到的引用列表
        
    Returns:
        list: 优化后的引用列表
    """
    if not citations_list:
        return []
    
    optimized_citations = []
    
    # 定义需要严格过滤的关键词模式（包括上下文内容）
    filter_patterns = [
        r'.*等[，,].*（\d{4}）.*',           # "张三等（2020）"后面还跟有其他内容
        r'.*[，,].*等.*（\d{4}）.*',         # "张三，李四等（2020）"后面还跟有其他内容  
        r'^在.*',                           # "在...张三（2020）"
        r'.*张玉利.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*汤天波.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*赵若羽.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*葛文静.*（\d{4}）.*$',          # 包含上下文的引用
        r'.*等[，,]?.*（\d{4}）的',          # "张三等（2020）的..."
        r'.*认为.*（\d{4}）',                # "张三认为（2020）"
        r'.*[，,].*（\d{4}）$',             # 以年份结尾但前面有逗号（可能是句子的一部分）
        r'^根据.*（\d{4}）',                 # "根据张三（2020）"
        r'^基于.*（\d{4}）',                 # "基于张三（2020）"  
        r'^参考.*（\d{4}）',                 # "参考张三（2020）"
        r'^如.*（\d{4}）',                   # "如张三（2020）"
        r'^例如.*（\d{4}）',                 # "例如张三（2020）"
        r'.*指出.*（\d{4}）',                # "张三指出（2020）"
        r'.*提到.*（\d{4}）',                # "张三提到（2020）"
        r'.*发现.*（\d{4}）',                # "张三发现（2020）"
        r'.*提出.*（\d{4}）',                # "张三提出（2020）"
        r'.*等.*[，,].*（\d{4}）',           # "张三等，李四（2020）"这种错误格式
        r'.*和.*（\d{4}）',                  # "张三和（2020）"这种不完整格式
    ]
    
    # 严格的标准引用格式模式 - 只接受干净的引用格式
    # 中文格式：作者（年份）或 作者 等（年份）或 作者1 & 作者2（年份）
    # 英文格式：Author（Year）或 Author et al.（Year）或 Author1 & Author2（Year）
    valid_citation_pattern = r'^([A-Za-z\u4e00-\u9fff\s&.&＆等，,]+?)\s*[（(]\s*(\d{4})\s*[)）]$'
    
    for citation in citations_list:
        # 检查是否需要过滤（包含上下文的引用）
        should_filter = False
        for pattern in filter_patterns:
            if re.search(pattern, citation.strip()):
                should_filter = True
                break
        
        if should_filter:
            continue
        
        # 验证是否为标准引用格式
        match = re.match(valid_citation_pattern, citation.strip())
        if match:
            # 如果符合标准格式，保留
            author_part = match.group(1).strip()
            year_part = match.group(2)
            standard_citation = f"{author_part}（{year_part}）"
            optimized_citations.append(standard_citation)
        # 注意：这里不处理不符合标准格式的引用，直接跳过
    
    # 去重
    unique_citations = list(set(optimized_citations))
    
    # 按年份排序
    try:
        unique_citations.sort(key=lambda x: int(re.search(r'（(\d{4})）', x).group(1)))
    except:
        # 如果排序失败，保持原有顺序
        pass
    
    return unique_citations

# 兼容接口
def optimize_citations(citations_list, config=None):
    return optimize_citations_with_ai(citations_list, config)