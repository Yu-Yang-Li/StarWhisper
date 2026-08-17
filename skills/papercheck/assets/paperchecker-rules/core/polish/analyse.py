from pathlib import Path
import json
from typing import Dict, Any

from core.ai.ai_client import SimpleAIClient
from .reviewer import Reviewer

def prompt_combination(essay_to_analyse: str) -> str:
    # 组合提示词
    PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
    basic_prompt = (PROMPTS_DIR / "basic_prompts.txt").read_text()
    polish_prompt = (PROMPTS_DIR / "polish_prompts.txt").read_text()
    full_prompt = """
你是一个论文写作专家，你的任务是分析论文，提出修改建议，并且以JSON格式输出
你需要完成两个任务，一个是检查论文的基本规范问题，另一个是对论文提出语言润色的建议
===== 任务 =====
任务一：基本规范问题
{basic_prompt}
任务二：润色建议
{polish_prompt}

===== 要求 =====
你必须输出一个JSON对象，不许输出任何额外内容
该对象必须满足以下的格式
其中，对于任务一（基本规范问题），全部放入“基本规范问题”对应的数组
对于任务二（润色建议），全部放入“润色建议”对应的数组
每个数组中包含多个对象，每个数组中的对象对应一条建议，严格按照所给的格式输出
{
    "基本规范问题": [
        {
            "原文": 存在问题的原文,
            "修改文段": 修改后的文段,
            "问题": 指出问题
        }
    ],
    "润色建议": [
        {
            "原文": 需要润色的原文,
            "修改文段": 润色后的文段,
            "建议": 简述润色建议与理由
        }
    ]
}
如果一个部分没有明显问题，该部分请输出空数组
注意！你的输出必须严格遵照以上JSON格式
除此之外不允许输出任何内容

===== 需要分析的论文 =====
{essay_to_analyse}
"""
    return full_prompt

def json_combination(result_list: list) -> Dict[str, Any]:
    final_result = {"基本规范问题": [], "润色建议": []}
    for obj in result_list:
        final_result["基本规范问题"].append(obj["基本规范问题"])
        final_result["润色建议"].append(obj["润色建议"])
    return final_result

def review_document(doc_path: str) -> Dict[str, Any]:

    # 尝试使用完整路径到配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.json")
    review = Reviewer(doc_path, config_path)
    essay_list = review.core()

    # 分段输出记录json
    ai_client = SimpleAIClient()
    dict_result = []

    for para in essay_list:
        result = ai_client.generate(para)
        dict_result.append(json.loads(result))

    return json_combination(dict_result)