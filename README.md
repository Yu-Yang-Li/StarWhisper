# 星语StarWhisper

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

🤖 <a href="https://modelscope.cn/models/AstroYuYang/StarWhisper">ModelScope



在国家天文台人工智能工作组的支持下，基于天文大模型StarGLM开发经验，我们进一步训练了星语StarWhisper系列模型(包括6B,7B,13B,14B,20B)。 

以进一步缓解大模型在天文通用知识的幻觉现象，为接下来可处理天文多模态任务、部署于望远镜阵列的科学具身智能——司天大脑打下基础。

## 版本更新：

1.通过清洗订正科普、科研数据飞轮得到的数据，改进训练方法，进一步提升了模型的天文物理、代码与Agent能力。

2.添加了专业英文语料，并进行英文反馈对齐，显著提升模型双语处理专业问题的能力。

3.发布了天文AI绘画的LORA权重，通过BLIP2+多模态模型+专家订正标注，可基于基模型输出星空图像，也可在其它模型上基于星空光效风格进行AI绘画生成。

V1版本在CG-Eval评测上的结果，总排名达到第二，仅次于GPT-4，数学推理和天文能力接近或超过GPT 3.5 Turbo。

完整榜单：http://cgeval.besteasy.com/leaderboard.html

## 功能展示

<div align=center><img src="example/starwhisper.png"/></div>

## 快速使用
模型已上传魔搭社区：https://modelscope.cn/models/AstroYuYang/StarWhisper

下面是一个使用StarWhisper模型，进行多轮对话交互的样例：

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("AstroYuYang/StarWhisper", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("AstroYuYang/StarWhisper", device_map="auto", trust_remote_code=True).eval()

# 在cpu上推理
# model = AutoModelForCausalLM.from_pretrained("AstroYuYang/StarWhisper", device_map="cpu", trust_remote_code=True).eval()

# model.generation_config = GenerationConfig.from_pretrained("AstroYuYang/StarWhisper", trust_remote_code=True) # 可指定不同超参

# 1st
response, history = model.chat(tokenizer, "你好", history=None)
print(response)

# 2nd
response, history = model.chat(tokenizer, "什么是黑洞？", history=history)
print(response)

```

## 司天工程

司天工程是我国天文学家面向时域天文学所提出的“十五五”天文重大基础设施，一期计划在国内多个优选观测台址布置54台（18组）口径1米级的大视场望远镜，组成多波段同时监测网络，每30分钟完成1万平方度天区的高精度三色“凝视”巡天。司天的采样频率比全球其它巡天项目高近两个量级，将突破目前探测时标的限制，在新的空域和时域下发现大批新天体、新现象，在宇宙极端高能爆发源、引力波电磁对应体、系外行星和太阳系天体等理论和观测研究中形成新的突破，在“两暗一黑三起源”等重大科学问题研究以及地球文明灾难预警等国家空间安全问题方面发挥重要作用。

<div align=center><img src="example/sitian.png"/></div>

其中司天"大脑"作为数据智能处理中枢，需要适配于天文的AI工具。StarGLM作为其备选方案，在使用大模型整合天文知识的同时，探索多模态解决具体天文问题的可能性。
## 许可证信息

项目源码遵从Apache-2.0 license，ChatGLM2-6B、Qwen-14B Chat的模型权重使用需遵从相应许可。

## 使用/推荐的相关项目

- THUDM/ChatGLM2-6B: ChatGLM2-6B: An Open Bilingual Chat LLM | 开源双语对话语言模型 (github.com)
- qwen/Qwen-14B-Chat: 通义千问-14B（Qwen-14B） 是阿里云研发的通义千问大模型系列的140亿参数规模的模型。
- wenda-LLM/wenda: 闻达：一个LLM调用平台。目标为针对特定环境的高效内容生成，同时考虑个人和中小企业的计算资源局限性，以及知识安全和私密性问题 (github.com) 
- THUDM/VisualGLM-6B: Chinese and English multimodal conversational language model | 多模态中英双语对话语言模型 (github.com) 
- hiyouga/LLaMA-Factory: 基于 PEFT 的高效模型微调 (github.com)
- MeteorCollector/iris_AstroQnA_ZH: Astronomy Q-A pairs in simplified Chinese. (github.com)
- HIT-SCIR/huozi (github.com)
## To do list

### 大语言模型（科普方式）

- [ ]  在相关材料上进行二次预训练，扩充天文知识。
- 调整监督微调中，通用数据和专业数据的比例，缓解灾难性遗忘问题。
- 通过人工反馈的强化学习，进一步提升模型性能。
- 通过特定数据集微调，提升模型总结能力，进一步适配知识库。
- [ ]  完成司天-变星知识图谱，与模型链接，进一步降低变星领域的幻觉现象。

### 专业多模态（科研工具）

- [ ]  开源在变星光变曲线上训练的多模态微调权重。
- [ ]  进一步探索多模态模型在天文图像生成与识别上应用的可能性。


### 观测Agent（司天大脑）

- 提升模型在天文领域的编程能力。
- [ ]  在MiniSiTian/司天样机上，进行与天文环境交互的Agent探索工作。
- 考虑通过工具学习，链接天文专业工具。
- 尝试Agent相关工作，验证作为司天大脑备选方案的可行性。

## 引用
如果这篇工作对你有帮助，请引用：

@Misc{chatglm-for-variable-star,

  title = {StarGLM},
  
  author = {YuYang Li, CunShi Wang, MengWei Qu, Yu Bai, Roberto Soria, JiFeng Liu},
  
  howpublished = {\url{https://github.com/Yu-Yang-Li/StarGLM}},
  
  year = {2023}
  
}

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)
