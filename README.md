# StarGLM

## 项目描述

StarGLM：ChatGLM for Variable Star
我们整合了司天工程相关的语料数据与知识库资料，训练得到了天文大语言模型StarGLM（GLM for Variable Star）。
以期解决大语言模型在部分天文通用知识和前沿变星领域的幻觉现象，为接下来可处理天文多模态任务、部署于望远镜阵列的观测智能体——司天大脑打下基础。

## 功能展示

![监督微调](example/example1.png)

<br>

![链接知识库](example/example2.png)

<br>

![链接StableDiffusion](example/example3.png)

<br>

![多模态探索](example/example4.png)

<br>

![未来计划](example/example5.png)
## 安装指南

Checkpoint中保存了相应Lora文件，Model中保存了Lora导出的模型，可根据配置条件自行选择合适方式加载。

## 许可证信息

项目源码遵从Alpaca 2.0，ChatGLM2-6b的模型权重使用需遵从相应许可。

## 使用/推荐的相关项目

- [Wenda](https://github.com/wenda-LLM/wenda)
- [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B)
- [ChatGLM2-6B](https://github.com/thudm/chatglm2-6b)
## To do list

### 大语言模型（科普方式）

- [ ]  二次预训练，扩充天文知识。
- [ ]  调整SFT中，通用数据和专业数据的比例，缓解灾难性遗忘问题。
- [ ]  RLHF，进一步提升模型性能。
- [ ]  完善司天-变星知识图谱，与模型链接，进一步降低变星领域的幻觉问题。

### 专业多模态（科研工具）

- [ ]  继续训练输入/输出视觉编码器/解码器，提升在天文专业领域的多模态能力。
- [ ]  结合CodeGeeX2工作，提升模型在天文领域的编程能力。
- [ ]  考虑通过工具学习，链接相关天文工具。

### 观测Agent（司天大脑）

- [ ]  紧跟Agent相关工作，验证作为司天大脑备选方案的可行性。
