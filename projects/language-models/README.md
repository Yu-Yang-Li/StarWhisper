# 天文语言模型训练数据

本目录保存 StarWhisper 3 的中英文训练问答数据，可以直接阅读 JSON 文件。这里没有训练或对话程序；使用模型时，还需获取权重并准备对应的推理环境。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [StarWhisper 3 权重](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)

## 数据内容

| 文件 | 内容 |
| --- | --- |
| [`astro_cn_1.json`](astro_cn_1.json) … `astro_cn_6.json` | 中文天文问答，共 6 份 |
| [`Astro_en.json`](Astro_en.json) | 英文天文问答 |
| [`Physic.json`](Physic.json) | 物理问答 |

这些文件是清洗后的问答文本，用于语言模型训练。查看内容不需要 GPU 或其他数据；重新训练时，需要自行选择基座模型、训练框架和数据划分方式。本目录没有附带这些设置，不能仅凭这几份 JSON 复现完整训练。

## 使用模型

模型权重发布在魔搭 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)，下载和加载方法以该页面的说明为准。本目录未提供 StarWhisper 4.0 权重。

观测数据和调度实验另见 [NGSS](../ngss/README.md)、[全天相机](../allsky-camera/README.md)与 [Explore](../explore/README.md)。
