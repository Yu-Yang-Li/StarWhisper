# StarWhisper 资产地图

这些线**没有配套技能**，因为在这个仓库里它们是可读的材料，不是可跑的流程。被路由到这里时，读对应目录再回答，不要假装跑过。

| 线 | 位置 | 论文 / 权重 | 边界 |
| --- | --- | --- | --- |
| 语言模型 | `LLM_Data/` | [StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) | 清洗后的问答文本，不是观测日志；4.0 未发布 |
| 光变分类 | `StarWhisper_LC/` | [Intelligent Computing 2025](https://spj.science.org/doi/10.34133/icomputing.0110) | 测试代码，不是完整训练复现；约 90% 准确率是论文口径 |
| 脉冲星 | [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) | [NeurIPS 2024 FM4Science](https://openreview.net/pdf?id=8SKgWpZiDL) | 未在本仓库 vendored；候选分类不是确认星表 |
| 全天相机 | `AllSky-Camera-XL/` | 打包技能见该目录 `skill/photo-to-replan/` | 流水线代码不在本仓库树内；产出的是序列文件，不是已执行的夜次 |
| 稀疏光变 | `Early Classification from Sparse Light Curves/` | [HF castor0705](https://huggingface.co/castor0705/sparse-lc-early-classification) | varlen 3–30 点、7 类、75/10/15、`random_state=42`；测试集指标不是爆发时刻 |
| 低信噪光谱 | `Low-SNR-Stellar-Spectra-as-Language/` | [HF Jaredxjc](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language) | 训练代码已公开，完整 tokenized 数据集仍是 coming soon；生成谱不是新观测 |
| GOTTA 样机 | `GOTTA_Prototype/` | [Virtual-GOTTA 地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) | 样机分数不是 broker 警报，也不是光谱分类 |
| 虚拟司天工作流 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | 同上 | `snc-*` 技能装在那边跑，不要在 StarWhisper 里重写 |

稀疏克隆里这些目录可能只有 README 甚至完全没有。缺了就说缺了。

已经有技能能跑的线不在这张表里：SN Clock 候选筛选、Explore 门槛判定、NGSS 夜计划检查、ADS 文献检索。
