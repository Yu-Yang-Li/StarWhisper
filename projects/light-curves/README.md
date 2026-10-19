# StarWhisper LC

Kepler / K2 光变分类的代码与结果，对应论文 [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110)（*Intelligent Computing*, 2025）。下载后可以阅读预测结果和对照模型脚本；重新运行模型需要额外准备数据与权重。

[项目目录](../README.md) · [研究文档](../../docs/research.md)

## 先看结果

[`Result/`](Result/) 保存 LLM、MLLM、LALM 的预测 JSONL，以及采样、周期、观测时间节省和相位重要性的 CSV。它们是论文相关输出，可以直接用文本或表格工具查看，无需启动模型。

## 重新运行需要准备什么

[`Code/`](Code/) 保存图像和序列对照模型的实验脚本，并非统一的训练或推理程序。所需输入没有随本目录提供：

| 脚本 | 输入与运行条件 |
| --- | --- |
| [`get_image.py`](Code/get_image.py)、[`get_CWT.py`](Code/get_CWT.py) | 按 `T2CEP`、`RR`、`HYB`、`GDOR`、`EB`、`DSCT` 分目录的光变 CSV；分别生成图像或小波图，输出目录也需先准备 |
| [`CNN.py`](Code/CNN.py) | 按类别存放的 `data/` 图像目录；使用 EfficientNet，并在构建时获取预训练权重 |
| [`Convnext.py`](Code/Convnext.py)、[`swin_transformer.py`](Code/swin_transformer.py) | `data/` 图像目录，以及脚本指定的 `convnext_tiny-983f1562.pth` 或 `swin_tiny_patch4_window7_224.pth` |
| [`lstm.py`](Code/lstm.py)、[`lstm+attention.py`](Code/lstm+attention.py) | `merged.csv`，第一列为标签，其余列为序列特征；脚本使用 CUDA 与 Optuna |

这些相对路径都按运行时的当前目录查找。若从 `Code/` 运行，就在该目录下放置对应输入，或先修改脚本中的路径。依赖因模型而异，包括 PyTorch / torchvision、pandas、NumPy，以及相应的 `efficientnet_pytorch`、`timm`、`optuna`、PyWavelets（`pywt`）或 OpenCV；本目录没有统一的环境清单。

运行模型脚本会开始训练并写入权重，不是只读取 `Result/`。现有材料未包含完整数据准备和语言模型调用流程，不能直接作为论文的完整复现包。

## 论文与相关项目

![StarWhisper LC 模型结构](../../docs/assets/starwhisper-lc.png)

脉冲星候选分类见独立工作 [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar)。稀疏 ZTF / ATLAS 光变的早期分类见 [Early Classification from Sparse Light Curves](../sparse-light-curves/README.md)，其数据设定与本目录不同。
