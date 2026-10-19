# Contributing to StarWhisper

欢迎报告问题、补充复现记录或提交改进。提交前请阅读[对应项目](README.md#项目)的 README；已公开的数据和复现说明见[研究文档](docs/research.md)。

目录职责和依赖关系见[仓库结构](docs/architecture.md)。科研代码与数据处理放在 `projects/<项目>/`；技能的简短用法、按需读取的细则和实际脚本分别维护在 `SKILL.md`、`references/` 与 `scripts/`。运行日志与编译缓存不提交到仓库。安装和环境设置统一写在 [skills/INSTALL.md](skills/INSTALL.md)。

## 报告问题

请在 [Issue](https://github.com/Yu-Yang-Li/StarWhisper/issues) 中提供：对应目录、运行命令、Python / 操作系统版本、预期行为和实际输出。分享日志前请移除密钥、账号和未公开的观测数据。

## 提交修改

- 一次 Pull Request 尽量解决一个问题，说明改动目的和实际验证方式。
- 文档修改请同步相关的中英文入口，检查相对链接与图片能否在当前分支打开。
- 代码修改请运行对应目录的检查；本仓库原生技能的回归测试见下方命令。
- 科学结果请注明数据来源、样本范围和比较条件，区分历史预测、合成实验、真实观测与计划中的能力。
- 保留原有作者和许可证信息；改编技能的来源见 [skills/NOTICE.md](skills/NOTICE.md)。

原生技能测试（从仓库根目录执行，需要 `pytest`）：

```bash
python -m pytest skills/tests skills/starwhisper-snclock/tests skills/starwhisper-explore/tests skills/starwhisper-night-plan/tests skills/starwhisper-varlen/tests -q
git diff --check
```

涉及 NGSS 的修改请注明哪些外部接口已验证。观测设备的联锁与台站操作规则应始终优先于智能体建议。
