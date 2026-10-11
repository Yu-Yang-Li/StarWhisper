# 验收说明

## 当前定位

这个技能优先查询 NASA ADS 和 arXiv，集思谱接口用于补充检索。

请求格式见 [examples/request_matrix.json](examples/request_matrix.json)，使用方法见 [SKILL.md](SKILL.md)。

它提供以下功能：

- 把用户问题路由到合适的集思谱论文检索 POST 接口。
- 用 `scripts/ads_first_search.py` 优先检索天文文献，保留 ADS 或 arXiv 的实际来源。
- 在没有登录态时只构造 dry-run 请求和输出表，不擅自真实调用受限接口。
- 把结果整理成可核验的论文候选表和引用审计表。

## 必须通过

```powershell
python -m pytest tests/test_dry_run_paper_search.py
python -m pytest tests/test_progressive_paper_search.py
python scripts/dry_run_paper_search.py --mode oa --query "科研图像生成 文献检索 skill" --format end-to-end-example
python scripts/progressive_paper_search.py --mode arxiv-title --query "科研图像生成 文献检索 skill" --expand arxiv-abstract,oa --max-pages 2 --dry-run
```

预期：

- 5 个 mode 的 URL 和 JSON body 正确。
- 所有示例输出都声明 `no request was sent`。
- 不读取、不打印、不保存任何密钥。
- 不把模拟论文当作真实检索结果。

## 可交付文件

- `SKILL.md`
- `agents/openai.yaml`
- `scripts/dry_run_paper_search.py`
- `scripts/progressive_paper_search.py`
- `tests/test_dry_run_paper_search.py`
- `tests/test_progressive_paper_search.py`
- `examples/request_matrix.json`
- `examples/normalized_result_example.json`
- `examples/failure_response_examples.json`
- `examples/end_to_end_example.json`

## 后续真实测试边界

只有在用户明确提供可用登录态或正式接口鉴权方式时，才允许做真实接口测试。真实测试结果必须单独记录请求摘要、响应摘要、归一化结果和失败处理，不得写入明文 token。
