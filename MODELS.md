# 模型清单与下载地址（含镜像）

本文件列出 V-DriftBench 实验用到的所有模型、权重下载地址（HuggingFace 原始地址 + 国内可用的
hf-mirror 镜像），以及被测目标模型（API 类型，无权重）。

> 本实验环境无法直连 `huggingface.co`（被墙），全部通过镜像 `https://hf-mirror.com` 下载。
> 下载脚本：`python scripts/deploy/download_models.py --all`（内部默认 `HF_ENDPOINT=https://hf-mirror.com`）。

## 模型总览

| 角色 | 模型 ID | 类型 | 权重下载地址（HF 原始） | 镜像下载地址 |
| --- | --- | --- | --- | --- |
| 被测目标 | `deepseek-v4-flash` | API（无权重） | —（仅 API，见下） | — |
| 起草 draft | `ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ` | AWQ 4bit，~19.3GB | [huggingface.co/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ](https://huggingface.co/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ) | [hf-mirror.com/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ](https://hf-mirror.com/ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ) |
| 裁判 judge | `mistralai/Mistral-7B-Instruct-v0.3` | bf16，~15GB | [huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [hf-mirror.com/mistralai/Mistral-7B-Instruct-v0.3](https://hf-mirror.com/mistralai/Mistral-7B-Instruct-v0.3) |
| 合理性 plausibility | 复用 draft 模型 | — | 同上 | 同上 |
| Embedding | `BAAI/bge-m3` | sentence-transformers | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | [hf-mirror.com/BAAI/bge-m3](https://hf-mirror.com/BAAI/bge-m3) |

## 各角色说明

### 1. 被测目标模型（target）— API 类型

- 模型名：`deepseek-v4-flash`（DeepSeek V4 Flash）
- 调用方式：OpenAI 兼容 API，`base_url` 默认 `https://api.deepseek.com`
- 鉴权：环境变量 `DEEPSEEK_API_KEY`
- 无权重文件；如需研究其行为，通过 DeepSeek 开放平台申请 API Key。

### 2. 起草模型（draft）— AWQ 量化版

- 模型 ID：`ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ`
- 说明：`huihui-ai/Qwen2.5-32B-Instruct-abliterated` 的社区 AWQ 4bit 量化版（~19.3GB，匹配单卡 48GB 部署预算）。
  原 HANDOFF 4.1 指定的 `huihui-ai/Qwen2.5-32B-Instruct-abliterated` 只提供 bf16 全量权重（~65GB），单卡放不下，故改用此量化版。
- 部署：`scripts/deploy/serve_draft.sh`（vLLM，`--quantization awq`，served-model-name `draft-model`）

### 3. 裁判模型（judge）

- 模型 ID：`mistralai/Mistral-7B-Instruct-v0.3`
- 说明：HANDOFF 4.1 原选 `meta-llama/Llama-3.1-8B-Instruct`（gated，需 HF token 授权），部署时无可用 token，改用无门控、同量级的 Mistral-7B-Instruct-v0.3（与 Qwen 系 draft、常见远程 target 不同模型家族，保持"裁判≠目标家族"约束）。
- 部署：`scripts/deploy/serve_judge.sh`（vLLM，served-model-name `judge-model`）

### 4. 合理性模型（plausibility）

- 复用 draft 模型（`ibrahimkettaneh/Qwen2.5-32B-Instruct-abliterated-pass2-AWQ`），不单独部署。
- `main.py` 参数：`--plausibility-model draft-model`

### 5. Embedding 模型

- 模型 ID：`BAAI/bge-m3`
- 通过 `sentence-transformers` 本地加载（`--embedding-model BAAI/bge-m3`）。

## 下载命令

```bash
# 全部下载（embedding + draft + judge），默认走 hf-mirror 镜像
python scripts/deploy/download_models.py --all

# 只下部分
python scripts/deploy/download_models.py --only embedding,judge
python scripts/deploy/download_models.py --only draft
```

默认缓存目录 `/hy-tmp/hf_cache`，可通过 `--cache-dir` 覆盖；镜像通过环境变量
`HF_ENDPOINT=https://hf-mirror.com` 指定（脚本已内置默认值）。
