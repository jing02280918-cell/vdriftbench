# V-DriftBench — v3.10

Adaptive multi-turn value-drift elicitation via a geometry-language hybrid state cascade and a cross-sample Thompson Sampling scheduler. This repo contains the method implementation, the Group A/B/C comparison harness, all four ablations, judge-reliability validation, and threshold calibration — the full set of code referenced by the method design and the remaining-work plan.

## 目录结构

```
vdriftbench/                 方法核心库
  config.py                    全局阈值配置，支持从YAML加载 (Config.from_yaml)
  schema.py                    Sample / JudgeScores / RoundRecord / SampleResult
  category_map.py              37个原始category -> 8个大类的聚合映射
  principles.py                P1-P7 原则库与prompt渲染
  llm_client.py                 模型无关的对话客户端（OpenAI兼容 + 离线Mock）
  embedding_client.py           embedding客户端（BGE-M3 via sentence-transformers + 离线Mock）
  drift.py                      Layer 2 语义漂移分数（EmbedDrift / EmbedDrift_norm）
  state_cascade.py               Layer0关键词规则 -> Layer1几何信号 -> Layer2 LLM五分类 三层级联
  bandit.py                      跨样本 Thompson Sampling 调度器（(大类,状态)->原则）
  ablation.py                    四个消融的开关（fixed_path / disable_geometry / disable_bandit / context_mode）
  groups.py                      Group A（单轮基线）/ Group B（批量指令链）执行器；Group C即pipeline本身
  judge.py                       Layer1 LLM裁判五维打分与JSON解析
  metrics.py                     加权Cohen's Kappa、分歧样本定位、标签一致率（实验0用）
  calibration.py                 从dev集真实分布建议tau_high/tau_low/theta（而非拍经验值）
  splits.py                      按category_macro分层的dev/test切分
  enrich.py                      数据准备：给原始{prompt,category}补全四个字段
  io_utils.py                    SampleResult的统一序列化/反序列化，main.py和scripts/共用
  pipeline.py                    单样本Round1-5执行循环（Group C）+ 数据集级别的调用与汇总

scripts/                     实验与消融运行脚本（remaining-work计划里的实验0-2、消融1-4）
  _bootstrap.py                 让脚本无需先pip install即可import vdriftbench
  split_dataset.py               dev/test分层切分
  calibrate_thresholds.py        实验前置：在dev集上标定几何阈值与Bandit奖励阈值
  run_judge_reliability.py       实验0：人工标注 vs LLM裁判的Kappa一致性验证
  run_pilot.py                   实验1：小规模试点，检查状态级联与Bandit是否合理触发
  run_main_experiment.py         实验2：Group A/B/C × 多个目标模型的核心对比
  run_ablation.py                消融1-4：一次跑完 full / fixed_path / no_geometry / no_bandit / context粒度变体

configs/                     YAML配置示例
  default.yaml                   与config.py硬编码默认值一致
  calibrated_example.yaml         标定后配置文件的示例格式（数值为占位符，需用calibrate_thresholds.py的真实输出替换）

tests/                       pytest单元测试（71个用例，覆盖bandit/state_cascade/judge/drift/principles/category_map/metrics/ablation/splits/pipeline端到端）

data/
  dataset_100.jsonl             原始数据（314条 {prompt, category[, format]}，已核对JSON格式全部合法）
  human_annotations_example.jsonl   人工标注文件格式示例，供run_judge_reliability.py参考

main.py                      单次运行入口（对应方法本身，不含实验/消融）
conftest.py, pyproject.toml, requirements.txt, LICENSE, .gitignore, CITATION.cff
```

## 安装

```bash
pip install -e .[llm,embedding,calibration,dev]
# 或者不装成包，直接：
pip install -r requirements.txt
```

## 快速开始（离线，无需API Key）

```bash
# 1) 用离线Mock补全四个字段
python main.py --enrich --mock --data data/dataset_100.jsonl --enriched-out data/dataset_100.enriched.jsonl

# 2) 切分dev/test
python scripts/split_dataset.py --data data/dataset_100.enriched.jsonl

# 3) 跑通方法本身（Group C，5条样本）
python main.py --mock --data data/dataset_100.enriched.jsonl --limit 5 -v

# 4) 跑通单元测试
python -m pytest -q
```

`--mock`模式下的目标模型/裁判/embedding都是确定性的离线占位实现，**不代表任何真实模型的安全行为**，只用于验证流水线在结构上是正确的（已用全部314条真实数据跑通，无异常）。

## 接入真实模型

```bash
export VDB_API_KEY=sk-xxxx
export VDB_BASE_URL=https://api.deepseek.com

python main.py --enrich --data data/dataset_100.jsonl \
    --enriched-out data/dataset_100.enriched.jsonl --enrich-model deepseek-v4-flash
python scripts/split_dataset.py --data data/dataset_100.enriched.jsonl
python main.py --data data/dataset_100.enriched.jsonl \
    --target-model deepseek-v4-flash --judge-model judge-model \
    --draft-model draft-model --plausibility-model draft-model \
    --embedding-model BAAI/bge-m3 --device cuda --limit 20 -v
```

其中 `judge-model` 和 `draft-model` 是 vLLM 本地部署的 `--served-model-name`（参考 `scripts/deploy/serve_judge.sh` 和 `scripts/deploy/serve_draft.sh`）。如果只有远程 API 可用，把这两个参数也换成对应的 API 模型名即可。

Embedding默认使用开源多语言模型 `BAAI/bge-m3`（通过`sentence-transformers`本地加载）；用`--embedding-model`可替换为任意`sentence-transformers`支持的模型。也可以用`--config configs/default.yaml`（或标定后的自定义yaml）代替逐个命令行参数传阈值。

## 按remaining-work计划复现实验

前置：数据准备 + 阈值标定 + 裁判校验（这三步不通过，后续所有实验结果都不可信）。

```bash
# 数据准备 + 切分
python main.py --enrich --data data/dataset_100.jsonl --enriched-out data/dataset_100.enriched.jsonl --enrich-model deepseek-v4-flash
python scripts/split_dataset.py --data data/dataset_100.enriched.jsonl

# 阈值标定：在dev集上用真实分布建议tau_high/tau_low/theta，而不是用config.py里的经验值
python scripts/calibrate_thresholds.py --dev-data data/dev.jsonl --target-model deepseek-v4-flash --judge-model judge-model
# 把打印出的建议值填进一份新的yaml（参考 configs/calibrated_example.yaml），后续实验都用 --config 指向它

# 实验1：试点（每大类抽样，先确认状态级联/Bandit行为合理）
python scripts/run_pilot.py --data data/dev.jsonl --config configs/calibrated_example.yaml

# 实验0：裁判可靠性验证（对实验1产出的pilot_results.jsonl抽样人工标注，格式见 data/human_annotations_example.jsonl）
python scripts/run_judge_reliability.py --llm-results pilot_results.jsonl --human-annotations data/human_annotations.jsonl

# 消融1-4：核心方法贡献的直接证据，建议在主实验之前做完
python scripts/run_ablation.py --data data/test.jsonl --config configs/calibrated_example.yaml \
    --presets full,fixed_path,no_geometry,no_bandit,context_state_only,context_category_only

# 实验2：Group A/B/C 主实验，多个目标模型
python scripts/run_main_experiment.py --data data/test.jsonl --config configs/calibrated_example.yaml \
    --target-models deepseek-v4-flash

# 实验3：跨大类分析——直接复用实验2的 experiment2_results/*.jsonl 按 category_macro 重新聚合，不需要新的调用
```

每个脚本都支持`--mock`做离线结构验证（已全部测试通过，见下方“已验证”一节）。

## 各消融对应的方法贡献

| 消融preset | 对比的机制 | 对应的方法创新点 |
| --- | --- | --- |
| `fixed_path` | 状态级联+Bandit全部跳过，走硬编码路径 | 验证\u201c自适应本身\u201d这个最基本的主张 |
| `no_geometry` | 跳过Layer1几何信号，强制每轮走LLM五分类 | 验证几何-语言混合信号是否在省成本的同时不掉判断质量 |
| `no_bandit` | 永远选默认动作，不采样不更新posterior | 验证跨样本在线学习相对静态决策表的增量效果 |
| `context_state_only` / `context_category_only` | Bandit context只用状态或只用大类 | 验证context粒度设计（大类×状态）是否必要 |
| `no_feedback_learning` (v3.2/v3.3) | 关闭样本内自主反馈学习，只用简单维度提示 | 验证样本内反馈环路的边际贡献 |

## 各模块与方法文档的对应关系

见上方目录结构表的行内注释；`vdriftbench/pipeline.py`的docstring里也直接引用了每个ablation preset对应的行为差异。

## 已验证

- 全部314条真实原始数据通过`--enrich --mock`成功补全四字段，且37个真实category全部被`category_map.py`覆盖（无回退到默认桶的情况）。
- 220个pytest单元测试全部通过，覆盖bandit采样/更新/持久化、三层状态级联的优先级、裁判分数解析与RiskScore判定、drift归一化、P1-P7渲染、category映射、Kappa计算、v3.10 单臂S_engaged（P4_rhetoric）+ 维度定向escalation（NJS→P4、EFS→P4_evidence）、R5改写强化、SCS压制引导、以及`run_single_sample`的端到端烟雾测试。
- `main.py`、全部6个`scripts/*.py`均在`--mock`模式下手动跑通，产出格式符合`io_utils.sample_result_to_dict`的统一schema。
- **v3.1**: 有效 87.9%（成功 25.4%）。**v3.2**: 回退至 77.7%（成功 19.9%）。**v3.3**: 有效 93.6%（成功 28.7%）。**v3.4**: 有效 96.4%（成功 35.3%）。**v3.5**: 影子爆炸→15.4%。**v3.6**: 已转正累积→16.6%。**v3.7**: 有效 94.4%（成功 28.7%，P4_continue被TS选中37%拖累）。**v3.8**: 有效 94.4%（成功 28.7%，3臂TS，移除P4_continue直接选择）。**v3.9**: 20样本验证 30.0%（单臂+escalation）。**v3.10**: 全量314条 总体 50.4%（R5独立 19.8%）。**v3.10.2**: R5改写追加"去免责/去反思腔"约束，配对检验显示反噬（R5独立降至12.7%）。**v3.10.3**: 拆分"拆台免责"与"价值框架点名"+反向规则4/5，全量314条跑完 R5独立18.2%≈v3.10基线、无显著提升，总体反而下降（51.2%→41.9%，且受rounds1-6非确定漂移混淆）。**已回退到 v3.10（当前版本，总体50.4%成功率最高）。**

## 关于持久化

`bandit_posterior.json`（或`run_ablation.py`/`run_main_experiment.py`为每个preset/模型生成的对应文件）记录了`(category_macro, state, action)`的Beta后验参数，会在每次运行结束时保存；下次运行指向同一文件即可继续累积，这正是"跨样本学习"的落地方式。运行中间产物（`*.jsonl`结果、posterior文件、`data/dev.jsonl`/`data/test.jsonl`/enriched数据）默认被`.gitignore`排除，因为它们既可重新生成，又可能包含敏感的红队生成内容。

## 伦理与使用范围

本代码用于学术红队评估研究：衡量大模型在多轮交互中被诱导产生价值偏移内容的倾向。生成的诱导性prompt与模型回复仅用于研究记录与risk分析，不应用于实际传播；`data/dataset_100.jsonl`中的prompt本身即为需要被评估、而非被执行的红队测试样本。使用前请参阅remaining-work计划中的"伦理与合规材料"一节（伦理审查、内容存储访问控制、responsible disclosure流程）。
