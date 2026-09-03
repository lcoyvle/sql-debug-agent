# SQL Debug Agent

一个面向零基础学习者的、由数据库执行反馈驱动的 SQL 自纠错 Agent。项目使用虚构的金融数据，完整跑通：

```text
错误 SQL → 安全检查 → 数据库执行 → 结果验证 → 错误反馈 → 修正 → 再次验证
```

## 最终结果

项目已完整跑通 **Base → 数据构建 → SFT → Bad Case → GRPO → 冻结评测**：

| 模型 | 冻结测试集正确率 | 相对上一阶段 |
| --- | ---: | ---: |
| Qwen2.5-Coder-1.5B Base | 16/28（57.1%） | — |
| 60 步 QLoRA SFT | 23/28（82.1%） | +25.0 pp |
| SFT + 40 步 GRPO | **25/28（89.3%）** | **+7.1 pp** |

GRPO 修复 2 题、回退 0 题。评测使用训练前冻结的 28 道题、确定性生成和两套数据库
execution-match；完整实验解释见 [`docs/final_report.md`](docs/final_report.md)，机器可读
指标见 [`docs/final_metrics.json`](docs/final_metrics.json)，面试讲法见
[`docs/interview_guide.md`](docs/interview_guide.md)。

项目包含四个运行模式：

- `rule`：无需 API、GPU 和第三方依赖的 V0 教学基线。
- `ollama`：默认模式，在本机调用开源模型，不需要 API Key 或充值。
- `mlx`：Apple Silicon 上的 Base/LoRA 模型推理，用于严格比较 SFT 前后效果。
- `openai`：可选模式，通过 Responses API 调用闭源模型。

规则修复器只用于把 Agent 链路讲清楚，并不是训练好的大模型。真实模型与规则模式共用相同的数据库、验证器、奖励和评测代码。

## 已实现

- 虚构金融数据库：客户、账户、交易流水三张表
- 只读 SQL 安全检查，拦截删除、更新及多语句执行
- SQL 执行工具和 Schema 获取工具
- “可执行率”与“结果正确率”分离评测
- 最多三轮的反馈—修正 Agent 循环
- 五类 Bad Case：语法、字段匹配、聚合、重复计数、JOIN 类型
- 原始 Bad Case 到 SFT 对话数据的处理脚本
- JSON 评测轨迹、奖励和单元测试
- Ollama Chat API 与 OpenAI Responses API 结构化 SQL 修复器
- 与演示数据分离的 30 道模型基线评测集
- 按错误类型统计准确率和平均修正轮数
- 自动 Bad Case 分析与 SFT/GRPO 阶段决策
- 200 条由业务问题倒推、经 SQLite 验证的训练 Bad Case
- 按错误模板分组切分和独立测试集防泄漏检查
- 在 M1 8GB MacBook Air 上完成 MLX 4-bit LoRA SFT
- Base/SFT 同条件对比、回退检测与训练后 Bad Case 报告
- 35 道训练前冻结的最终留出集，以及 Base/SFT V1/SFT V2 三方对比
- 自动 RL 阶段门槛：未达到泛化标准时阻止盲目进入强化学习
- SFT V3 的多 checkpoint 开发集选型与执行等价评测修复
- 90 条 GRPO prompts、270 对偏好数据和多数据库防作弊奖励审计
- V3-30 的 90 条真实 rollout 回放和 26 条 hard preference pairs
- 从 V3 Adapter 继续训练的 corrective replay 负实验与自动候选拒绝
- 可上传 Colab 的标准 PEFT SFT → 双数据库 GRPO 训练包
- 26 条真实失败难例、4-candidate GRPO 与奖励方差自动审计
- Base/SFT/SFT+GRPO 的 28 题冻结集最终对比：57.1% → 82.1% → 89.3%

## 项目状态：完整训练与最终验证已完成

本机 MLX Adapter 与云端 PyTorch/PEFT 格式不同，因此云端训练包会先用 V3 数据重建
标准 SFT Adapter，再在它上面运行真正的 GRPO。整个训练不调用 OpenAI API。

生成零基础 Colab 上传包：

```bash
.venv/bin/python cloud_grpo/prepare_bundle.py
```

输出为 `artifacts/sql-debug-agent-colab.zip`。完整的上传、SFT、GRPO 和结果下载步骤见
`cloud_grpo/README_COLAB.md`。训练包包含 218 条 SFT 训练数据、42 条内部验证数据、
90 条 GRPO prompts、26 条真实失败难例和两套奖励数据库，不包含最终 holdout。
最终 holdout 只在所有训练结束后通过独立评测包上传。

## 快速开始

规则演示要求 Python 3.10 或更高版本，不需要安装依赖。

```bash
cd /Users/chenyilin/Documents/ChatGPT/项目LLM/sql-debug-agent
python3 -m sql_debug_agent init-db
python3 -m sql_debug_agent eval
```

当前演示集的预期输出：

```text
任务数：5
基线准确率：0.0%
修正后准确率：100.0%
成功修复：5
```

这只是用于验证代码闭环的小型演示集，不能作为模型能力结论。下一阶段需要扩大数据量并设置真正隔离的测试集。

## 使用本地 Ollama 跑真实模型基线

本项目默认使用 `qwen2.5-coder:1.5b-instruct`。Ollama 只负责方便地运行基线；
后续 SFT 和 GRPO 会使用同一模型的原始开源权重。

1. 从 <https://ollama.com/download/mac> 下载并安装 Ollama。
2. 启动 Ollama 应用。
3. 在终端下载约 1 GB 的模型：

```bash
ollama pull qwen2.5-coder:1.5b-instruct
```

确认模型已经存在：

```bash
ollama list
```

先只评测 1 道题，检查整个连接和输出格式：

```bash
cd /Users/chenyilin/Documents/ChatGPT/项目LLM/sql-debug-agent
python3 -m sql_debug_agent baseline --limit 1
```

成功后运行默认的 5 道题，再运行完整的 30 道题：

```bash
python3 -m sql_debug_agent baseline
python3 -m sql_debug_agent baseline --limit 30
```

默认连接 `http://localhost:11434`。如果 Ollama 使用其他地址，可以设置：

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5-coder:1.5b-instruct"
```

只让本地模型修复一条 SQL：

```bash
python3 -m sql_debug_agent debug \
  --repairer ollama \
  --question "计算所有 debit 交易的平均金额" \
  --sql "SELECT AVG(amounts) FROM transactions"
```

本地模型基线可能低于大型闭源模型，这是后续用 SFT、Bad Case 分析和 GRPO
证明优化效果的起点，不应使用规则模式的 100% 结果替代它。

## 可选：使用 OpenAI API 跑基线

首先准备 OpenAI API Key。不要把 Key 发给别人，也不要写进代码或提交到 Git。当前终端可以这样设置：

```bash
read -s "OPENAI_API_KEY?请输入 API Key: "
export OPENAI_API_KEY
echo
```

首次只运行 5 道题：

```bash
python3 -m sql_debug_agent baseline --provider openai
```

默认使用 `gpt-5.6-luna`、每题最多两轮，因此首次评测每道题至多产生一次模型修复请求。确认流程正常后再运行完整 30 题：

```bash
python3 -m sql_debug_agent baseline --provider openai --limit 30
```

可以通过环境变量更换模型：

```bash
export OPENAI_MODEL="你的模型名称"
python3 -m sql_debug_agent baseline --provider openai
```

也可以只让真实模型修复一条 SQL：

```bash
python3 -m sql_debug_agent debug \
  --repairer openai \
  --question "计算所有 debit 交易的平均金额" \
  --sql "SELECT AVG(amounts) FROM transactions"
```

模型请求采用 Responses API 的 JSON Schema 结构化输出；实现见 `sql_debug_agent/openai_repair.py`。API Key 只从环境变量读取，不会写入评测报告。

## 修复单条 SQL

```bash
python3 -m sql_debug_agent debug \
  --question "计算所有 debit 交易的平均金额" \
  --sql "SELECT AVG(amounts) FROM transactions"
```

不提供参考 SQL 时，系统只能判断 SQL 是否安全、能否执行，不能证明业务语义正确。离线评测通过隐藏的参考 SQL 对比执行结果。

## 分析基线 Bad Case

完整本地基线已经得到 `16/30（53.3%）`。先运行分析，而不是直接训练：

```bash
python3 -m sql_debug_agent analyze-badcases
```

当前结论是 JOIN `0/5`，日期和过滤均为 `2/5`；14 个失败案例都没有被模型
真正改写。因此下一阶段先做 SFT，让模型学会稳定纠错，不直接做 RL。报告输出到：

- `artifacts/badcase_analysis.md`：适合阅读和面试展示。
- `artifacts/badcase_analysis.json`：供后续程序读取。

## 构建正式 SFT 数据

运行一条命令即可重建训练数据库、生成数据、执行验证、防泄漏并完成切分：

```bash
python3 -m sql_debug_agent build-data
```

这 200 条数据不是下载的开源数据。构造逻辑是：从基线短板倒推所需能力，先写
正确的金融分析 SQL，再通过可解释的错误模板注入 JOIN、日期、过滤、聚合等错误，
最后用 SQLite 确认正确 SQL 可执行、错误 SQL 的结果确实不正确。

当前分布：

| 错误类型 | 数量 | 为什么需要 |
| --- | ---: | --- |
| JOIN | 60 | 基线 0/5，是第一优先级 |
| Filter | 35 | 基线 2/5，容易造成错误报表 |
| Aggregation | 30 | 覆盖分组、粒度与聚合函数 |
| Date | 30 | 基线 2/5，覆盖月份和时间边界 |
| Schema Linking | 20 | 学习真实字段与错误字段的映射 |
| Duplicate Counting | 15 | 防止多表连接后重复计数 |
| Syntax Error | 10 | 基线已是 5/5，只保留少量巩固样本 |

产物包括：

- `data/train_finance.db`：只用于数据构建的扩展金融数据库。
- `data/sft_raw.jsonl`：200 条原始、可追踪 Bad Case。
- `artifacts/sft_v1/sft_train.jsonl`：158 条 SFT 训练数据。
- `artifacts/sft_v1/sft_eval.jsonl`：42 条内部评测数据。
- `artifacts/data_manifest.json`：数量、分布、验证与防泄漏记录。

切分先按错误类型分层，再按 `template_id` 分组；七类错误在两侧都有样本，但同一种
错误模板不会同时进入训练和内部评测。实际比例因此不会严格等于 80/20。原来的
30 题 `baseline_eval.jsonl` 始终作为最终独立测试集，不进入这 200 条训练数据。

## 在 M1 Mac 上运行 SFT

本项目使用 Apple MLX-LM 和约 880MB 的
`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`。首次安装训练环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train]'
```

先运行 5 步 smoke test，确认数据、Metal GPU 和 Adapter 加载都正常：

```bash
.venv/bin/python -m sql_debug_agent train-sft --smoke
```

再运行第一轮 50 步 SFT：

```bash
.venv/bin/python -m sql_debug_agent train-sft
```

针对本机 M1 8GB 的配置为：4-bit QLoRA、batch size 1、梯度累积 4、只训练
最后 4 层、最大长度 1024、梯度检查点。实测峰值内存约 1.831GB。模型缓存约
839MB，位于 `artifacts/hf_cache/`；不会使用 API，也不会产生充值费用。

## SFT V1 对比结果

为避免把接口差异误当成训练收益，Base 和 SFT 都通过相同的 MLX 推理代码、相同
prompt 和相同 30 题评测。因为这些题随后参与了 V1 Bad Case 分析，从 V2 开始它们
被严格定义为开发集，不再称为最终独立测试集：

| 模型 | 正确题数 | 准确率 |
| --- | ---: | ---: |
| MLX Base | 24/30 | 80.0% |
| SFT V1 | 25/30 | 83.3% |
| 净变化 | +1 | +3.3 个百分点 |

SFT 修复了 `eval_aggregation_04` 和 `eval_join_03`，但让 `eval_filter_04`
发生回退。JOIN 从 60% 提升到 80%，重复计数从 0% 提升到 100%，过滤从
100% 降到 80%。因此这一轮证明训练链路有效，但提升还不够稳定，下一步应做
SFT V2 的针对性数据优化，暂不进入 RL。

复现实验：

```bash
.venv/bin/python -m sql_debug_agent baseline \
  --provider mlx --limit 30 --output artifacts/mlx_base_report.json

.venv/bin/python -m sql_debug_agent baseline \
  --provider mlx --adapter-path artifacts/adapters/sft_v1 \
  --limit 30 --output artifacts/sft_v1_eval_report.json

.venv/bin/python -m sql_debug_agent compare-runs
```

完整实验解释见 `docs/sft_v1_report.md`，自动对比见
`artifacts/sft_v1_comparison.md`。

## SFT V2 与冻结留出集

V2 根据开发集剩余的 5 类错误补充 60 条针对性数据，训练集从 158 条增加到 218 条，
并给监督答案增加 SQL 分号终止协议。训练前先冻结了全新的 35 题最终留出集，保证 V2
数据模板不由最终答案倒推。

```bash
.venv/bin/python -m sql_debug_agent build-v2

.venv/bin/python -m sql_debug_agent train-sft \
  --sft-train artifacts/sft_v2/sft_train.jsonl \
  --sft-eval artifacts/sft_v2/sft_eval.jsonl \
  --data-dir artifacts/mlx_data_v2 \
  --adapter-path artifacts/adapters/sft_v2 \
  --sql-terminator
```

最终留出集结果并没有因为训练损失下降而提高：

| 模型 | 正确题数 | 准确率 |
| --- | ---: | ---: |
| MLX Base | 27/35 | 77.1% |
| SFT V1 | 26/35 | 74.3% |
| SFT V2 | 23/35 | 65.7% |

这暴露了小数据模板记忆和能力遗忘：V1 提升 JOIN、损害过滤；V2 又明显损害日期能力。
因此项目没有为了“凑齐流程”直接做 RL，而是增加了可执行阶段门槛：

```bash
.venv/bin/python -m sql_debug_agent summarize-experiment
```

只有 SFT 在冻结留出集超过 Base、没有任务回退、且后续 SFT 迭代不退化时，才允许进入
RL 奖励和数据设计。本轮三个条件均未通过。完整分析见
`docs/sft_v2_report.md`，自动汇总见 `artifacts/sft_final_summary.md`。

## SFT V3：从负结果继续优化

V3 不累加导致退化的 V2 增量数据，而是使用 158 条 V1 能力回放和 60 条新的多模板
样本，强调最小必要修改、SQLite 日期方言和不破坏 `LEFT JOIN`。训练 10/20/30 步
三个候选后，用开发集预先选择 V3-30，再只在新冻结的 28 题上运行一次最终对比。

最终结果：Base 18/28（64.3%），V3-30 19/28（67.9%），净提升 1 题。JOIN 从
25% 提升到 50%，重复计数从 0% 提升到 25%，但 Schema Linking 出现 1 题回退，
所以仍不进入 RL。

这一轮还修正了重要的评测漏洞：语义等价 SQL 不再因为 SQLite 返回列标题或别名不同
而被判错。详情见 `docs/sft_v3_report.md`，自动对比见
`artifacts/final_holdout_v3_comparison.md`。

## RL 前置数据与奖励审计

当前没有直接启动 GRPO。项目先把 SFT 剩余问题转成可验证的 RL 数据：使用 30 道
开发题和 60 条 V3 增量题生成 90 条 prompts，以及 270 对偏好数据；两套最终留出集
均不参与构建。

```bash
.venv/bin/python -m sql_debug_agent build-rl-data
```

每题包含三类负例：重复错误 SQL、空结果捷径和危险写操作。奖励同时在演示库和扩展库
执行候选与参考 SQL，只有两套不同数据分布上的结果都一致才得到答案奖励。这可以防止
错误筛选在小数据库上碰巧得到相同结果。当前最小偏好奖励间隔为 1.50，全部 270 对
满足正确答案奖励更高。

产物：

- `artifacts/rl_v1/grpo_prompts.jsonl`
- `artifacts/rl_v1/preference_pairs.jsonl`
- `artifacts/rl_v1/manifest.json`
- `artifacts/rl_v1/reward_audit.md`

`training_ready=false` 是有意设置：V3 仍有回退，所以本阶段只完成奖励工程，不为了
凑流程启动 GRPO。完整设计见 `docs/rl_data_and_reward.md`。

## 真实 Rollout 离线回放

合成奖励通过后，使用 V3-30 实际生成 90 条 SQL，再离线打分：

```bash
.venv/bin/python -m sql_debug_agent replay-rl
.venv/bin/python -m sql_debug_agent export-hard-preferences
```

结果为安全率 100%、双库可执行率 98.9%、双库正确率 71.1%、重复原 SQL 6.7%。
奖励分布有四档，且 3 条真实候选触发了“双库判定不一致”，证明多数据库防作弊确有
必要。26 条真实失败已转换为 hard preference pairs，其中重复计数和 JOIN 各 7 条。

完整结果见 `docs/rl_replay_report.md` 和
`artifacts/rl_v1/replay_v3_30/report.md`。奖励层已经通过，但 SFT 最终留出集仍有回退，
所以 GRPO 训练开关继续保持关闭。

## Corrective Replay 偏好替代实验

当前 MLX-LM 没有 DPO/GRPO trainer，因此先测试适合本机的替代方案：将 26 条真实失败
的 chosen SQL 与 26 条强项保护样本组成 52 条数据，从 V3-30 Adapter 继续训练 5/10
步，学习率降为 `5e-6`。

```bash
.venv/bin/python -m sql_debug_agent build-preference-replay

.venv/bin/python -m sql_debug_agent train-sft \
  --iters 10 --learning-rate 5e-6 \
  --resume-adapter-file artifacts/adapters/sft_v3_30/adapters.safetensors \
  --sft-train artifacts/preference_v1/sft_train.jsonl \
  --sft-eval artifacts/preference_v1/sft_eval.jsonl \
  --data-dir artifacts/mlx_preference_v1 \
  --adapter-path artifacts/adapters/preference_v1_10 \
  --sql-terminator
```

结果是 V3-30、5 步、10 步开发集均为 21/30；10 步候选在 90 条回放上仍为 64/90，
重复错误仍为 6 条。自动门槛因此拒绝候选，当前模型仍是 V3-30：

```bash
.venv/bin/python -m sql_debug_agent compare-replays
```

完整负实验见 `docs/preference_replay_report.md`。它说明普通 SFT 没有利用 rejected SQL
的相对信号，后续若要真正做偏好/RL，需要 DPO 或 GRPO trainer。

## 小型演示数据转换

```bash
python3 -m sql_debug_agent prepare-sft
```

处理逻辑不是简单下载数据集，而是把每条原始 Bad Case 转成：

```text
问题 + Schema + 错误 SQL + 执行反馈 → 正确 SQL
```

该命令只转换最初的 5 条教学样例，生成文件位于 `artifacts/sft/`。正式训练应使用
上面的 `build-data` 命令。每条 SFT 数据都保留错误类型、模板、基线可执行性和
基线奖励，便于后续分类型分析。

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 项目结构

```text
sql-debug-agent/
├── data/
│   ├── tasks.jsonl             # 原始问题、错误 SQL、参考 SQL、错误类型
│   ├── baseline_eval.jsonl      # 用于迭代和 Bad Case 分析的 30 题开发集
│   ├── final_holdout.jsonl       # V2 训练前冻结的 35 题最终留出集
│   ├── final_holdout_v3.jsonl    # V3 训练前冻结的 28 题最终留出集
│   ├── sft_v2_increment.jsonl    # 60 条 V2 针对性增量数据
│   ├── sft_v3_increment.jsonl    # 60 条 V3 泛化与防遗忘数据
│   ├── sft_raw.jsonl            # 200 条正式训练 Bad Case
│   └── train_finance.db         # 与测试库分离的训练数据库
├── docs/
│   └── roadmap.md              # SFT、Bad Case、GRPO 迭代路线
├── sql_debug_agent/
│   ├── agent.py                # 多轮 Agent 循环
│   ├── database.py             # 建库、造数、Schema 工具
│   ├── verifier.py             # 执行与结果验证、奖励
│   ├── repair.py               # 可替换的修复器接口
│   ├── model_prompt.py          # 两种模型共用的提示词与 JSON Schema
│   ├── ollama_repair.py         # 本地 Ollama 模型修复器
│   ├── openai_repair.py         # 可选 Responses API 模型修复器
│   ├── preparation.py          # SFT 数据加工与切分
│   ├── data_generation.py      # 针对性数据生成、验证与去重
│   ├── badcase_analysis.py     # 基线失败分析与阶段决策
│   ├── mlx_training.py         # MLX 数据整理与 8GB LoRA 训练入口
│   ├── mlx_repair.py           # Base/Adapter 的 MLX 推理修复器
│   ├── comparison.py           # Base 与 SFT 对比及回退检测
│   ├── experiment_summary.py   # 三方实验总结与 RL 阶段门槛
│   ├── v3_data.py              # V3 新留出集和多模板训练数据
│   ├── rl_reward.py            # 多数据库、轨迹感知的稳健奖励函数
│   ├── rl_data.py              # 偏好数据与 GRPO prompts 构建
│   ├── rl_replay.py            # 真实模型 rollout、奖励回放与 hard pairs
│   ├── preference_replay.py    # 真实失败纠错与强项保护数据
│   ├── evaluation.py           # 基线和修正后对比
│   └── cli.py                  # 命令行入口
└── tests/
    └── test_agent.py
```

## 核心指标

- `execution accuracy`：SQL 能安全执行，但不代表答案正确。
- `execution-match accuracy`：候选 SQL 与参考 SQL 的执行结果完全一致。
- `repair success rate`：初始结果错误、经过修正后正确的比例。
- `turns to success`：成功前使用的工具调用轮数。
- `accuracy by error type`：按 JOIN、聚合、Schema 等类型分析表现。

完整迭代安排见 [docs/roadmap.md](docs/roadmap.md)。
