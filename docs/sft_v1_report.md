# SFT V1 实验报告

## 实验目的

验证“基线评测 → Bad Case 定位 → 针对性数据构建 → SFT → 独立对比”是否能够
在一台 M1 8GB MacBook Air 上真实跑通，并判断下一步应继续 SFT 还是进入 RL。

## 实验设置

- 框架：[Apple MLX-LM](https://github.com/ml-explore/mlx-lm)
- 模型：[mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit)
- 训练方式：4-bit QLoRA
- 训练数据：158 条
- 模板隔离内部验证数据：42 条
- 最终独立测试集：30 条，从未进入训练
- 可训练参数：1.319M / 1543.714M（0.085%）
- batch size：1
- 梯度累积：4
- LoRA 层数：最后 4 层
- 最大序列长度：1024
- 训练步数：50
- 学习率：1e-5
- 峰值内存：1.831GB

## 为什么选 MLX QLoRA

机器只有 8GB 统一内存，CUDA 训练方案不可用。MLX 针对 Apple Silicon，量化模型
可以直接进行 QLoRA；模型权重约 880MB，只训练约 132 万个 Adapter 参数，能够在
本机完成真实训练，而不是只写一份无法运行的脚本。

## 训练结果

验证损失从第 1 步的 0.507 降到第 50 步的 0.026。训练损失在第 10、20、30、
40、50 步分别为 0.795、0.653、0.200、0.054、0.053。

训练中出现一个重要 Bad Case：Adapter 可以先生成正确 SQL，但之后会重复输出
`!`，直至达到生成上限。这说明低验证损失不等于 Agent 可用。项目因此增加流式
首行停止和特殊 token 清洗，再将清洗后的 SQL 交给 SQLite 验证。

## 同条件独立测试

| 错误类型 | MLX Base | SFT V1 | 变化 |
| --- | ---: | ---: | ---: |
| Aggregation | 75% | 75% | 0 |
| Date | 80% | 80% | 0 |
| Duplicate Counting | 0% | 100% | +100% |
| Filter | 100% | 80% | -20% |
| JOIN | 60% | 80% | +20% |
| Schema Linking | 80% | 80% | 0 |
| Syntax Error | 100% | 100% | 0 |
| **总体** | **80.0%** | **83.3%** | **+3.3 个百分点** |

SFT 新修复 `eval_aggregation_04`、`eval_join_03`，但回退 `eval_filter_04`。
最终剩余 5 个失败任务：`eval_aggregation_05`、`eval_date_03`、
`eval_filter_04`、`eval_join_04`、`eval_schema_03`。

## 阶段判断

暂不进入 RL。理由不是 RL 无用，而是第一轮 SFT 只有 1 题净提升并出现监督能力
回退，剩余问题仍可以直接构造正确答案进行监督学习。下一轮应：

1. 为 5 个剩余失败模式构造新的、与测试题不重复的困难训练模板。
2. 加入过滤值保持的对照样本，防止 `high` 被错误保留成 `low`。
3. 在训练答案中强化单条 SQL 结束协议。
4. 运行 SFT V2 后继续使用同一 30 题测试集比较。
5. 只有连续 SFT 迭代进入平台期，且失败集中在多轮探索时，再构建 GRPO 数据与奖励。
