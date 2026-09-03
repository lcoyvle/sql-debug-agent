# Hard Preference Corrective Replay 实验

## 方法选择

本机安装的 MLX-LM 0.31.3 提供 LoRA/DoRA/全量微调和 Adapter resume，但没有 DPO 或
GRPO trainer。为了在 M1 8GB 上验证 26 条真实 hard preferences 是否能产生收益，本轮
采用明确标注的 `corrective replay SFT`，不把它称为 DPO。

## 数据

- 26 条真实失败纠错：参考正确 SQL 作为监督答案。
- 26 条强项保护：从 Syntax、Schema Linking、Filter 的成功 rollout 中轮换抽样。
- 总训练数据：52 条。
- 验证数据：42 条 V3 模板隔离数据。

强项保护与失败纠错保持 1:1，目的是降低只训练 JOIN 和重复计数造成的灾难性遗忘。

## 训练

两个候选都从 `sft_v3_30/adapters.safetensors` 继续训练：

| 候选 | 步数 | 学习率 | 验证损失 | 峰值内存 |
| --- | ---: | ---: | ---: | ---: |
| preference_v1_5 | 5 | 5e-6 | 0.086 | 1.922 GB |
| preference_v1_10 | 10 | 5e-6 | 0.075 | 1.922 GB |

## 开发集结果

| 模型 | 正确题数 | 准确率 |
| --- | ---: | ---: |
| SFT V3-30 | 21/30 | 70.0% |
| preference_v1_5 | 21/30 | 70.0% |
| preference_v1_10 | 21/30 | 70.0% |

三个模型的任务级成功集合相同。短训练没有造成回退，但也没有修复新题。

## 90 条真实 Rollout 回放

根据较低验证损失选择 10 步候选做训练域回放：

| 指标 | V3-30 | preference_v1_10 | 变化 |
| --- | ---: | ---: | ---: |
| 双库正确数 | 64/90 | 64/90 | 0 |
| 重复原 SQL | 6 | 6 | 0 |
| 平均奖励 | 1.05 | 1.05 | 0.00 |

部分 SQL 的空格、分号或无效附加条件发生变化，但没有任何任务从失败变为成功，也没有
任务从成功变为失败。

## 决策

拒绝两个候选，当前继续使用 SFT V3-30。原因不是训练报错，而是任务级指标没有收益。

这次负实验说明：把 26 条 hard pairs 的 chosen 答案再次做低学习率 SFT，不能利用
chosen/rejected 之间的相对信息。真正的偏好优化需要 DPO 类 pairwise loss，真正的
策略优化需要 GRPO 类 group-relative reward；普通 SFT 只看 chosen token，无法直接
惩罚模型真实 rejected 行为。

下一步有两个合理方向：

1. 在保留本地项目的前提下，准备可在免费云端 GPU 运行的 TRL/GRPO 训练包。
2. 若坚持纯本地，先扩充真实失败的语义多样性，并使用 Agent 层重复检测与定向反馈，
   但不能把它称作 GRPO。
