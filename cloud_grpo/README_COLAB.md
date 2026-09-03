# SQL Debug Agent：Colab SFT → GRPO 操作说明

这一阶段不需要 OpenAI API，也不需要中转站。训练使用 Hugging Face 开源权重和
Google Colab 的 GPU。建议先选 `T4 GPU`，所有命令先跑 5 步 smoke test。

## 为什么云端还要再做一次 SFT

Mac 上的 `mlx-community/...-4bit` 和 Adapter 属于 MLX 格式；Colab 的 TRL 使用
PyTorch + PEFT，不能安全地直接续训 MLX Adapter。因此云端先用同一批 V3 数据重建
标准 PEFT SFT Adapter，再从这个 Adapter 继续 GRPO。这不是重复做实验，而是格式迁移。

## 第 1 步：在 Mac 生成上传包

在项目目录运行：

```bash
.venv/bin/python cloud_grpo/prepare_bundle.py
```

得到 `artifacts/sql-debug-agent-colab.zip`。压缩包只包含训练代码、218 条 SFT 训练数据、
42 条 SFT 验证数据、90 条 GRPO prompts、其中筛出的 26 条真实 rollout 失败难例和
两套奖励数据库；两个最终留出集没有打包。

## 第 2 步：打开 Colab 并选择 GPU

最简单的方法是把 `cloud_grpo/SQL_Debug_Agent_GRPO_Colab.ipynb` 上传到 Colab，
然后按单元格顺序运行。也可以新建 Notebook，手动运行下面的代码块。先在 Colab
菜单中选择 `运行时 → 更改运行时类型 → T4 GPU`。

```python
from google.colab import files
files.upload()  # 选择 artifacts/sql-debug-agent-colab.zip
```

```python
!unzip -q sql-debug-agent-colab.zip
%cd /content/sql-debug-agent-colab
!pip install -q -r cloud_grpo/requirements-colab.txt
!pip install -q -e .
```

安装后执行 `运行时 → 重新启动会话`，然后重新运行下面的 `%cd`：

```python
%cd /content/sql-debug-agent-colab
```

依赖文件把 Transformers 限定在 4.x，以避免 Colab 自动安装的 5.x 版本与当前训练
配置发生参数兼容问题。

## 第 3 步：SFT smoke test

```python
!python cloud_grpo/train_sft.py --max-steps 5
```

出现 `SFT Adapter 已保存` 才算成功。若显存不足，不要继续完整训练，先保存错误截图。

若旧压缩包出现 `_amp_foreach_non_finite... not implemented for 'BFloat16'`，说明 T4
遇到了 LoRA 参数精度兼容问题。新版脚本会按 PEFT 建议，把可训练 Adapter 参数转为
FP32，同时保留 4-bit 基座和 FP16 混合精度。GRPO 的转换必须放在 `GRPOTrainer`
初始化之后，因为 Trainer 可能在初始化参考策略时重新处理 Adapter。

## 第 4 步：完整 SFT

正式结果应直接保存到 Google Drive，避免 Colab 临时运行环境被回收后丢失。运行：

```python
%cd /content/sql-debug-agent-colab
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/sql-debug-agent-outputs
!python cloud_grpo/train_sft.py --max-steps 60 \
  --output-dir /content/drive/MyDrive/sql-debug-agent-outputs/sft_adapter
```

## 第 5 步：GRPO V2 smoke test

第一版 smoke test 的两个候选每次都完全正确，因此 `reward_std=0`、优势为 0，实际没有
发生策略学习。V2 默认只使用 26 条真实失败难例，对每题采样 4 个 SQL，并适当提高采样
温度，再用双数据库执行结果给奖励：

```python
%cd /content/sql-debug-agent-colab
!python cloud_grpo/train_grpo.py --max-steps 5 --dataset hard \
  --num-generations 4 --temperature 1.2 \
  --sft-adapter /content/drive/MyDrive/sql-debug-agent-outputs/sft_adapter \
  --output-dir /content/drive/MyDrive/sql-debug-agent-outputs/grpo_v2_smoke
```

重点观察日志中的 `reward`、`reward_std` 和 `frac_reward_zero_std`。如果奖励始终没有方差，
说明同组候选全对或全错，暂时不应直接增加训练步数。脚本末尾会自动生成
`grpo_signal_summary.json`，并打印“GRPO 学习信号：x/5 个步骤有奖励差异”。

## 第 6 步：小规模 GRPO

smoke test 正常后，从头创建一次 40 步实验：

```python
%cd /content/sql-debug-agent-colab
!python cloud_grpo/train_grpo.py --max-steps 40 --dataset hard \
  --num-generations 4 --temperature 1.2 \
  --sft-adapter /content/drive/MyDrive/sql-debug-agent-outputs/sft_adapter \
  --output-dir /content/drive/MyDrive/sql-debug-agent-outputs/grpo_v2_40
```

这只是第一轮小规模实验。不能因为训练命令成功，就宣称 GRPO 有效；下一阶段必须在
未参与训练的冻结测试集上比较 Base、SFT、SFT+GRPO。

## 第 7 步：下载 Adapter 和日志

```python
%cd /content/sql-debug-agent-colab
!zip -qr sql-debug-agent-cloud-outputs.zip /content/drive/MyDrive/sql-debug-agent-outputs
from google.colab import files
files.download("sql-debug-agent-cloud-outputs.zip")
```

把下载的 zip 保存好，下一阶段会导回项目进行统一评测。

## 第 8 步：训练结束后运行冻结评测

最终测试集不在训练包中。只有 SFT 和 GRPO 都结束后，才在 Mac 生成并上传：

```bash
.venv/bin/python cloud_grpo/prepare_final_eval.py
```

把 `artifacts/sql-debug-agent-final-eval.zip` 解压到现有
`/content/sql-debug-agent-colab`，再运行：

```python
%cd /content/sql-debug-agent-colab
!python cloud_grpo/evaluate_checkpoints.py \
  --sft-adapter cloud_outputs/sft_adapter \
  --grpo-adapter cloud_outputs/grpo_v2_40 \
  --output-dir cloud_outputs/final_evaluation
```

本次正式实验结果为 Base 16/28（57.1%）、SFT 23/28（82.1%）、SFT+GRPO
25/28（89.3%）。GRPO 相对 SFT 修复 2 题、回退 0 题。

## 安全边界

- 奖励只允许单条 `SELECT/WITH`，危险 SQL 直接负奖励。
- 正确性要同时通过两套同 Schema、不同数据的 SQLite 数据库。
- 最终 holdout 不上传到训练环境，避免人为查看或训练泄漏。
- 这份配置面向免费 T4 的小规模验证，不代表生产级训练配方。
