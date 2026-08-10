# Qwen2.5-7B LoRA 微调：燃料电池领域知识问答

基于 Qwen2.5-7B + LoRA 的参数高效微调项目，面向燃料电池领域知识问答场景。

## 项目概览

- **Base 模型**：Qwen2.5-7B
- **微调方式**：LoRA（rank=8, alpha=16），覆盖 Q/K/V/O 及 MLP 全部线性层
- **训练数据**：1066 条燃料电池领域 SFT 问答对（GPT-4 蒸馏 + self-instruct 生成）
- **硬件**：A800 单卡 bf16
- **训练耗时**：约 6 分钟（3 epoch）

## 评测结果

| 指标 | 数值 |
|---|---|
| eval loss（held-out 103 条） | 1.58 → 1.32（epoch 2 最优） |
| 人工盲评胜率（40 题 vs base） | 65% |
| C-Eval 通用能力（中文/近代史/逻辑）| 平均变化在噪声范围内 |

## 环境依赖

```
torch >= 2.0
transformers >= 4.36
peft >= 0.7
openpyxl
```

见 `requirements.txt`。

## 使用流程

```bash
# 1. 数据准备（切 train/eval + label mask）
python prepare_data.py

# 2. 训练
python train_with_eval.py

# 3. 盲评：生成 → 人工打分 → 统计
python blind_eval_generate.py
# 打开 blind_eval.xlsx 在"你的选择"列填 A/B/T
python blind_eval_score.py

# 4. C-Eval 通用能力评测
python ceval_run_base.py
python ceval_run_lora.py
python ceval_run_compare.py
```

## 关键技术点

- **SFT label mask**：只对 Assistant 回答段算 loss，User prompt 与 padding 位置设为 -100
- **eval loss 监控 + load_best_model_at_end**：自动加载 eval loss 最低的 checkpoint，避免过拟合版本
- **多维度评测**：eval loss（收敛）+ 人工盲评（领域能力）+ C-Eval（通用能力）

## 配置

`config.py` 支持通过环境变量 `LORA_ENV` 切换 local / a800 环境。

## License

MIT
