"""
带 eval 的训练脚本：在原 loraA800.py 基础上加了 eval_dataset + eval_strategy=epoch。

跑完你会拿到：
1. output_lora_7b_full/ 下的每个 checkpoint（每 epoch 一个）
2. 训练日志里每个 step 的 train loss + 每个 epoch 末的 eval loss
3. loss_log.json：结构化的 loss 曲线，方便写简历时直接引用数字

跟原脚本的区别：
- 加载 train.pt / eval.pt（由 prepare_data.py 产出，已做过 label mask）
- Trainer 加 eval_dataset 参数
- 加 eval_strategy='epoch' —— 每 epoch 结束自动跑一次 eval
- 用 TrainerCallback 记录 loss 曲线到 json
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from config import (
    MODEL_ID, OUTPUT_DIR, LOCAL_FILES_ONLY, TORCH_DTYPE,
    PER_DEVICE_BATCH, GRAD_ACC, NUM_EPOCHS, LORA_R, LORA_ALPHA, ENV,
)


class LossLogger(TrainerCallback):
    """把每个 step 的 train loss 和每个 epoch 的 eval loss 存到 json，方便回填简历。"""

    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {'step': state.global_step, 'epoch': state.epoch}
        entry.update(logs)
        self.records.append(entry)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


def main():
    print('加载数据...')
    train_data = torch.load('train.pt')
    eval_data = torch.load('eval.pt')
    print(f'train={len(train_data)}, eval={len(eval_data)}')

    print('加载 tokenizer / model...')
    dtype = torch.bfloat16 if TORCH_DTYPE == 'bfloat16' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=LOCAL_FILES_ONLY,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # transformers 版本兼容：>=4.41 用 eval_strategy，老版本用 evaluation_strategy
    import inspect
    _ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    _eval_key = 'eval_strategy' if 'eval_strategy' in _ta_params else 'evaluation_strategy'

    ta_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(TORCH_DTYPE == 'bfloat16'),
        fp16=(TORCH_DTYPE == 'float16'),
        report_to="none",
    )
    ta_kwargs[_eval_key] = "epoch"
    print(f'[compat] TrainingArguments 使用参数名: {_eval_key}')
    train_args = TrainingArguments(**ta_kwargs)

    logger = LossLogger()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[logger],
    )

    trainer.train()

    # 保存最终权重和 loss 曲线
    trainer.model.save_pretrained(f'{OUTPUT_DIR}/final_weights')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/final_weights')
    logger.save(f'{OUTPUT_DIR}/loss_log.json')

    # 打印关键节点，直接写简历
    eval_losses = [r for r in logger.records if 'eval_loss' in r]
    if eval_losses:
        print('\n===== eval loss 曲线（写简历用）=====')
        for r in eval_losses:
            print(f"  epoch {r['epoch']:.1f}: eval_loss = {r['eval_loss']:.4f}")
        print(f"\n起始 eval_loss = {eval_losses[0]['eval_loss']:.4f}")
        print(f"最终 eval_loss = {eval_losses[-1]['eval_loss']:.4f}")


if __name__ == '__main__':
    main()
