"""
数据准备脚本：把 fc_data.json 切分为 train/eval，并生成正确的 label mask。

修复了原 loraA800.py 的两个问题：
1. label mask：User prompt 段 + padding 段的 label 设为 -100，Trainer 只对 Assistant 回答算 loss
2. 划分 train/eval：9:1 切分，保证 eval 集训练时不可见

产出：
  train.pt / eval.pt —— 两个 torch pickle，直接被 train_with_eval.py 加载
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import random
import torch
from transformers import AutoTokenizer
from config import MODEL_ID, LOCAL_FILES_ONLY

DATA_PATH = 'fc_data.json'
MAX_LEN = 512
EVAL_RATIO = 0.1
SEED = 42

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_sample(instruction: str, output: str):
    """
    构造一条 SFT 样本，Assistant 回答之外的位置 label 设为 -100。
    -100 是 PyTorch CrossEntropyLoss 的 ignore_index，不参与 loss 计算。
    """
    prompt = f"User: {instruction}\nAssistant: "
    full_text = prompt + output + tokenizer.eos_token

    # 分别 tokenize，拿到 prompt 段长度
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full = tokenizer(
        full_text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
    )
    input_ids = full['input_ids']
    attention_mask = full['attention_mask']

    labels = list(input_ids)
    # 1) mask 掉 User prompt 段
    prompt_len = min(len(prompt_ids), MAX_LEN)
    for i in range(prompt_len):
        labels[i] = -100
    # 2) mask 掉 padding 段
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    random.seed(SEED)
    random.shuffle(raw)

    n_eval = int(len(raw) * EVAL_RATIO)
    eval_raw = raw[:n_eval]
    train_raw = raw[n_eval:]

    print(f'总样本: {len(raw)}  train: {len(train_raw)}  eval: {len(eval_raw)}')

    train_data = [build_sample(d['instruction'], d['output']) for d in train_raw]
    eval_data = [build_sample(d['instruction'], d['output']) for d in eval_raw]

    torch.save(train_data, 'train.pt')
    torch.save(eval_data, 'eval.pt')

    # 同时把原始 eval 问答对存一份 json，供盲评脚本用
    with open('eval_raw.json', 'w', encoding='utf-8') as f:
        json.dump(eval_raw, f, ensure_ascii=False, indent=2)

    # 检查：随机看一条样本的 label mask 效果
    sample = train_data[0]
    label_visible = sum(1 for x in sample['labels'] if x != -100)
    label_masked = sum(1 for x in sample['labels'] if x == -100)
    print(f'\n样本自检: 有效 label token 数={label_visible}, 被 mask 的 token 数={label_masked}')
    print(f'样本 tokens (前 40):\n  input_ids: {sample["input_ids"][:40]}')
    print(f'  labels:    {sample["labels"][:40]}')
    print('\n✅ 数据准备完成: train.pt / eval.pt / eval_raw.json')


if __name__ == '__main__':
    main()
