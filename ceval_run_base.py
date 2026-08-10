"""
C-Eval 评测 - base 模型：在 STEM 三科 val 集上跑 base Qwen2.5-7B。

前置：
  ceval-exam/ 数据集目录（解压 ceval-exam.zip 得到）
    ├── dev/<subject>_dev.csv
    └── val/<subject>_val.csv

产出：
  ceval_result_base.json —— 三科准确率，供 ceval_run_compare.py 用
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import csv
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_ID as BASE_MODEL_PATH, LOCAL_FILES_ONLY, TORCH_DTYPE

CEVAL_ROOT = 'ceval-exam'
SUBJECTS = [
    'chinese_language_and_literature',
    'modern_chinese_history',
    'logic',
]
FEW_SHOT_N = 5


def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def format_question(row, with_answer=False):
    q = row['question']
    a, b, c, d = row['A'], row['B'], row['C'], row['D']
    text = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n答案："
    if with_answer:
        text += row['answer']
    return text


def build_prompt(dev_rows, test_row):
    shots = [format_question(r, with_answer=True) for r in dev_rows[:FEW_SHOT_N]]
    query = format_question(test_row, with_answer=False)
    return "以下是中国关于该学科的多项选择题，请选出其中的正确答案。\n\n" + "\n\n".join(shots) + "\n\n" + query


def predict_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    choices = ['A', 'B', 'C', 'D']
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
    choice_logits = [logits[cid].item() for cid in choice_ids]
    return choices[choice_logits.index(max(choice_logits))]


def evaluate(model, tokenizer):
    results = {}
    for subj in SUBJECTS:
        dev_path = os.path.join(CEVAL_ROOT, 'dev', f'{subj}_dev.csv')
        val_path = os.path.join(CEVAL_ROOT, 'val', f'{subj}_val.csv')
        if not (os.path.exists(dev_path) and os.path.exists(val_path)):
            print(f'  ⚠️ 找不到 {subj}，跳过')
            continue
        dev_rows = load_csv(dev_path)
        val_rows = load_csv(val_path)

        correct = 0
        for i, row in enumerate(val_rows):
            prompt = build_prompt(dev_rows, row)
            pred = predict_answer(model, tokenizer, prompt)
            if pred == row['answer']:
                correct += 1
            if (i + 1) % 10 == 0:
                print(f'  {subj} [{i+1}/{len(val_rows)}] acc={correct/(i+1):.3f}')
        acc = correct / len(val_rows)
        results[subj] = {'correct': correct, 'total': len(val_rows), 'acc': acc}
        print(f'  ✅ {subj}: {correct}/{len(val_rows)} = {acc:.3f}')
    return results


def main():
    dtype = torch.bfloat16 if TORCH_DTYPE == 'bfloat16' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
    print('加载 base 模型...')
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True,
        torch_dtype=dtype,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.eval()

    print('评测中 (mode=base)...')
    results = evaluate(model, tokenizer)
    with open('ceval_result_base_general.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n✅ 结果已存至 ceval_result_base_general.json')

    total_correct = sum(r['correct'] for r in results.values())
    total_all = sum(r['total'] for r in results.values())
    if total_all:
        print(f'STEM 3 科总平均: {total_correct}/{total_all} = {total_correct/total_all:.3f}')


if __name__ == '__main__':
    main()
