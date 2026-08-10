"""
盲评 - 生成 Excel：让 base 和 lora 模型各答 40 题，匿名+随机顺序写入 Excel。

产出：
  blind_eval.xlsx        —— 下载到本地打分（在"你的选择"列填 A/B/T）
  blind_eval_mapping.json —— A/B 到 base/lora 的映射（打分后回传，score 脚本用）

上传打分后的 xlsx 后，运行 blind_eval_score.py 统计胜率。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 自动检测并安装 openpyxl（A800 环境不一定预装）
try:
    import openpyxl
except ImportError:
    import subprocess
    import sys
    print('openpyxl 未安装，尝试 pip 安装...')
    # 依次尝试三个源：清华 → 阿里 → 官方
    for index_url in [
        'https://pypi.tuna.tsinghua.edu.cn/simple',
        'https://mirrors.aliyun.com/pypi/simple',
        'https://pypi.org/simple',
    ]:
        r = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'openpyxl', '-i', index_url, '--user'],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f'✅ openpyxl 已从 {index_url} 装好')
            break
        else:
            print(f'❌ 从 {index_url} 失败，试下一个源')
    import openpyxl
    print(f'openpyxl 版本: {openpyxl.__version__}')

import json
import random
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import openpyxl
from config import MODEL_ID as BASE_MODEL_PATH, LORA_WEIGHTS_PATH, LOCAL_FILES_ONLY, TORCH_DTYPE, N_BLIND_QUESTIONS

EVAL_RAW_JSON = 'eval_raw.json'
OUTPUT_XLSX = 'blind_eval.xlsx'
MAPPING_JSON = 'blind_eval_mapping.json'
SEED = 2026


def generate_answers(model, tokenizer, question, max_new_tokens=400):
    prompt = f"User: {question}\nAssistant: "
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    with open(EVAL_RAW_JSON, 'r', encoding='utf-8') as f:
        eval_raw = json.load(f)

    random.seed(SEED)
    picked = random.sample(eval_raw, min(N_BLIND_QUESTIONS, len(eval_raw)))
    questions = [d['instruction'] for d in picked]

    dtype = torch.bfloat16 if TORCH_DTYPE == 'bfloat16' else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)

    print('加载 base 模型...')
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        local_files_only=LOCAL_FILES_ONLY,
    )
    base_model.eval()
    print('base 模型生成回答...')
    base_answers = []
    for i, q in enumerate(questions):
        ans = generate_answers(base_model, tokenizer, q)
        base_answers.append(ans)
        print(f'  base [{i+1}/{len(questions)}] done')
    del base_model
    torch.cuda.empty_cache()

    print('加载 LoRA 模型...')
    lora_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        local_files_only=LOCAL_FILES_ONLY,
    )
    lora_model = PeftModel.from_pretrained(lora_base, LORA_WEIGHTS_PATH)
    lora_model.eval()
    print('LoRA 模型生成回答...')
    lora_answers = []
    for i, q in enumerate(questions):
        ans = generate_answers(lora_model, tokenizer, q)
        lora_answers.append(ans)
        print(f'  lora [{i+1}/{len(questions)}] done')

    mapping = []
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '盲评'
    ws.append(['编号', '问题', '回答 A', '回答 B', '你的选择 (A/B/T)', '备注'])

    rng = random.Random(SEED + 1)
    for i, q in enumerate(questions):
        if rng.random() < 0.5:
            a, b, a_src = base_answers[i], lora_answers[i], 'base'
        else:
            a, b, a_src = lora_answers[i], base_answers[i], 'lora'
        mapping.append({'idx': i + 1, 'question': q, 'A_from': a_src})
        ws.append([i + 1, q, a, b, '', ''])

    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 60
    ws.column_dimensions['D'].width = 60
    ws.column_dimensions['E'].width = 18
    wb.save(OUTPUT_XLSX)

    with open(MAPPING_JSON, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f'\n✅ 已生成 {OUTPUT_XLSX}（{len(questions)} 题），映射存于 {MAPPING_JSON}')
    print('下载 xlsx 到本地打分（A/B/T），改完上传回来，再运行 blind_eval_score.py')


if __name__ == '__main__':
    main()
