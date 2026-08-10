"""
盲评 - 统计胜率：读取打过分的 blind_eval.xlsx 和 mapping json，输出胜率。

前置：
  blind_eval.xlsx        —— 你已经在本地填好 A/B/T 并上传回服务器
  blind_eval_mapping.json —— 由 blind_eval_generate.py 生成

产出：
  直接打印到日志，你从 stdout 里看结果。
"""
import json

# 自动检测并安装 openpyxl
try:
    import openpyxl
except ImportError:
    import subprocess
    import sys
    print('openpyxl 未安装，尝试 pip 安装...')
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

OUTPUT_XLSX = 'blind_eval.xlsx'
MAPPING_JSON = 'blind_eval_mapping.json'


def main():
    with open(MAPPING_JSON, 'r', encoding='utf-8') as f:
        mapping = {m['idx']: m for m in json.load(f)}

    wb = openpyxl.load_workbook(OUTPUT_XLSX)
    ws = wb.active

    base_win = lora_win = tie = invalid = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        idx, question, a, b, choice, note = row[:6]
        if choice is None:
            continue
        c = str(choice).strip().upper()
        if c not in ('A', 'B', 'T'):
            invalid += 1
            continue
        m = mapping.get(idx)
        if m is None:
            invalid += 1
            continue
        if c == 'T':
            tie += 1
        elif c == 'A':
            if m['A_from'] == 'lora':
                lora_win += 1
            else:
                base_win += 1
        elif c == 'B':
            if m['A_from'] == 'lora':
                base_win += 1
            else:
                lora_win += 1

    total_decided = base_win + lora_win
    total = base_win + lora_win + tie
    print(f'\n===== 盲评统计 =====')
    print(f'有效打分: {total}  (无效/未填: {invalid})')
    print(f'  LoRA 胜: {lora_win}')
    print(f'  Base 胜: {base_win}')
    print(f'  平局:    {tie}')
    if total_decided > 0:
        win_rate = lora_win / total_decided * 100
        print(f'\nLoRA 胜率（排除平局）: {win_rate:.1f}%')
    if total > 0:
        win_rate_all = lora_win / total * 100
        print(f'LoRA 胜率（含平局作 0）: {win_rate_all:.1f}%')

    if total_decided > 0:
        print(f'\n简历可写：{total} 题人工盲评，LoRA 模型相比 base Qwen2.5-7B 胜率 {win_rate:.0f}%（排除平局）')


if __name__ == '__main__':
    main()
