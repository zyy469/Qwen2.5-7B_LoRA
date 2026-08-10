"""
C-Eval 对比：读取 base / lora 的评测结果 json，打印对比表和简历可用文案。

前置：
  ceval_result_base.json  —— 由 ceval_run_base.py 产出
  ceval_result_lora.json  —— 由 ceval_run_lora.py 产出
"""
import json

SUBJECTS = [
    'chinese_language_and_literature',
    'modern_chinese_history',
    'logic',
]


def main():
    with open('ceval_result_base_general.json', 'r', encoding='utf-8') as f:
        base = json.load(f)
    with open('ceval_result_lora_general.json', 'r', encoding='utf-8') as f:
        lora = json.load(f)

    print('\n===== C-Eval STEM 对比 =====')
    print(f'{"科目":<25} {"Base":>10} {"LoRA":>10} {"差值":>10}')
    total_b_c = total_l_c = total_all = 0
    for subj in SUBJECTS:
        if subj not in base or subj not in lora:
            continue
        b_acc = base[subj]['acc']
        l_acc = lora[subj]['acc']
        diff = (l_acc - b_acc) * 100
        print(f'{subj:<25} {b_acc*100:>9.1f}% {l_acc*100:>9.1f}% {diff:>+9.1f}pp')
        total_b_c += base[subj]['correct']
        total_l_c += lora[subj]['correct']
        total_all += base[subj]['total']

    if total_all:
        b_avg = total_b_c / total_all
        l_avg = total_l_c / total_all
        diff = (l_avg - b_avg) * 100
        print(f'{"平均":<25} {b_avg*100:>9.1f}% {l_avg*100:>9.1f}% {diff:>+9.1f}pp')

        print(f'\n简历可写：')
        if abs(diff) < 2:
            print(f'  C-Eval STEM 子集平均准确率变化 {diff:+.1f}pp，通用能力无明显退化。')
        elif diff > 0:
            print(f'  C-Eval STEM 子集平均准确率不降反升 {diff:+.1f}pp。')
        else:
            print(f'  C-Eval STEM 子集下降 {abs(diff):.1f}pp，仍在可接受范围。')


if __name__ == '__main__':
    main()
