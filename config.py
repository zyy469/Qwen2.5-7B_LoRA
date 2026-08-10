"""
统一配置：本地调试和 A800 生产共用一套代码。

用法：
  本地调试：设环境变量 LORA_ENV=local，或者不设也默认 local
  A800 生产：设环境变量 LORA_ENV=a800

  Windows PowerShell:  $env:LORA_ENV="local"
  Linux bash:          export LORA_ENV=a800
"""
import os

ENV = os.environ.get('LORA_ENV', 'a800').lower()

if ENV == 'a800':
    # A800 生产环境
    MODEL_ID = '/share/home/u01109/zyy多堆/LoRA/model/Qwen2.5-7B'
    LORA_WEIGHTS_PATH = 'output_lora_7b_full/final_weights'
    OUTPUT_DIR = 'output_lora_7b_full'
    # 训练参数
    PER_DEVICE_BATCH = 4
    GRAD_ACC = 4
    NUM_EPOCHS = 3
    LORA_R = 8
    LORA_ALPHA = 16
    LOCAL_FILES_ONLY = True
    TORCH_DTYPE = 'bfloat16'
    # 数据规模
    N_BLIND_QUESTIONS = 40
else:
    # 本地调试：小模型 + 少量样本
    # 用在线的 Qwen2.5-0.5B，第一次跑会自动下载（约 1GB）
    MODEL_ID = 'Qwen/Qwen2.5-0.5B'
    LORA_WEIGHTS_PATH = 'output_lora_local/final_weights'
    OUTPUT_DIR = 'output_lora_local'
    # 训练参数（小模型 + 少量步数，快速验证代码能跑通）
    PER_DEVICE_BATCH = 1
    GRAD_ACC = 2
    NUM_EPOCHS = 1
    LORA_R = 8
    LORA_ALPHA = 16
    LOCAL_FILES_ONLY = False
    TORCH_DTYPE = 'float16'    # 老显卡不支持 bf16，用 fp16
    # 数据规模
    N_BLIND_QUESTIONS = 4      # 本地只生成 4 题，验证 Excel 流程即可

print(f'[config] LORA_ENV={ENV}, MODEL_ID={MODEL_ID}')
