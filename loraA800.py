import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import json
print('包导入完成')
# 1. 切换为 7B 模型
# 路径一定要写对，建议写绝对路径
model_id = '/share/home/u01109/zyy多堆/LoRA/model/Qwen2.5-7B'

# 1. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    local_files_only=True  # 强制只从本地读取，禁止联网检查
)

# 2. 加载 Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True  # 强制只从本地读取
)
print('模型加载完成')
# 2. 满血版 LoRA 配置（核心提效区）
lora_config = LoraConfig(
    r = 64,                  # 扩大 Rank，增强模型拟合复杂特征的能力 (原来是 8)
    lora_alpha = 128,        # 行业惯例 alpha 设置为 r 的 2 倍
    lora_dropout = 0.05,
    # 覆盖所有的线性层 (QKV + MLP)，效果最接近全量微调
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias = "none",
    task_type = "CAUSAL_LM"  # 显式指定任务类型
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 你会发现可训练参数比例比之前高很多

# 3. 数据处理 (增加 truncation 防止超长文本截断报错)
with open('fc_data.json','r',encoding='utf-8') as f:
    data = json.load(f)
train_data = []
for d in data:
    text = f"User: {d['instruction']}\nAssistant: {d['output']}"
    tokenized = tokenizer(text,
                          max_length=512,
                          padding='max_length',
                          truncation=True) # 加上截断保平安
    train_data.append({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].copy()
    })

# 4. 科学的训练参数配置
train_args = TrainingArguments(
    output_dir='output_lora_7b_full',
    # 放弃粗暴的 max_steps，改用 epochs 按数据集轮数训练
    num_train_epochs=3,              # 让模型把整个数据集完完整整看 3 遍
    per_device_train_batch_size=4,   # A800 显存大，可以增加单卡 batch_size
    gradient_accumulation_steps=4,   # 实际 equivalent batch size = 16
    learning_rate=1e-4,              # 针对大 Rank，学习率稍微调小一点点更稳
    lr_scheduler_type="cosine",      # 使用余弦退火学习率（前期学得快，后期慢慢收敛找最优解）
    warmup_ratio=0.05,               # 预留 5% 的步数做预热，防止训练初期 Loss 爆炸
    logging_steps=1,
    save_strategy="epoch",           # 每个 epoch 保存一次检查点
    bf16=True,                       # 必须开启！A800 跑 BF16 速度极快且完全无损
)

trainer = Trainer(
    model = model,
    train_dataset = train_data,
    args = train_args
)

trainer.train()

# 5. 保存最终模型
trainer.model.save_pretrained('output_lora_7b_full/final_weights')
tokenizer.save_pretrained('output_lora_7b_full/final_weights')