import torch
from transformers import AutoModelForCausalLM,AutoTokenizer, TextStreamer
from peft import PeftModel

model_path = r'E:\PycharmProjects\LoRA\share\Qwen2.5-7B'
lora_weights_path = r'E:\PycharmProjects\LoRA\final_weights'

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=True,
                                             device_map = 'cuda',
                                             dtype=torch.bfloat16
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 搭载lora
model = PeftModel.from_pretrained(model,lora_weights_path)
model.eval()

def ask_model(instruction):

    prompt = f"User: {instruction}\nAssistant: "
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    print('模型回复：', end="", flush=True)  # 把提示语提前，配合流式输出

    with torch.no_grad():
        model.generate(**inputs,
                        streamer = streamer,
                        max_new_tokens = 512,
                        do_sample = True,
                        top_p = 0.9,
                        temperature = 0.7,
                        pad_token_id=tokenizer.eos_token_id,
                       repetition_penalty=1.1
                                 )
    print("\n")



print('模型准备完成')
while True:
    User_input = input('你的问题： ')
    if User_input == 'exit':
        break
    ask_model(User_input)

