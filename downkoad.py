import os


# 1. 设置镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
# 2. 定义模型信息
model_id = "Qwen/Qwen2.5-7B"
save_path = "share/Qwen2.5-7B"

print(f"正在开始下载模型 {model_id} 到 {save_path}...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=save_path,
        local_dir_use_symlinks=False, # 建议设为 False，直接存文件而非软连接
        resume_download=True,         # 支持断点续传
        token=None                    # 如果是公共模型不需要 token
    )
    print("✅ 下载完成！")
except Exception as e:
    print(f"❌ 下载出错: {e}")