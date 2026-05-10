import torch
# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
# 如果CUDA可用，打印当前GPU型号和计算能力
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU capability: {torch.cuda.get_device_capability(0)}")