import torch
from io import BytesIO
from typing import Union


def tensor2Bytes(data: Union[torch.Tensor, dict]):
    buffer = BytesIO()
    if isinstance(data, dict):
        # 将字典中的所有张量移动到CPU
        cpu_dict = {k: v.cpu() if torch.is_tensor(v) else v for k, v in data.items()}
        torch.save(cpu_dict, buffer)
    else:
        # 处理单个张量的情况
        data_cpu = data.cpu() if torch.is_tensor(data) else data
        torch.save(data_cpu, buffer)
    buffer.seek(0)
    return buffer

def bytes2Tensor(bytes: bytes):
    return torch.load(BytesIO(bytes), map_location="cpu", weights_only=True)
