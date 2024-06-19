import torch
from io import BytesIO


def tensor2Bytes(image):
    buffer = BytesIO()
    torch.save(image, buffer)
    buffer.seek(0)
    return buffer
