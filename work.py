import sys
import torch

#verify that the torch is using the GPU
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

print("hello from work.py")
