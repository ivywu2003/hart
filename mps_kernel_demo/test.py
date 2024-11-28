import torch
from compiler import compiled_lib

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

n = 1025 # seems like max number of threads per group is 1024
x = torch.ones(n, device=mps_device)
y = torch.randn(n, device=mps_device)

out = compiled_lib.mps_rms_norm(x, y)
print(x)
print(y)
print(out)
