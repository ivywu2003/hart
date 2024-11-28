import torch
from compiler import compiled_lib

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

x = torch.ones(64, device=mps_device)
y = torch.randn(64, device=mps_device)

out = compiled_lib.mps_rms_norm(x, y)
print(x)
print(y)
print(out)
