import torch
from compiler import compiled_lib

assert torch.backends.mps.is_available()
mps_device = torch.device("mps")  # Device object representing GPU.

n = 1024 # seems like max number of threads per group is 1024
x = torch.ones(n, device=mps_device)

batch_size = 2
seq_len = 3
hidden_size = 3

device = torch.device("mps")
input_tensor = torch.ones(batch_size, seq_len, hidden_size, device='cpu')
input_tensor[0] *= -1
input_tensor[0] = input_tensor[0] * 5;
input_tensor[1][0][0] = 100
input_tensor = input_tensor.to(device)
weight = torch.ones(hidden_size, device=device)

print("INPUT:")
print(input_tensor)
print(input_tensor.size())

out = compiled_lib.mps_rms_norm(input_tensor, weight, 1e-6, False)
print("OUTPUT:")
print(out)
print(out.size())
