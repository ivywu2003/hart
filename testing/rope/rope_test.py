import torch
import hart_backend.fused_kernels as fused
import cProfile
import time

def test_rope():
    device = torch.device("mps")
    t = torch.load("rope_tests/forward_with_pos_t.pt").to(device)
    freqs = torch.load("rope_tests/forward_with_pos_freqs.pt").to(device)
    correct = torch.load("rope_tests/forward_with_pos_output.pt").to(device)

    output = fused.fused_rope_with_pos_forward_func(t, freqs, False)
    print(torch.isclose(output, correct))

if __name__ == "__main__":
    test_rope()
