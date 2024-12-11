import torch
import hart_backend.fused_kernels as fused
import cProfile
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda")

def test_with_pos():
    t = torch.load("rope_tests/t.pt").to(device)
    freqs = torch.load("rope_tests/freqs.pt").to(device)
    correct = torch.load("rope_tests/forward_with_pos_output.pt").to(device)

    output = fused.fused_rope_with_pos_forward_func(t, freqs, False)
    print(torch.isclose(output, correct))

def test_forward():
    t = torch.load("rope_tests/t.pt").to(device)
    freqs = torch.load("rope_tests/freqs.pt").to(device)
    correct = torch.load("rope_tests/forward_output.pt").to(device)

    output = fused.fused_rope_forward_func(t, freqs, False)
    print(torch.isclose(output, correct))

def test_backward():
    t = torch.load("rope_tests/t.pt").to(device)
    freqs = torch.load("rope_tests/freqs.pt").to(device)
    correct = torch.load("rope_tests/backward_output.pt").to(device)

    output = fused.fused_rope_backward_func(t, freqs, False)
    print(torch.isclose(output, correct))

if __name__ == "__main__":
    test_with_pos()
    test_forward()
    test_backward()
