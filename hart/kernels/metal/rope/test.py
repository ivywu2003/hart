import torch
import math
from hart_backend.fused_kernels import fused_rope_with_pos_forward_func, fused_rope_forward_func, fused_rope_backward_func

def get_rotary_matrix(freqs, dtype=torch.float32):
    """Helper function to generate rotary matrix from frequencies."""
    t = torch.tensor(freqs, dtype=dtype)
    cos = torch.cos(t).view(-1, 1)
    sin = torch.sin(t).view(-1, 1)
    zeros = torch.zeros_like(cos)
    rot_mat = torch.cat([
        torch.cat([cos, -sin], dim=1),
        torch.cat([sin, cos], dim=1)
    ], dim=0)
    return rot_mat

def test_fused_rope_with_pos():
    print("🚀 Starting RoPE forward with pos Metal test...")

    # Test parameters
    device = torch.device("mps")
    t = torch.load("../../testing/rope/rope_tests/forward_with_pos_t.pt").to(device)
    freqs = torch.load("../../testing/rope/rope_tests/forward_with_pos_freqs.pt").to(device)
    correct = torch.load("../../testing/rope/rope_tests/forward_with_pos_output.pt").to(device)

    output = fused_rope_with_pos_forward_func(t, freqs, False)
    print("✅ Successfully called fused_rope_with_pos_forward_func")

    print("Output values:")
    print(output)

    print("Correct values:")
    print(correct)

    torch.testing.assert_close(
        output,
        correct,
    )

    print("✅ Successfully validated fused_rope_with_pos_forward_func output")

def test_fused_rope():
    # Test parameters
    batch_size = 2
    seq_len = 4
    num_heads = 3
    head_dim = 8
    dtype = torch.float32

    device = torch.device("mps")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # Create position frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    freqs = freqs.to(device)
    
    # Run Metal implementation
    out_metal = fused_rope_forward_func(x, freqs, False)

    assert out_metal.shape == (batch_size, seq_len, num_heads, head_dim)
    print("✅ Successfully called fused_rope_forward_func")
    print("Output values:")
    print(out_metal)

def test_fused_rope_backward():
    # Test parameters
    batch_size = 2
    seq_len = 4
    num_heads = 3
    head_dim = 8
    dtype = torch.float32

    device = torch.device("mps")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # Create position frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    freqs = freqs.to(device)
    
    # Run Metal implementation
    out_metal = fused_rope_backward_func(x, freqs, False)

    assert out_metal.shape == (batch_size, seq_len, num_heads, head_dim)
    print("✅ Successfully called fused_rope_backward_func")
    print("Output values:")
    print(out_metal)

if __name__ == "__main__":
    test_fused_rope_with_pos()
