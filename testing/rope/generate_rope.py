import torch
import hart_backend.fused_kernels as fused

def test_rope(s, b, h, d, transpose_mem=False, test_num=0):
    device = torch.device("cuda")

    # input 
    t = torch.randn((s, b, h, d), device='cpu')
    freqs = torch.randn((s, 1, 1, d), device='cpu')
    t = t.to(device)
    freqs = freqs.to(device)
    # t = torch.randn((s, b, h, d), device=device)
    # freqs = torch.randn((s, 1, 1, d), device=device)
    with_pos = fused.fused_rope_with_pos_forward_func(t, freqs, transpose_mem)
    forward = fused.fused_rope_forward_func(t, freqs, transpose_mem)
    backward = fused.fused_rope_backward_func(t, freqs, transpose_mem)

    t = t.to('cpu')
    freqs = freqs.to('cpu')
    with_pos = with_pos.to('cpu')
    forward = forward.to('cpu')
    backward = backward.to('cpu')

    print(t)
    print(freqs)
    print(with_pos)
    print(forward)
    print(backward)

    # in_name = "tests/in_{}{}.pt".format(test_num, "_quant" if use_quant else "")
    # out_name = "tests/out_{}{}.pt".format(test_num, "_quant" if use_quant else "")
    torch.save(t, "rope_tests/t.pt")
    torch.save(freqs, "rope_tests/freqs.pt")
    torch.save(with_pos, "rope_tests/forward_with_pos_output.pt")
    torch.save(forward, "rope_tests/forward_output.pt")
    torch.save(backward, "rope_tests/backward_output.pt")

if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [(2, 2, 2, 2)]

    for i, test in enumerate(tests):
        test_rope(*test, False, i)

