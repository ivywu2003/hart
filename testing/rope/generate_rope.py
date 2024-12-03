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
    output_tensor = fused.fused_rope_with_pos_forward_func(t, freqs, transpose_mem)

    t = t.to('cpu')
    freqs = freqs.to('cpu')
    output_tensor = output_tensor.to('cpu')

    print(t)
    print(freqs)
    print(output_tensor)

    # in_name = "tests/in_{}{}.pt".format(test_num, "_quant" if use_quant else "")
    # out_name = "tests/out_{}{}.pt".format(test_num, "_quant" if use_quant else "")
    torch.save(t, "forward_with_pos_t.pt")
    torch.save(freqs, "forward_with_pos_freqs.pt")
    torch.save(output_tensor, "forward_with_pos_output.pt")

if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [(2, 2, 2, 2)]

    for i, test in enumerate(tests):
        test_rope(*test, False, i)

