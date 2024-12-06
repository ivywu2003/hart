import torch

def rms_norm_pytorch(output, input, weight, epsilon, use_quant):
    if use_quant:
        input = input.float()

    rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + epsilon)
    norm_input =  input / rms
    output = weight * norm_input

    if use_quant:
        output = torch.quantize_per_tensor(output, scale=0.1, zero_point=0, dtype=torch.qint8)
    return output