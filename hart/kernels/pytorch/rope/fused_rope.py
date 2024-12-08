import torch

def fused_rope_block_forward(src: torch.Tensor, freqs: torch.Tensor, d: int, d2: int):
    batch_size, num_heads, seq_len, _ = src.shape
    freqs = freqs.view(1, 1, seq_len, d2)

    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)

    # (batch_size, num_heads, seq_len, d2)
    src_1 = src[:, :, :, :d2] 
    src_2 = src[:, :, :, d2:2 * d2]

    dst_1 = src_1 * cos_freqs - src_2 * sin_freqs
    dst_2 = src_1 * sin_freqs + src_2 * cos_freqs
    dst = torch.cat([dst_1, dst_2], dim=-1)

    if d > 2 * d2:
        dst = torch.cat([dst, src[:, :, :, 2 * d2:]], dim=-1)

    return dst

def fused_rope_block_backward(src: torch.Tensor, freqs: torch.Tensor, d: int, d2: int):
    batch_size, num_heads, seq_len, _ = src.shape

    src_1 = src[:, :, :, :d2] 
    src_2 = src[:, :, :, d2:2 * d2]

    # (seq_len, 1, 1, d2)
    cos_freqs = torch.cos(freqs).unsqueeze(1).unsqueeze(1)  
    sin_freqs = torch.sin(freqs).unsqueeze(1).unsqueeze(1)

    dst = src.clone()
    dst[..., :d2] = src_1 * cos_freqs + src_2 * sin_freqs
    dst[..., d2:2 * d2] = -src_1 * sin_freqs + src_2 * cos_freqs

    return dst