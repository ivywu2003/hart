import torch
from layernorm_kernels import rms_norm_pytorch

def test_rms_norm_metal():
    # Test parameters
    batch_size = 2
    seq_len = 32
    hidden_size = 64
    epsilon = 1e-6

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.ones(hidden_size)  # Initialize weights to 1 for simple testing
    output = torch.empty_like(input_tensor)

    try:
        # Call the Pytorch implementation
        output = rms_norm_pytorch(
            output,
            input_tensor,
            weight,
            epsilon,
            False
        )
        print("✅ Successfully called rms_norm_metal")
        
        # Basic validation
        if torch.isnan(output).any():
            print("❌ Output contains NaN values")
        else:
            print("✅ Output contains no NaN values")
            
        if torch.isinf(output).any():
            print("❌ Output contains Inf values")
        else:
            print("✅ Output contains no Inf values")
            
        print("\nOutput statistics:")
        print(f"Mean: {output.mean().item():.6f}")
        print(f"Std: {output.std().item():.6f}")
        print(f"Min: {output.min().item():.6f}")
        print(f"Max: {output.max().item():.6f}")

        print("\nOutput values:")
        print(output)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting RMS LayerNorm Pytorch test...")
    test_rms_norm_metal()