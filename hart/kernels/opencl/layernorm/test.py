import torch
# import rms_norm

def test_rms_norm_opencl():
    # Test parameters
    batch_size = 2
    seq_len = 32
    hidden_size = 64
    epsilon = 1e-6

    # Create test tensors on OpenCL device
    device = torch.device("xpu")
    print("success")
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight = torch.ones(hidden_size, device=device)  # Initialize weights to 1 for simple testing
    output = torch.empty_like(input_tensor)

    try:
        # Call the OpenCL implementation
        rms_norm(
            output,
            input_tensor,
            weight,
            epsilon,
            False
        )
        print("✅ Successfully called rms_norm_opencl")
        
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
    # Verify OpenCL is available
    # if not torch.backends.opencl.is_available():
    #     print("❌ OpenCL not available on this machine")
    #     exit(1)
        
    print("🚀 Starting OpenCL LayerNorm Metal test...")
    test_rms_norm_opencl()