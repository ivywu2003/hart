import torch
import hart_backend.fused_kernels as fused

def test_rms_norm_metal_basic():
    print("🚀 Starting basic RMS LayerNorm Metal test...")
    # Test parameters
    batch_size = 2
    hidden_size = 4

    input_tensor = torch.ones(batch_size, hidden_size).to(device="mps")
    weight = torch.ones(hidden_size).to(device="mps")
    epsilon = 1e-6
    output = torch.empty_like(input_tensor)

    try:
        # Call the Metal implementation
        fused.rms_norm(
            output,
            input_tensor,
            weight,
            epsilon,
            False
        )
        print("✅ Successfully called rms_norm_metal")

        torch.testing.assert_close(
            output,
            torch.ones_like(input_tensor),
        )
        print("✅ Successfully validated rms_norm_metal output")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    # Verify MPS is available
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this machine")
        exit(1)
        
    test_rms_norm_metal_basic()