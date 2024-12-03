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
        print("Input values:")
        print(input_tensor)

        # Call the Metal implementation
        fused.rms_norm(
            output,
            input_tensor,
            weight,
            epsilon,
            False
        )
        print("✅ Successfully called rms_norm_metal")

        print("Output values:")
        print(output)

        torch.testing.assert_close(
            output,
            torch.ones_like(input_tensor),
        )
        print("✅ Successfully validated rms_norm_metal output")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

def test_rms_norm_metal_edge():
    # Must manually set max number of threads to 3 for this test to fully work
    print("🚀 Starting RMS LayerNorm Metal test where hidden_size > threads_per_threadgroup...")
    # Test parameters
    batch_size = 3
    hidden_size = 5

    input_tensor = torch.ones(batch_size, hidden_size)
    input_tensor[1] *= 2
    input_tensor[2] *= 3
    input_tensor[2][0] *= 4
    input_tensor = input_tensor.to(device="mps")
    weight = torch.ones(hidden_size).to(device="mps")
    epsilon = 1e-6
    output = torch.empty_like(input_tensor)

    try:
        print("Input values:")
        print(input_tensor)

        # Call the Metal implementation
        fused.rms_norm(
            output,
            input_tensor,
            weight,
            epsilon,
            False
        )
        print("✅ Successfully called rms_norm_metal")

        print("Output values:")
        print(output)

        expected_output = torch.tensor(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [2.0000, 0.5000, 0.5000, 0.5000, 0.5000]], device='mps:0')

        torch.testing.assert_close(
            output,
            expected_output,
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
    test_rms_norm_metal_edge()