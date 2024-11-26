#pragma once

#include <metal_stdlib>

namespace hart {

struct LayerNormParams {
    uint num_tokens;
    uint hidden_size;
    float epsilon;
    bool use_quant;
};

// Forward declarations for Metal kernel functions
void rms_norm_metal(const void* commandBuffer,
                   const void* input,      // [..., hidden_size]
                   const void* weight,     // [hidden_size]
                   void* output,           // [..., hidden_size]
                   const LayerNormParams& params);

} // namespace hart
