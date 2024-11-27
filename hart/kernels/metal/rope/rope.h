#pragma once

#include <torch/extension.h>

at::Tensor fused_rope_forward_metal(const at::Tensor &input,
                                  const at::Tensor &freqs,
                                  const bool transpose_output_memory);

at::Tensor fused_rope_backward_metal(const at::Tensor &output_grads,
                                   const at::Tensor &freqs,
                                   const bool transpose_output_memory);
