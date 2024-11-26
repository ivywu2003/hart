#pragma once

#include <metal_stdlib>

// Metal dispatch macros for type handling
#define HART_DISPATCH_CASE_FLOATING_TYPES(TYPE, NAME, FUNC) \
    switch (TYPE) { \
        case MTLDataTypeFloat: FUNC<float>(); break; \
        case MTLDataTypeHalf: FUNC<half>(); break; \
        default: assert(false && "Unsupported data type"); \
    }

#define HART_DISPATCH_FLOATING_TYPES(TYPE, NAME, FUNC) \
    HART_DISPATCH_CASE_FLOATING_TYPES(TYPE, NAME, FUNC)

// Helper macros for Metal kernel declarations
#define HART_METAL_KERNEL(NAME) \
    kernel void NAME( \
        uint thread_position_in_grid [[thread_position_in_grid]], \
        uint threads_per_grid [[threads_per_grid]], \
        uint thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
        uint threads_per_threadgroup [[threads_per_threadgroup]])

// Helper macros for Metal buffer bindings
#define HART_METAL_BUFFER(TYPE, NAME, INDEX) \
    device TYPE* NAME [[buffer(INDEX)]]

// Helper macros for Metal texture bindings
#define HART_METAL_TEXTURE(TYPE, NAME, INDEX) \
    texture2d<TYPE> NAME [[texture(INDEX)]]
