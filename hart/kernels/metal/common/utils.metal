#include <metal_stdlib>

using namespace metal;

namespace hart {

// Type converters for Metal
template<typename T>
struct FloatTypeConverter {
    using Type = T;
};

template<>
struct FloatTypeConverter<half> {
    using Type = half;
};

template<>
struct FloatTypeConverter<float> {
    using Type = float;
};

// Number of elements in vector types
template<typename T>
struct num_elems {
    static constexpr int value = 1;
};

template<>
struct num_elems<float2> {
    static constexpr int value = 2;
};

template<>
struct num_elems<float4> {
    static constexpr int value = 4;
};

template<>
struct num_elems<half> {
    static constexpr int value = 1;
};

template<>
struct num_elems<half2> {
    static constexpr int value = 2;
};

// Packed type definitions
template<typename T, int num>
struct packed_as {
    using type = T;
};

template<>
struct packed_as<half, 2> {
    using type = half2;
};

template<>
struct packed_as<float, 2> {
    using type = float2;
};

template<>
struct packed_as<float, 4> {
    using type = float4;
};

// Metal casting functions
template<typename To, typename From>
inline To metal_cast(From val);

template<>
inline float2 metal_cast<float2, int2>(int2 val) {
    return float2(float(val.x), float(val.y));
}

template<>
inline float metal_cast<float, int>(int val) {
    return float(val);
}

template<>
inline half metal_cast<half, float>(float val) {
    return half(val);
}

template<>
inline float metal_cast<float, half>(half val) {
    return float(val);
}

// Metal absolute value functions
template<typename T>
inline T metal_abs(T val) {
    return abs(val);
}

// Specialized abs functions for vector types
template<>
inline float2 metal_abs(float2 val) {
    return abs(val);
}

template<>
inline float4 metal_abs(float4 val) {
    return abs(val);
}

// Metal math functions
template<typename T>
inline T metal_max(T a, T b) {
    return max(a, b);
}

template<typename T>
inline T metal_min(T a, T b) {
    return min(a, b);
}

// Vector type constructors
inline float2 make_float2(float x, float y) {
    return float2(x, y);
}

inline float4 make_float4(float x, float y, float z, float w) {
    return float4(x, y, z, w);
}

inline half2 make_half2(half x, half y) {
    return half2(x, y);
}

} // namespace hart
