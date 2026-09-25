// Includes' hash value: e3afd8b449dc20f4594c8993c84b13d9

#include <deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm120_tf32_hc_prenorm_gemm_impl<
        24, 5120,
        128, 32, 64,
        16,
        4,
        256, 128
    >);
};
