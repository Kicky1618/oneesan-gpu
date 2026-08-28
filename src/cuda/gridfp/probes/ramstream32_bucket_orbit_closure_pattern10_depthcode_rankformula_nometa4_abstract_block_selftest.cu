#pragma push_macro("main")
#undef main
#define main p10dc_rankformula_nometa4_abstract_base_selftest_main
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_rankformula_nometa4_abstract_selftest.cu"
#pragma pop_macro("main")

int main() {
    const int rc = p10dc_rankformula_nometa4_abstract_base_selftest_main();
    if (rc == 0) {
        std::cout << "rankformula-nometa4-abstract-block-selftest compiled_block="
                  << P10DC_RANKFORMULA_NOMETA4_BLOCK
                  << " warpshare=" << P10DC_RANKFORMULA_NOMETA_WARPSHARE
                  << " coopgroup=" << P10DC_RANKFORMULA_NOMETA_COOPGROUP
                  << " max_locator_steps_bound="
                  << (P10DC_RANKFORMULA_NOMETA4_BLOCK - 1u)
                  << " wrapper_ok=1\n";
    }
    return rc;
}
