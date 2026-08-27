#include "../ramstream32_bucket_orbit_closure_pattern10_depth8_lut.cuh"

struct BucketForwardPattern10Depth8LutTestDeviceTables : BucketForwardPattern10Depth8DeviceTables {
    BucketPattern10DecodeLutHost host_lut;
    BucketPattern10DecodeLutDeviceTables device_lut;
    void install(const BucketForwardPattern10Depth8Host& h) {
        BucketForwardPattern10Depth8DeviceTables::install(h);
        host_lut = build_bucket_pattern10_decode_lut();
        device_lut.install(host_lut);
    }
    void release() {
        device_lut.release();
        BucketForwardPattern10Depth8DeviceTables::release();
    }
};

#define BucketForwardPattern10Depth8DeviceTables BucketForwardPattern10Depth8LutTestDeviceTables
#define main pattern10_depth8_lut_selftest_main
#include "ramstream32_bucket_orbit_closure_pattern10_depth8_selftest.cu"
#undef main
#undef BucketForwardPattern10Depth8DeviceTables

int main() {
    int rc = pattern10_depth8_lut_selftest_main();
    if (!rc) std::cout << "bucket-closure-pattern10-depth8-lut-selftest OK decode_lut=1\n";
    return rc;
}
