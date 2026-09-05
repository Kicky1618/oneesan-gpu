#pragma once

#include "ramstream32_bucket_direct_high_row_affine.cuh"

// Experimental fixed-owner LOW ternary keys.  Each GPU binds exactly one LOW
// owner for the lifetime of its graph, so duplicating all BUCKET_NGPU owner
// slices on every GPU is unnecessary.  Pack only the bound owner's per-height
// ranges and keep their compact offsets in constant memory.
__constant__ uint32_t* D_P10DC_LOW_PREKEY;
__constant__ uint32_t D_P10DC_LOW_PREKEY_HOFF[MAXW + 2];

__device__ __forceinline__ uint32_t p10dc_low_prekey_fixed(uint32_t h, uint32_t rank) {
    return D_P10DC_LOW_PREKEY[D_P10DC_LOW_PREKEY_HOFF[h] + rank];
}

struct BucketFusedDirectHighRowsPrekeyTables {
    BucketFusedDirectHighRowsTables base;
    const BucketFusedHost* host_fused = nullptr;
    uint32_t* low_prekey = nullptr;
    size_t low_prekey_count = 0;
    size_t low_prekey_capacity = 0;

    void install_metadata(
        const StorageLayout& layout, const BucketOrbitStreamsHost& o, const BucketFusedHost& f
    ) {
        base.install_metadata(layout, o, f);
        host_fused = &f;
        constexpr size_t P = size_t(MAXW + 2);
        if (f.low_code_off.size() < size_t(BUCKET_NGPU) * P) {
            std::cerr << "p10dc LOW prekey offset table too small got=" << f.low_code_off.size()
                      << " expected_at_least=" << size_t(BUCKET_NGPU) * P << '\n';
            std::exit(620);
        }
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        base.bind_owner(fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc LOW prekey invalid bind owner=" << fixed << '\n';
            std::exit(621);
        }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_begin = f.low_code_off[owner_base];
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());
        if (owner_begin > owner_end || owner_end > f.low_codes.size()) {
            std::cerr << "p10dc LOW prekey owner range invalid owner=" << fixed
                      << " begin=" << owner_begin << " end=" << owner_end
                      << " total=" << f.low_codes.size() << '\n';
            std::exit(622);
        }

        std::array<uint32_t, MAXW + 2> hoff{};
        std::vector<uint32_t> key;
        key.reserve(size_t(owner_end - owner_begin));
        uint32_t max_key = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            hoff[h] = uint32_t(key.size());
            uint32_t a = f.low_code_off[owner_base + h];
            uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            if (a < owner_begin || a > b || b > owner_end) {
                std::cerr << "p10dc LOW prekey height range invalid owner=" << fixed
                          << " h=" << h << " a=" << a << " b=" << b
                          << " owner_begin=" << owner_begin << " owner_end=" << owner_end << '\n';
                std::exit(623);
            }
            for (uint32_t i = a; i < b; ++i) {
                uint32_t k = gpu_direct_ternary_key_host(f.low_codes[i], LOW_LUT_K);
                key.push_back(k);
                max_key = std::max(max_key, k);
            }
        }
        if (key.size() != size_t(owner_end - owner_begin)) {
            std::cerr << "p10dc LOW prekey compact size mismatch owner=" << fixed
                      << " got=" << key.size() << " expected=" << (owner_end - owner_begin) << '\n';
            std::exit(624);
        }

        low_prekey_count = key.size();
        if (low_prekey_count > low_prekey_capacity) {
            if (low_prekey) cudaFree(low_prekey);
            low_prekey = nullptr;
            low_prekey_capacity = low_prekey_count;
            if (low_prekey_capacity)
                ck(cudaMalloc(&low_prekey, low_prekey_capacity * sizeof(uint32_t)),
                   "p10dc compact LOW prekey alloc");
        }
        if (!key.empty())
            ck(cudaMemcpy(low_prekey, key.data(), key.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "p10dc compact LOW prekey H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &low_prekey, sizeof(low_prekey)),
           "p10dc compact LOW prekey ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY_HOFF, hoff.data(), hoff.size() * sizeof(uint32_t)),
           "p10dc compact LOW prekey height offsets");

        std::cerr << "p10dc_low_prekey fixed_owner=" << fixed
                  << " entries=" << low_prekey_count
                  << " full_entries=" << f.low_codes.size()
                  << " mib=" << double(low_prekey_count * sizeof(uint32_t)) / double(1 << 20)
                  << " full_mib=" << double(f.low_codes.size() * sizeof(uint32_t)) / double(1 << 20)
                  << " max_key=" << max_key
                  << " fold_runtime=0 hot_code_load=0 hot_code_off_load=0\n";
    }

    void release() {
        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr;
        low_prekey_count = 0;
        low_prekey_capacity = 0;
        host_fused = nullptr;
        base.release();
    }
};
