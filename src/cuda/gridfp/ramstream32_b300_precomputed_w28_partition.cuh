#pragma once

#include "ramstream32_b300_direct_mask_partition_opt.cuh"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Offline W28 / HIGH13 / 8-GPU partition.
// Seed: recursive normalized-Laplacian bisection (3 levels), followed by
// lambda=1, +/-4 GiB single-move and 512-candidate pair-swap refinement.
// Reconstructed exact metrics from the transition graph:
//   orbit cut        = 0.331336
//   closure cut      = 0.357321
//   auth range       = 238.685659 .. 246.441913 GiB
//   work max / avg   = 1.058527
// The 8192 owner bytes are RLE-compressed, then base64 encoded.  This keeps the
// checked-in table small while making runtime decode deterministic and trivial.
static int b300_w28_b64v(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static std::vector<uint8_t> b300_w28_precomputed_high_mask_owner() {
    static const char enc[] =
        "QAQQBwIEBAcBBAkHEAQIBwMEJQcEBBwHEAQIBwIEFgcBBE8HCAQ4BxAEBAcBBAMHBAQCBwIEEAcEBCwHAgSeBwgEAQcHBAgHAgQW"
        "BwEELwcBBB8HIAQIBwIEBgcIBAMHBQQQBwEEDwcHBAUHAQQzBwEEHwcIBAQHAQRzBwUEAQcCBAQHAgQKBwEEJwcQBAIHAQQBBwME"
        "AQcIBAgHAQQHBwMEAwcBBAUHAQQjBwQEAgcBBBkHIAQEBwIEAgcYBBAHBAQMBwcEAwcBBAEHAQQLBwEEJwcBBB8HCwQBBwEECwcB"
        "BBcHAQQPB0AECAcEBAMHCQQDBxUEAQcPBBAHAQQPBwcEGQcPBAMHAQQBBwEEAwcBBA8HAQQHBwEELwcCBB4HCAQCBwEEAQcBBAsH"
        "AQQXBwEEDwcQBAEHBgQBBwgECAcBBAcHAQQfBwEEDwcBBB8HCAIBBwcCAgQBAgEFAgIBBwUCAQcDAgMEAQUBAgMFAgIBBAECAQQC"
        "BQMCAQcBAgEHAwIBBAEFBgIDBAEFAQQDBQEEBwUBBAEFAQINBQQCAQcDAgEEAgUFAggFAgIBBQECAQUDAgcEAQUDBAEFAQQDBQME"
        "AQUBBAMFAQQHBQMEAQUBBAsFAQQPBQgCAQQBBQECAQUEAggFAgIBBQECAQUDAgEEDwUBAg8FFwQBBQUEAgUIBAEFAwQBBQEEAwUD"
        "BAEFAQQDBQEEBwUCBgEEAQYBBAEFAQQBBQEEAwUBBgMFAQQHBQEGBwUDBA0FAQQPBQgGAQQHBggFAwYFBQEEDwUCBg4FAQQfBQIG"
        "HgWABAgGAgQGBgMEAQUBBAMFAQYBBQEEBQUDBAEFAQQDBQEEBwUBBg8FCQQBBQEEAQUBBAMFAQQDBQEEAwUBBAcFAQQBBQEEDQUB"
        "BA8FEAYBBAMFAQYCBQkGAQQHBQEGBwUGBgYFAQYDBQMEAQUBBAsFAQQPBQUGBwUBBhMFCgQCBQEEAwUBBAcFAQQHBQEEDwUBBA8F"
        "CAYQBQEGJwWABAgGBAQEBggEAwYBBAEGEwQDBgkEAQZDBBAGAQQPBggEBAYBBBMGIAQIBgEEBwYEBAQFCAZABBAGAQQPBgkEBwUI"
        "BgQFFAYBBA8GAwQBBQEEAgUFBgEECwYBBAcGAwQBBQEEAwUBBAcFAQYBBQEGDQUIBgEEBwYIBQgGEwQBBQEEAwUBBAcFAgYBBAEF"
        "AQQLBQEEDwUIBgEEBwYIBQgGAQQPBQEGDwVABAQGAQQDBgIEBgUCBAYFAQQHBQIEBgUBBAcFAQQPBRAGAQQPBgEEDwUIBgQFBAYC"
        "BB4FBAYCBQIGBAUEBhAF/wL/AgICBQQCAgIEAwIBBAMCAQQDAgEEAwIBBBcCAQQvAgEE/wKeAgIDRwQBAwMEAQcCBAIDBwQBAwME"
        "AQMBBAMDBwQBAwMEAQMBBAMDAwQBAwEEAwMBBAcDAQcDAwECAwMBBAEDAQIFAwEEDwMHBAEDAQQDAwEEEwMHBAEDAwQBAwEEAwMD"
        "BAEDAQQDAwEEBwMBAgcDAQQHAwEEDwNAAgIEBgIBAwECAgMBAgsDAQIHAwICBgMBAgcDAQIPAwsEAQMBBAcDAQQDAwEEBwMBAg8D"
        "AQQvAwEEHwN3AgEDAwIBAwECAwMBBCYCAQMDAgEDAQIDAwMCAQMBAgMDAQIHAwcCAQMDAgEDAQIDAwMCAQMBAgMDAQIHAwMCAQMB"
        "AgMDAQIHAwECDwMHBAEDAwQBAwEEAwMBAgcDAQQHAwICBgMBAgcDAQQPAwMCBQMBAgcDAQIPAwECHwMDAgEDAgICAwMCAQMBAgMD"
        "AwIBAwECAwMBAgcDAQIPAwECDwMBAj8DIAQBBx8EAwcBBgIHAgYFBwEGAQcBBgYEAQcBBgEHAQYBBwEGAQcDBhAEAwcBBgIHAgYB"
        "BAoHAQYBBwMGBQcDBgcHAQYBBwMGAQcDBgEEBAcBBgEHAQYBBwEGAQcBBgEHAwYBBwMGAQcDBgIHBgYIBAEHAgQBBgEEAwYBBwMG"
        "AQcDBgEEBwYBBwMGAgcCBgQHBAYBBBAHAwYBBwMGAgcCBgEHAwYFBwMGAQcDBgEHAwYFBwEGAQcBBgEHAQYBBwEGAQcDBgEHAwYB"
        "BwMGAgcGBgMEAQYBBwMGAQcDBgEHAwYBBwMGAQcDBgEHBwYBBwMGAQcDBgIHBgYCBw4GEAQBBwEGAQcBBgEHAwYBBAcGAQcDBgEH"
        "AwYBBwcGAQQPBgEHAwYBBwMGAgcGBgIHDgYBBBEGAQcBBgIHAgYIBwECAQABAgEAAQIDAAEHBwACBwECAQcBAgsAAwcBAgEHAQYB"
        "AgEGAQIDAAECAwABAgcAAQIHAAEEBwYBBwcGAQcBAgIAAQcDAAEHBwABAgcAAQIHAAEHDwAHBAEGAQcDBgEEAwYBBw8GAQcfBgEC"
        "AwABAgMAAQcHAAIHDgABBwgGFwAgBAgGAQQHBgEEFwYBBwcGAQcPBgEEHwYBAgMAAQIDAAEHBwADBwEAAQILAAEHDwYQAAEEHwYQ"
        "AAgGBAcBAwMHAwIBAAECAwABAgcAAwIBAAECCwADAgEAAQIDAAECBwABAg8AEQIHAAECBwABAgcAAQIHAAECDwABBCICAQABAgMA"
        "AQIHAAMCAQABAgsAAQIHAAECBwACAg4AAwIBAAECGwAHBAECAwQBAgEEAwIBBAMCAQQDAgEEFwIBBBICAQABAgMAAgICAAECAwAF"
        "AgMAAQIHAAEEBAIDAAECBwABAg8AAQIHAAECBwACAg4ABAIcAAEEBQICAAICBgABAi0AAgFABAECPwQQAgQEAQICBAcCAgEIBAEC"
        "BQQCAQECAgQBAQEEAwEBBAcBEwQBAQEEAwEBBAcBAQIHAQEEBwEBBA8BAgIOAAECDwABAhsABAEEBAECAwQBAgcEAQIDAQEHAwEB"
        "BwcBAQIfARAEAQIPBAECHwEBAj8B";

    std::vector<uint8_t> raw;
    raw.reserve(2200);
    const std::string s(enc);
    if (s.size() % 4) std::exit(492);
    for (size_t i = 0; i < s.size(); i += 4) {
        int a = b300_w28_b64v(s[i]), b = b300_w28_b64v(s[i + 1]);
        int c = b300_w28_b64v(s[i + 2]), d = b300_w28_b64v(s[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) std::exit(493);
        uint32_t x = (uint32_t(a) << 18) | (uint32_t(b) << 12) | (uint32_t(c) << 6) | uint32_t(d);
        raw.push_back(uint8_t(x >> 16)); raw.push_back(uint8_t(x >> 8)); raw.push_back(uint8_t(x));
    }
    if (raw.size() != 2196 || raw.size() % 2) std::exit(494);

    std::vector<uint8_t> owner;
    owner.reserve(1u << 13);
    for (size_t i = 0; i < raw.size(); i += 2) {
        uint8_t n = raw[i], g = raw[i + 1];
        if (!n || g >= 8) std::exit(495);
        owner.insert(owner.end(), n, g);
    }
    if (owner.size() != (1u << 13)) std::exit(496);
    return owner;
}

static B300DirectMaskShardHost build_b300_direct_mask_shards_w28_precomputed(
    const StorageFactorHost& storage, const StorageLayout& layout, int ngpu
) {
    if constexpr (TARGET_W == 28 && HIGH_LUT_K == 13 && LOW_LUT_K == 14) {
        if (ngpu == 8)
            return b300_build_mask_shard_from_owner(
                storage, layout, ngpu, b300_w28_precomputed_high_mask_owner());
    }
    return build_b300_direct_mask_shards(storage, layout, ngpu);
}
