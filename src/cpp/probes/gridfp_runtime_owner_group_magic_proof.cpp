#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

constexpr int MAX_W = 28;
constexpr int W_MIN = 8;
constexpr int W_COUNT = 11;
constexpr int GROUP_SLOTS = 14;

constexpr std::uint64_t MAGIC[W_COUNT][GROUP_SLOTS] = {
    {1844674407370955162ULL,1229782938247303442ULL,768614336404564651ULL,472993437787424401ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {737869762948382065ULL,472993437787424401ULL,292805461487453201ULL,175683276892471921ULL,103633393672525571ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {288230376151711744ULL,180850432095191683ULL,109802048057794951ULL,65182841249857073ULL,37956263526151341ULL,21804662025661409ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {111124964299455131ULL,68321274347072414ULL,40901871560331601ULL,23987963684927896ULL,13848906962244409ULL,7889967525111015ULL,4448214148471076ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {42309046040618238ULL,25584943236767756ULL,15120282027630781ULL,8779982900385318ULL,5026360783027126ULL,2844524914989908ULL,1594084347883647ULL,886053320222372ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {15943599026542396ULL,9503732134832330ULL,5554575150168489ULL,3196455393122432ULL,1816518372595722ULL,1021584098892926ULL,569502147933363ULL,315086584229389ULL,173194228409896ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {5954404155490495ULL,3505652617580683ULL,2028898380302415ULL,1158278542867610ULL,653861621781850ULL,365658580592086ULL,202854139985370ULL,111761872318815ULL,61204342704315ULL,33340220777804ULL,0ULL,0ULL,0ULL,0ULL},
    {2206548334175784ULL,1285129167737882ULL,737338878955535ULL,417933392399057ULL,234512383342354ULL,130472642404442ULL,72060408897651ULL,39547098453660ULL,21583116185292ULL,11721208440321ULL,6337582406965ULL,0ULL,0ULL,0ULL},
    {812131023761097ULL,468524435479772ULL,266741050287894ULL,150220232200114ULL,83832070284623ULL,46421618107148ULL,25533946173675ULL,13962810792630ULL,7596016626770ULL,4113460587031ULL,2218431271056ULL,1192009815993ULL,0ULL,0ULL},
    {297116001573778ULL,169972210615782ULL,96099815964813ULL,53805379952601ULL,29877237258567ULL,16473042910439ULL,9026707167791ULL,4919643395903ULL,2668429550025ULL,1441185320762ULL,775381184929ULL,415722044609ULL,222188684832ULL,0ULL},
    {108119756137888ULL,61390512821765ULL,34493047954192ULL,19210155659623ULL,10618468963104ULL,5831325172161ULL,3184215117923ULL,1730046865227ULL,935781482669ULL,504143673942ULL,270625621320ULL,144798841098ULL,77244853275ULL,41095342785ULL}
};

std::uint64_t binom(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    std::uint64_t x = 1;
    for (int i = 1; i <= k; ++i)
        x = x * std::uint64_t(n - k + i) / std::uint64_t(i);
    return x;
}

std::array<std::array<std::uint64_t, MAX_W + 2>, MAX_W + 1> primitive_table() {
    std::array<std::array<std::uint64_t, MAX_W + 2>, MAX_W + 1> p{};
    p[0][0] = 1;
    for (int rem = 1; rem <= MAX_W; ++rem)
        for (int h = 0; h <= MAX_W; ++h)
            p[rem][h] = p[rem - 1][h + 1] + (h ? p[rem - 1][h - 1] : 0);
    return p;
}

std::uint64_t owner_group(
    const std::array<std::array<std::uint64_t, MAX_W + 2>, MAX_W + 1>& p,
    int W, int r
) {
    const int K = (W - 2) / 2;
    const int L = K + 2;
    const int O = W - L;
    if (r > O) return 0;
    std::uint64_t total = 0;
    for (int l = 0; l <= L - 1; ++l) {
        const int occupied = r + l;
        if (!(occupied & 1)) continue;
        total += (binom(L - 1, l) - binom(L - 3, l)) * p[occupied][1];
    }
    return total;
}

} // namespace

int main() {
    const auto primitive = primitive_table();
    std::uint64_t slots = 0, active = 0;
    std::uint64_t max_divisor = 0;
    for (int wi = 0; wi < W_COUNT; ++wi) {
        const int W = W_MIN + 2 * wi;
        for (int r = 0; r < GROUP_SLOTS; ++r) {
            const std::uint64_t d = owner_group(primitive, W, r);
            const std::uint64_t expected = d <= 1 ? 0 :
                std::numeric_limits<std::uint64_t>::max() / d + 1;
            if (MAGIC[wi][r] != expected) {
                std::cerr << "owner-group magic mismatch W=" << W
                          << " r=" << r << " divisor=" << d
                          << " got=" << MAGIC[wi][r]
                          << " expected=" << expected << '\n';
                return 2;
            }
            ++slots;
            if (d) {
                ++active;
                if (d > max_divisor) max_divisor = d;
            }
        }
    }
    if (slots != 154 || active != 99 || max_divisor != 448876754ULL)
        return 3;

    std::cout << "gridfp-runtime-owner-group-magic-proof OK"
              << " W_min=8 W_max=28 W_step=2"
              << " table_rows=11 group_slots=14"
              << " slots=" << slots
              << " active_divisors=" << active
              << " max_divisor=" << max_divisor
              << " table_bytes=" << sizeof(MAGIC)
              << " magic_exact=1\n";
    return 0;
}
