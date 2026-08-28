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
    {1537228672809129302ULL,1024819115206086201ULL,658812288346769701ULL,401016175515425036ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {614891469123651721ULL,401016175515425036ULL,249280325320399347ULL,151202820276307801ULL,89114705670094453ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {242720316759336206ULL,153722867280912931ULL,94116041192395672ULL,56069130923129337ULL,32823388031511658ULL,18881007240234956ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {94116041192395672ULL,58375772385156809ULL,35136655378494385ULL,20703416468809823ULL,11986188481942529ULL,6849886399446548ULL,3868052856722490ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {36028797018963968ULL,21934297352805650ULL,13027361633975673ULL,7591252705230269ULL,4358871472993751ULL,2472091138261801ULL,1388225773156951ULL,772735592900032ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {13633957186777201ULL,8173125420340963ULL,4796345312977003ULL,2768949875969612ULL,1577453743262319ULL,888999714395642ULL,496413995525015ULL,275061792820434ULL,151374046656953ULL,0ULL,0ULL,0ULL,0ULL,0ULL},
    {5109901405459710ULL,3022569895741366ULL,1755495248735207ULL,1004943564704160ULL,568571818324176ULL,318541600305812ULL,176991327081187ULL,97640049933358ULL,53532752370105ULL,29189826022789ULL,0ULL,0ULL,0ULL,0ULL},
    {1899180899177345ULL,1110513760382250ULL,639091743130182ULL,363124883340740ULL,204160790598198ULL,113774680657417ULL,62925956246665ULL,34575932541058ULL,18889785093288ULL,10268054132333ULL,5556377082755ULL,0ULL,0ULL,0ULL},
    {700757638417777ULL,405645828998561ULL,231556839647891ULL,130685237922479ULL,73059884326026ULL,40517004932547ULL,22314681749429ULL,12215901403726ULL,6652116241713ULL,3605385867581ULL,1945904237621ULL,1046285878837ULL,0ULL,0ULL},
    {256922019439123ULL,147410032633389ULL,83538226384215ULL,46861727340349ULL,26063116474952ULL,14389619605249ULL,7894275949453ULL,4306843339981ULL,2338137066104ULL,1263802738648ULL,680428731515ULL,365046118187ULL,195216628579ULL,0ULL},
    {93667773988309ULL,53320915818179ULL,30021065771100ULL,16748238240762ULL,9271025105900ULL,5097655734952ULL,2786581213604ULL,1515428102977ULL,820375738534ULL,442296973144ULL,237583780436ULL,127195807467ULL,67891130727ULL,36136888735ULL}
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

std::uint64_t turn_compress_group(
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
        total += binom(L - 1, l) * p[occupied][1];
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
            const std::uint64_t d = turn_compress_group(primitive, W, r);
            const std::uint64_t expected = d <= 1 ? 0 :
                std::numeric_limits<std::uint64_t>::max() / d + 1;
            if (MAGIC[wi][r] != expected) {
                std::cerr << "turn-compress magic mismatch W=" << W
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
    if (slots != 154 || active != 99 || max_divisor != 510468519ULL)
        return 3;

    std::cout << "gridfp-runtime-turn-compress-group-magic-proof OK"
              << " W_min=8 W_max=28 W_step=2"
              << " table_rows=11 group_slots=14"
              << " slots=" << slots
              << " active_divisors=" << active
              << " max_divisor=" << max_divisor
              << " table_bytes=" << sizeof(MAGIC)
              << " magic_exact=1\n";
    return 0;
}
