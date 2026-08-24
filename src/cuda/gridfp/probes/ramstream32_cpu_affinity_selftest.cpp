#include "../ramstream32_cpu_affinity.hpp"

#include <iostream>
#include <vector>

static bool eq(const std::vector<int>& a, std::initializer_list<int> b) {
    return a == std::vector<int>(b);
}

int main() {
    if (!eq(cpu_high_parse_cpu_list("0-2, 5,7-8"), {0,1,2,5,7,8})) return 1;
    if (!eq(cpu_high_parse_cpu_list(" 3 "), {3})) return 2;
    if (!cpu_high_parse_cpu_list("").empty()) return 3;
    if (!cpu_high_parse_cpu_list(nullptr).empty()) return 4;
    std::cout << "cpu-high-affinity-selftest OK\n";
    return 0;
}
