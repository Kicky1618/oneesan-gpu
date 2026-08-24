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

    if (!eq(cpu_low_parse_cpu_list("1,4-6"), {1,4,5,6})) return 5;
    if (!eq(cpu_low_parse_cpu_list(" 2-3, 9 "), {2,3,9})) return 6;
    if (!cpu_low_parse_cpu_list("").empty()) return 7;
    if (!cpu_low_parse_cpu_list(nullptr).empty()) return 8;

    if (!eq(cpu_parse_cpu_list("TEST_CPU_LIST", "10-11,13"), {10,11,13})) return 9;

    std::cout << "cpu-affinity-selftest OK\n";
    return 0;
}
