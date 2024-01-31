#include <iostream>
#include <cassert>
#include <vector>
#include "avx2bitset.hpp"

int main() {
	auto tmp = avx2bitset<63>();
	auto tmp0 = avx2bitset<64>();
	auto tmp1 = avx2bitset<256>();
	auto tmp2 = avx2bitset<257>();
	auto tmp3 = avx2bitset<512>();
	auto tmp4 = avx2bitset<513>();
	assert(sizeof(tmp0) == 32);
	assert(sizeof(tmp1) == 32);
	assert(sizeof(tmp2) == 64);
	assert(sizeof(tmp3) == 64);
	assert(sizeof(tmp4) == 96);

	assert(tmp0.size() == 64);
	assert(tmp1.size() == 256);
	assert(tmp2.size() == 257);
	assert(tmp3.size() == 512);
	assert(tmp4.size() == 513);

	std::cout << tmp0 << std::endl;
	tmp0[0] = true;
	tmp0[2] = true;
	tmp0[61] = true;
	tmp0[63] = true;
	std::cout << tmp0 << std::endl;

	auto tmp5 = tmp0;
	assert(tmp5.size() == 64);
	std::cout << tmp5 << std::endl;
	std::cout << ~tmp5 << std::endl;
	std::cout << (tmp5 | ~tmp5) << std::endl;
	std::cout << (tmp5 & ~tmp5) << std::endl;
	std::cout << (tmp5 ^ ~tmp5) << std::endl;
	tmp5.flip();
	std::cout << tmp5 << std::endl;
	tmp5.flip(2);
	std::cout << tmp5 << std::endl;
	std::cout << tmp5.count() << std::endl;
	std::cout << (~tmp5).count() << std::endl;
	std::cout << (~tmp2).count() << std::endl;
	std::cout << (~tmp).count() << std::endl;
	std::cout << (~tmp3).count() << std::endl;
	std::cout << (~tmp4).count() << std::endl;
	assert(tmp5 == tmp5);
	assert(not(tmp5 != tmp5));
	assert(not(tmp5 == ~tmp5));
}
