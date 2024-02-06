#pragma once

#if defined(__AVX2__)

#include <immintrin.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <bitset>

#define ymmlen(x) ((x / 256) + ((x % 256) != 0))
#define u64len(x) ((x / 64) + ((x % 64) != 0))

union ymmunit_t {
	__m256i v256;
	uint64_t v64[4];
	uint32_t v32[8];
	uint16_t v16[16];
	uint8_t v8[32];
};

class binary_proxy {
private:
public:
	ymmunit_t *ptr;
	size_t pos;
	binary_proxy(ymmunit_t &data, size_t p) : ptr(&data), pos(p) {}
	void operator = (bool val) {
		if (val)
			this->ptr->v8[this->pos / 8] |= (uint64_t)1 << (this->pos % 8);
		else
			this->ptr->v8[this->pos / 8] &= ~((uint64_t)1 << (this->pos % 8));
	}
};

inline std::ostream& operator<<(std::ostream& ost, const binary_proxy& prox) {
	ost << (prox.ptr->v8[prox.pos / 8] & (1 << (prox.pos % 8)));
	return ost;
}

template<size_t len>
class avx2bitset {
private:
	friend class binary_proxy;
	ymmunit_t data[ymmlen(len)];

	void range_check(size_t pos) {
		if (pos >= this->size()) {
			std::stringstream ss;
			ss << "avx2bitset::range_check: pos (which is " << pos << ") "
			   << "this->size() (which is " << this->size() << ")";
			throw std::out_of_range(ss.str());
		}
	}
public:
	avx2bitset() {
		this->set();
	}

	avx2bitset(const avx2bitset<len>& src) {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = src.data[i].v256;
		}
	}

	size_t size() const {
		return len;
	}

	binary_proxy at(size_t pos) {
		this->range_check(pos);
		return binary_proxy(this->data[pos / 256], pos % 256);
	}

	void set() noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_setzero_si256();
		}
	}

	void reset() noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_set1_epi8(0x00);
		}
	}

	avx2bitset<len>& flip() noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_xor_si256(this->data[i].v256, _mm256_set1_epi8(0xff));
		}
		return *this;
	}

	avx2bitset<len>& flip(size_t pos) {
		this->range_check(pos);
		this->data[pos / 256].v8[(pos % 256) / 8] ^= 1 << (pos % 8);
		return *this;
	}

	size_t count() const noexcept {
		uint64_t cnt = 0;
		for (size_t i = 0; i < u64len(len) - 1; i++) {
			size_t ymmidx = i / 4;
			size_t u64idx = i % 4;
			cnt += _mm_popcnt_u64(this->data[ymmidx].v64[u64idx]);
		}
		if constexpr (len % 64) {
			constexpr size_t mask = ((uint64_t)1 << (len % 64)) - 1;
			return cnt + _mm_popcnt_u64(this->data[ymmlen(len) - 1].v64[u64len(len) % 4 - 1] & mask);
		}
		return cnt + _mm_popcnt_u64(this->data[ymmlen(len) - 1].v64[u64len(len) % 4 - 1]);
	}

	binary_proxy operator[](size_t pos) noexcept {
		return binary_proxy(this->data[pos / 256], pos % 256);
	}

	avx2bitset<len> operator~() const noexcept {
		avx2bitset<len> tmp;
		for (size_t i = 0; i < ymmlen(len); i++) {
			tmp.data[i].v256 = _mm256_xor_si256(this->data[i].v256, _mm256_set1_epi8(0xff));
		}
		return tmp;
	}

	avx2bitset<len>& operator|=(const avx2bitset<len>& operand) noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_or_si256(this->data[i].v256, operand.data[i].v256);
		}
		return *this;
	}

	avx2bitset<len>& operator&=(const avx2bitset<len>& operand) noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_and_si256(this->data[i].v256, operand.data[i].v256);
		}
		return *this;
	}

	avx2bitset<len>& operator^=(const avx2bitset<len>& operand) noexcept {
		for (size_t i = 0; i < ymmlen(len); i++) {
			this->data[i].v256 = _mm256_xor_si256(this->data[i].v256, operand.data[i].v256);
		}
		return *this;
	}

	bool operator==(const avx2bitset<len>& operand) const noexcept {
		bool tmp = true;
		for (size_t i = 0; i < ymmlen(len); i++) {
			tmp &= _mm256_testc_si256(this->data[i].v256, operand.data[i].v256) & _mm256_testc_si256(operand.data[i].v256, this->data[i].v256);
		}
		return tmp;
	}

	bool operator!=(const avx2bitset<len>& operand) const noexcept {
		return not (*this == operand);
	}

	void dump(std::ostream& ost) const {
		for (int64_t i = len-1; i >= 0; i--) {
			ost << ((this->data[i / 256].v8[(i % 256) / 8] & (1 << (i % 8))) != 0);
		}
	}
};

template<size_t len>
inline std::ostream& operator<<(std::ostream& ost, const avx2bitset<len>& a2bs) {
	a2bs.dump(ost);
	return ost;
}

template<size_t len>
inline avx2bitset<len> operator&(const avx2bitset<len>& rhs, const avx2bitset<len>& lhs) {
	auto tmp = avx2bitset<len>(rhs);
	tmp &= lhs;
	return tmp;
}

template<size_t len>
inline avx2bitset<len> operator|(const avx2bitset<len>& rhs, const avx2bitset<len>& lhs) {
	auto tmp = avx2bitset<len>(rhs);
	tmp |= lhs;
	return tmp;
}

template<size_t len>
inline avx2bitset<len> operator^(const avx2bitset<len>& rhs, const avx2bitset<len>& lhs) {
	auto tmp = avx2bitset<len>(rhs);
	tmp ^= lhs;
	return tmp;
}

#endif // defined(__AVX2__)
