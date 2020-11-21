#pragma once

#include <cstdint>
#include <limits>
#include <bit>
#include <immintrin.h>

#include <cassert>

#include <iostream>

#include "arithmetics.hpp"



namespace waffle {

	template <usize S> struct mregf;
	template <usize S> struct mregi32;
	template <usize S> struct mregd;
	template <usize S> struct mregb;

	template <> struct mregf<4>;
	template <> struct mregf<8>;

	template <> struct mregi32<4>;
	template <> struct mregi32<8>;

	template <> struct mregd<2>;
	template <> struct mregd<4>;

	template <> struct mregb<2>;
	template <> struct mregb<4>;
	template <> struct mregb<8>;



	// make sure bitwise tricks in _mregb_ work

	static_assert(sizeof(bool) == 1);

	static_assert(std::bit_cast<uint8_t>(bool(1)) == uint8_t(0x01));


	template <typename U>
	struct mregb_base
	{

		/* data member */

		U raw;

		// no dynamic memory, no care of move semantics

		mregb_base() = default;
		mregb_base(const mregb_base &) = default;
		mregb_base & operator=(const mregb_base &) = default;


		/* ctor */

	private:

		mregb_base(U raw) : raw(raw) {}

		template <usize S>
		struct ones {
			constexpr static U value = (0x01 << (8 * (S - 1))) | ones<S - 1>::value;
		};

		template <>
		struct ones<0> {
			constexpr static U value = 0;
		};

	public:

		mregb_base(bool val) : raw(ones<sizeof(U)>::value * val) {}
		mregb_base(const bool *src) : raw(*reinterpret_cast<const U *>(src)) {}
		mregb_base(const bool *base_addr, const usize *indices) : raw(0) {
			for (usize i = 0; i < sizeof(U); ++i) raw |= uint32_t(0x01 * base_addr[indices[i]]) << (8 * i);
		}
		

		/* access */

		void storeu(bool dest[4]) { *reinterpret_cast<U *>(dest) = raw; }

		bool operator[](usize i) const {
			assert(i < 4);
			return *(reinterpret_cast<const bool *>(&raw) + i);
		}


		/* bitwise */

		mregb_base & operator&=(const mregb_base &rhs) { raw &= rhs.raw; return *this; }
		mregb_base & operator|=(const mregb_base &rhs) { raw |= rhs.raw; return *this; }
		mregb_base & operator^=(const mregb_base &rhs) { raw ^= rhs.raw; return *this; }
		mregb_base operator~() { return mregb_base(~raw); }

		friend mregb_base operator&(const mregb_base &lhs, const mregb_base &rhs) { return mregb_base(lhs) &= rhs; }
		friend mregb_base operator|(const mregb_base &lhs, const mregb_base &rhs) { return mregb_base(lhs) |= rhs; }
		friend mregb_base operator^(const mregb_base &lhs, const mregb_base &rhs) { return mregb_base(lhs) ^= rhs; }

	};


	template <>
	struct mregb<2> : public mregb_base<uint16_t>
	{
		using Base = mregb_base<uint16_t>;
		
		mregb() = default;
		mregb(const mregb &) = default;
		mregb & operator=(const mregb &) = default;

		mregb(const Base &base) : Base(base) {}
		mregb & operator=(const Base &base) { Base::operator=(base); }
	};


	template <>
	struct mregb<4> : public mregb_base<uint32_t>
	{
		using Base = mregb_base<uint32_t>;
		
		mregb() = default;
		mregb(const mregb &) = default;
		mregb & operator=(const mregb &) = default;

		mregb(const Base & base) : Base(base) {}
		mregb & operator=(const Base & base) { Base::operator=(base); }
	};


	template <>
	struct mregb<8> : public mregb_base<uint64_t>
	{
		using Base = mregb_base<uint64_t>;
		
		mregb() = default;
		mregb(const mregb &) = default;
		mregb & operator=(const mregb &) = default;

		mregb(const Base & base) : Base(base) {}
		mregb & operator=(const Base & base) { Base::operator=(base); }
	};



	template <>
	struct mregf<4>
	{

		/* data member */

		__m128 raw;


		/* type casting between raw register and wrapper */

		mregf(__m128 m128) : raw(m128) {}
		operator __m128() const { return raw; }
		operator __m128 &() { return raw; }


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xF;


		/* loading / writing / accessing */

		mregf() = default;
		mregf(const mregf &) = default;
		mregf & operator=(const mregf &) = default;

		static mregf zeros() { return mregf(_mm_setzero_ps()); }

		mregf(float f) : raw(_mm_set1_ps(f)) {}
		mregf(const float src[4]) : raw(_mm_loadu_ps(src)) {}

		mregf(const float *base_addr, const int inds[4]) :
			raw(_mm_i32gather_ps(base_addr, _mm_loadu_si128((const __m128i *)(inds)), 4)) {}

		void storeu(float dest[4]) { _mm_storeu_ps(dest, raw); }

		float operator[](int i) const {
			switch (i) {
				case 0:
					return _mm_cvtss_f32(raw);
				case 1:
					return _mm_cvtss_f32(_mm_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 1)));
				case 2:
					return _mm_cvtss_f32(_mm_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 2)));
				case 3:
					return _mm_cvtss_f32(_mm_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 3)));
				default:
					throw std::out_of_range("mregf<4> subscript");
					break;
			}
		}


		/* (explicit) type casting between register types */

		explicit operator mregi32<4>() const;
		explicit operator mregd<2>() const;

		explicit operator mregb<4>() const {
			__m128i rawi = _mm_castps_si128(raw);
			__m128i pack = _mm_packs_epi32(rawi, rawi);
			pack = _mm_packs_epi16(pack, pack);
			const int result = _mm_cvtsi128_si32(pack);
			return std::bit_cast<mregb<4>>(result);
		}


		/* arithmetic op */

		mregf & operator+=(const mregf &rhs) { raw = _mm_add_ps(raw, rhs.raw); return *this; }
		mregf & operator-=(const mregf &rhs) { raw = _mm_sub_ps(raw, rhs.raw); return *this; }
		mregf & operator*=(const mregf &rhs) { raw = _mm_mul_ps(raw, rhs.raw); return *this; }
		mregf & operator/=(const mregf &rhs) { raw = _mm_div_ps(raw, rhs.raw); return *this; }

		friend mregf operator+(const mregf &lhs, const mregf &rhs) { return mregf(lhs) += rhs; }
		friend mregf operator-(const mregf &lhs, const mregf &rhs) { return mregf(lhs) -= rhs; }
		friend mregf operator*(const mregf &lhs, const mregf &rhs) { return mregf(lhs) *= rhs; }
		friend mregf operator/(const mregf &lhs, const mregf &rhs) { return mregf(lhs) /= rhs; }

		friend mregf abs(const mregf &a);
		friend mregf sqrt(const mregf &a) { return _mm_sqrt_ps(a); }


		/* bitwise op */

		mregf & operator&=(const mregf &rhs) { raw = _mm_and_ps(raw, rhs.raw); return *this; }
		mregf & operator|=(const mregf &rhs) { raw = _mm_or_ps(raw, rhs.raw); return *this; }
		mregf & operator^=(const mregf &rhs) { raw = _mm_xor_ps(raw, rhs.raw); return *this; }

		friend mregf operator&(const mregf &lhs, const mregf &rhs) { return mregf(lhs) &= rhs; }
		friend mregf operator|(const mregf &lhs, const mregf &rhs) { return mregf(lhs) |= rhs; }
		friend mregf operator^(const mregf &lhs, const mregf &rhs) { return mregf(lhs) ^= rhs; }

		mregf operator~() const {
			__m128i rawi = _mm_castps_si128(raw);
			return (*this) ^ _mm_castsi128_ps(_mm_cmpeq_epi32(rawi, rawi));
		}

		mregf and_not(const mregf &a) const { return _mm_andnot_ps(a, raw); }


		/* logical */

		friend mregb<4> operator==(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_EQ_OS))); }
		friend mregb<4> operator!=(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_NEQ_OS))); }
		friend mregb<4> operator<=(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_LE_OS))); }
		friend mregb<4> operator>=(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_GE_OS))); }
		friend mregb<4> operator<(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_LT_OS))); }
		friend mregb<4> operator>(const mregf & a, const mregf & b) { return mregb<4>(mregf(_mm_cmp_ps(a, b, _CMP_GT_OS))); }


		/* reducing to Boolean value */

		friend bool all(const mregf &a) { return _mm_movemask_ps(a.raw) == all_mask; }
		friend bool none(const mregf &a) { return _mm_testz_ps(a.raw, a.raw); }
		friend bool any(const mregf &a) { return !none(a); }

		friend bool all(const mregf &valid_mask, const mregf &b) { return all((~valid_mask) | b); }
		friend bool none(const mregf &valid_mask, const mregf &b) { return none(valid_mask & b); }
		friend bool any(const mregf &valid_mask, const mregf &b) { return any(valid_mask & b); };


		/* printing */

		friend std::ostream & operator<<(std::ostream &os, const mregf &reg) {
			return os << reg[0] << reg[1] << reg[2] << reg[3];
		}

	};



	template <>
	struct mregf<8>
	{

		/* data member  */

		__m256 raw;


		/* type casting between raw register and wrapper */

		mregf(__m256 m256) : raw(m256) {}
		operator __m256() const { return raw; }
		operator __m256 &() { return raw; }


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xFF;


		/* loading / storing / accessing */

		mregf() = default;
		mregf(const mregf &) = default;
		mregf & operator=(const mregf &) = default;

		static mregf zeros() { return mregf(_mm256_setzero_ps()); }

		mregf(float f) : raw(_mm256_set1_ps(f)) {}
		mregf(const float src[4]) : raw(_mm256_loadu_ps(src)) {}

		mregf(const float *base_addr, const int inds[8]) :
			raw(_mm256_i32gather_ps(base_addr, _mm256_loadu_si256((const __m256i *)(inds)), 4)) {}

		void storeu(float dest[4]) { _mm256_storeu_ps(dest, raw); }

		float operator[](int i) const {
			auto mm256_permute2x128_ps = [] (__m256 m) -> __m256 {
				__m256i mi = _mm256_castps_si256(m);
				__m256i perm_mi = _mm256_permute2x128_si256(mi, mi, 0b1001'0001);
				return _mm256_castsi256_ps(perm_mi);
			};
			
			switch (i) {
				case 0:
					return _mm256_cvtss_f32(raw);
				case 1:
					return _mm256_cvtss_f32(_mm256_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 1)));	
				case 2:
					return _mm256_cvtss_f32(_mm256_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 2)));	
				case 3:
					return _mm256_cvtss_f32(_mm256_permute_ps(raw, _MM_SHUFFLE(0, 0, 0, 3)));
				case 4:
					return _mm256_cvtss_f32(mm256_permute2x128_ps(raw));
				case 5:
					return _mm256_cvtss_f32(_mm256_permute_ps(
						mm256_permute2x128_ps(raw),
						_MM_SHUFFLE(0, 0, 0, 1)
					));
				case 6:
					return _mm256_cvtss_f32(_mm256_permute_ps(
						mm256_permute2x128_ps(raw),
						_MM_SHUFFLE(0, 0, 0, 2)
					));
				case 7:
					return _mm256_cvtss_f32(_mm256_permute_ps(
						mm256_permute2x128_ps(raw),
						_MM_SHUFFLE(0, 0, 0, 3)
					));
				default:
					throw std::out_of_range("mregf<8> subscript");
					break;
			}
		}


		/* (explicit) type casting between register types */

		explicit operator mregi32<8>() const;
		explicit operator mregd<4>() const;

		explicit operator mregb<8>() const {
			__m256i rawi = _mm256_castps_si256(raw);
			__m256i pack = _mm256_packs_epi32(rawi, rawi);
			pack = _mm256_packs_epi16(pack, pack);
			const int64_t result =
				int64_t(_mm256_cvtsi256_si32(pack))
				+ (int64_t(_mm256_cvtsi256_si32(_mm256_shuffle_epi32(pack, _MM_SHUFFLE(0, 0, 0, 1)))) << 32);
			//return mregb<8>(reinterpret_cast<const bool *>(&result));
			return std::bit_cast<mregb<8>>(result);
		}


		/* arithmetic op */

		mregf & operator+=(const mregf & rhs) { raw = _mm256_add_ps(raw, rhs.raw); return *this; }
		mregf & operator-=(const mregf & rhs) { raw = _mm256_sub_ps(raw, rhs.raw); return *this; }
		mregf & operator*=(const mregf & rhs) { raw = _mm256_mul_ps(raw, rhs.raw); return *this; }
		mregf & operator/=(const mregf & rhs) { raw = _mm256_div_ps(raw, rhs.raw); return *this; }

		friend mregf operator+(const mregf & lhs, const mregf & rhs) { return mregf(lhs) += rhs; }
		friend mregf operator-(const mregf & lhs, const mregf & rhs) { return mregf(lhs) -= rhs; }
		friend mregf operator*(const mregf & lhs, const mregf & rhs) { return mregf(lhs) *= rhs; }
		friend mregf operator/(const mregf & lhs, const mregf & rhs) { return mregf(lhs) /= rhs; }

		friend mregf abs(const mregf & a);
		friend mregf sqrt(const mregf & a) { return _mm256_sqrt_ps(a); }


		/* bitwise op */

		mregf & operator&=(const mregf & rhs) { raw = _mm256_and_ps(raw, rhs.raw); return *this; }
		mregf & operator|=(const mregf & rhs) { raw = _mm256_or_ps(raw, rhs.raw); return *this; }
		mregf & operator^=(const mregf & rhs) { raw = _mm256_xor_ps(raw, rhs.raw); return *this; }

		friend mregf operator&(const mregf & lhs, const mregf & rhs) { return mregf(lhs) &= rhs; }
		friend mregf operator|(const mregf & lhs, const mregf & rhs) { return mregf(lhs) |= rhs; }
		friend mregf operator^(const mregf & lhs, const mregf & rhs) { return mregf(lhs) ^= rhs; }

		mregf operator~() const {
			__m256i rawi = _mm256_castps_si256(raw);
			return (*this) ^ _mm256_castsi256_ps(_mm256_cmpeq_epi32(rawi, rawi));
		}

		mregf and_not(const mregf & a) const { return _mm256_andnot_ps(a, raw); }


		/* logical */

		friend mregb<8> operator==(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_EQ_OS))); }
		friend mregb<8> operator!=(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_NEQ_OS))); }
		friend mregb<8> operator<=(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_LE_OS))); }
		friend mregb<8> operator>=(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_GE_OS))); }
		friend mregb<8> operator<(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_LT_OS))); }
		friend mregb<8> operator>(const mregf & a, const mregf & b) { return mregb<8>(mregf(_mm256_cmp_ps(a, b, _CMP_GT_OS))); }


		/* reducing to Boolean value */

		friend bool all(const mregf & a) { return _mm256_movemask_ps(a.raw) == all_mask; }
		friend bool none(const mregf & a) { return _mm256_testz_ps(a.raw, a.raw); }
		friend bool any(const mregf & a) { return !none(a); }

		friend bool all(const mregf & valid_mask, const mregf & b) { return all((~valid_mask) | b); }
		friend bool none(const mregf & valid_mask, const mregf & b) { return none(valid_mask & b); }
		friend bool any(const mregf & valid_mask, const mregf & b) { return any(valid_mask & b); };


		/* printing */

		friend std::ostream & operator<<(std::ostream &os, const mregf &reg) {
			return os << reg[0] << reg[1] << reg[2] << reg[3] << reg[4] << reg[5] << reg[6] << reg[7];
		}

	};



	template <>
	struct mregi32<4>
	{

		/* data member */

		__m128i raw;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xF;


		/* type casting between raw register and wrapper */

		mregi32(__m128i m128i) : raw(m128i) {}

		operator __m128i() const { return raw; }
		operator __m128i &() { return raw; }


		/* loading / storing / accessing */

		mregi32() = default;
		mregi32(const mregi32 &) = default;
		mregi32 & operator=(const mregi32 &) = default;

		static mregi32 zeros() { return mregi32(_mm_setzero_si128()); }

		mregi32(int i) : raw(_mm_set1_epi32(i)) {}
		mregi32(const int src[4]) : raw(_mm_loadu_si128((__m128i *)src)) {}

		mregi32(const int *base_addr, const int inds[4]) :
			raw(_mm_i32gather_epi32(base_addr, _mm_loadu_si128((const __m128i *)(inds)), 4)) {}

		void storeu(int dest[4]) { _mm_storeu_si128((__m128i *)dest, raw); }

		int operator[](int i) const {
			switch (i) {
				case 0:
					return _mm_cvtsi128_si32(raw);
				case 1:
					return _mm_cvtsi128_si32(_mm_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 1)));
				case 2:
					return _mm_cvtsi128_si32(_mm_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 2)));
				case 3:
					return _mm_cvtsi128_si32(_mm_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 3)));
				default:
					throw std::out_of_range("mregi32<4> subscript");
					break;
			}
		}


		/* (explicit) type casting between register types */

		explicit operator mregf<4>() const;
		explicit operator mregd<2>() const;

		explicit operator mregb<4>() const {
			__m128i pack = _mm_packs_epi32(raw, raw);
			pack = _mm_packs_epi16(pack, pack);
			const int result = _mm_cvtsi128_si32(pack);
			return std::bit_cast<mregb<4>>(result);
		}


		/* arithmetic op */

		mregi32 & operator+=(const mregi32 &rhs) { raw = _mm_add_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator-=(const mregi32 &rhs) { raw = _mm_sub_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator*=(const mregi32 &rhs) { raw = _mm_mul_epi32(raw, rhs.raw); return *this; }
		// TODO: signed int division with approximation
		// https://github.com/vectorclass/version2/blob/master/vectori128.h
		// mregi32 & operator/=(const mregi32 &rhs) { raw = _mm_div_epi32(raw, rhs.raw); return *this; }

		friend mregi32 operator+(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) += rhs; }
		friend mregi32 operator-(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) -= rhs; }
		friend mregi32 operator*(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) *= rhs; }
		// friend mregi32 operator/(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) /= rhs; }

		friend mregi32 abs(const mregi32 &a) { return _mm_abs_epi32(a); };


		/* bitwise op */

		mregi32 & operator&=(const mregi32 &rhs) { raw = _mm_and_si128(raw, rhs.raw); return *this; }
		mregi32 & operator|=(const mregi32 &rhs) { raw = _mm_or_si128(raw, rhs.raw); return *this; }
		mregi32 & operator^=(const mregi32 &rhs) { raw = _mm_xor_si128(raw, rhs.raw); return *this; }

		friend mregi32 operator&(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) &= rhs; }
		friend mregi32 operator|(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) |= rhs; }
		friend mregi32 operator^(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) ^= rhs; }

		mregi32 operator~() const { return (*this) ^ _mm_cmpeq_epi32(raw, raw); }

		mregi32 and_not(const mregi32 &a) const { return _mm_andnot_si128(a, raw); }


		/* comparator */

		friend mregb<4> operator==(const mregi32 & a, const mregi32 & b) { return mregb<4>(mregi32(_mm_cmpeq_epi32(a, b))); }
		friend mregb<4> operator!=(const mregi32 & a, const mregi32 & b) { return ~(a == b); }
		friend mregb<4> operator<=(const mregi32 & a, const mregi32 & b) { return ~(a > b); }
		friend mregb<4> operator>=(const mregi32 & a, const mregi32 & b) { return (a == b) | (a > b); }
		friend mregb<4> operator<(const mregi32 & a, const mregi32 & b) { return ~(a >= b); }
		friend mregb<4> operator>(const mregi32 & a, const mregi32 & b) { return mregb<4>(mregi32(_mm_cmpgt_epi32(a, b))); }


		/* reducing to Boolean value */

		friend bool all(const mregi32 &a) { return _mm_movemask_ps(static_cast<mregf<4>>(a)) == all_mask; }
		friend bool none(const mregi32 &a) { return _mm_testz_ps(static_cast<mregf<4>>(a), static_cast<mregf<4>>(a)); }
		friend bool any(const mregi32 &a) { return !none(a); }

		friend bool all(const mregi32 &valid_mask, const mregi32 &b) { return all((~valid_mask) | b); }
		friend bool none(const mregi32 &valid_mask, const mregi32 &b) { return none(valid_mask & b); }
		friend bool any(const mregi32 &valid_mask, const mregi32 &b) { return any(valid_mask & b); };


		/* printing */

		friend std::ostream & operator<<(std::ostream &os, const mregi32 &reg) {
			return os << reg[0] << reg[1] << reg[2] << reg[3];
		}

	};



	template <>
	struct mregi32<8>
	{

		/* member and ctor */

		__m256i raw;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xFF;


		/* type casting between raw register and wrapper */

		mregi32(__m256i m256i) : raw(m256i) {}

		operator __m256i() const { return raw; }
		operator __m256i &() { return raw; }


		/* loading / storing / accessing */

		mregi32() = default;
		mregi32(const mregi32 &) = default;
		mregi32 & operator=(const mregi32 &) = default;

		static mregi32 zeros() { return _mm256_setzero_si256(); }

		mregi32(int i) : raw(_mm256_set1_epi32(i)) {}
		mregi32(const int src[8]) : raw(_mm256_loadu_si256((__m256i *)src)) {}

		mregi32(const int *base_addr, const int inds[8]) :
			raw(_mm256_i32gather_epi32(base_addr, _mm256_loadu_si256((const __m256i *)(inds)), 4)) {}

		void storeu(int dest[8]) { _mm256_storeu_si256((__m256i *)dest, raw); }

		int operator[](int i) const {
			auto mm256_cvtsi256_si32 = [] (__m256i m) -> int {
				return _mm_cvtsi128_si32(_mm256_castsi256_si128(m));
			};

			switch (i) {
				case 0:
					return mm256_cvtsi256_si32(raw);
				case 1:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 1)));
				case 2:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 2)));
				case 3:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(raw, _MM_SHUFFLE(0, 0, 0, 3)));
				case 4:
					return mm256_cvtsi256_si32(_mm256_permute2x128_si256(raw, raw, 0b1001'0001));
				case 5:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(
						_mm256_permute2x128_si256(raw, raw, 0b1001'0001),
						_MM_SHUFFLE(0, 0, 0, 1)
					));
				case 6:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(
						_mm256_permute2x128_si256(raw, raw, 0b1001'0001),
						_MM_SHUFFLE(0, 0, 0, 2)
					));
				case 7:
					return mm256_cvtsi256_si32(_mm256_shuffle_epi32(
						_mm256_permute2x128_si256(raw, raw, 0b1001'0001),
						_MM_SHUFFLE(0, 0, 0, 3)
					));
				default:
					throw std::out_of_range("mregi32<8> subscript");
					break;
			}
		}


		/* (explicit) type casting between register types */

		explicit operator mregf<8>() const;
		explicit operator mregd<4>() const;

		explicit operator mregb<8>() const {
			__m256i pack = _mm256_packs_epi32(raw, raw);
			pack = _mm256_packs_epi16(pack, pack);
			const int64_t result =
				int64_t(_mm256_cvtsi256_si32(pack))
				+ (int64_t(_mm256_cvtsi256_si32(_mm256_shuffle_epi32(pack, _MM_SHUFFLE(0, 0, 0, 1)))) << 32);
			return std::bit_cast<mregb<8>>(result);
		}


		/* arithmetic op */

		mregi32 & operator+=(const mregi32 & rhs) { raw = _mm256_add_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator-=(const mregi32 & rhs) { raw = _mm256_sub_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator*=(const mregi32 & rhs) { raw = _mm256_mul_epi32(raw, rhs.raw); return *this; }
		// TODO: signed int division with approximation
		// https://github.com/vectorclass/version2/blob/master/vectori128.h
		// mregi32 & operator/=(const mregi32 & rhs) { raw = _mm256_div_epi32(raw, rhs.raw); return *this; }

		friend mregi32 operator+(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) += rhs; }
		friend mregi32 operator-(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) -= rhs; }
		friend mregi32 operator*(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) *= rhs; }
		// friend mregi32 operator/(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) /= rhs; }

		friend mregi32 abs(const mregi32 & a) { return _mm256_abs_epi32(a); };


		/* bitwise */

		mregi32 & operator&=(const mregi32 & rhs) { raw = _mm256_and_si256(raw, rhs.raw); return *this; }
		mregi32 & operator|=(const mregi32 & rhs) { raw = _mm256_or_si256(raw, rhs.raw); return *this; }
		mregi32 & operator^=(const mregi32 & rhs) { raw = _mm256_xor_si256(raw, rhs.raw); return *this; }

		friend mregi32 operator&(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) &= rhs; }
		friend mregi32 operator|(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) |= rhs; }
		friend mregi32 operator^(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) ^= rhs; }

		mregi32 operator~() const { return (*this) ^ _mm256_cmpeq_epi32(raw, raw); }

		mregi32 and_not(const mregi32 & a) const { return _mm256_andnot_si256(a, raw); }


		/* logical */

		friend mregb<8> operator==(const mregi32 & a, const mregi32 & b) { return mregb<8>(mregi32(_mm256_cmpeq_epi32(a, b))); }
		friend mregb<8> operator!=(const mregi32 & a, const mregi32 & b) { return ~(a == b); }
		friend mregb<8> operator<=(const mregi32 & a, const mregi32 & b) { return ~(a > b); }
		friend mregb<8> operator>=(const mregi32 & a, const mregi32 & b) { return (a == b) | (a > b); }
		friend mregb<8> operator<(const mregi32 & a, const mregi32 & b) { return ~(a >= b); }
		friend mregb<8> operator>(const mregi32 & a, const mregi32 & b) { return mregb<8>(mregi32(_mm256_cmpgt_epi32(a, b))); }


		/* reducing to Boolean value */

		friend bool all(const mregi32 & a) { return _mm256_movemask_ps(static_cast<mregf<8>>(a)) == all_mask; }
		friend bool none(const mregi32 & a) { return _mm256_testz_ps(static_cast<mregf<8>>(a), static_cast<mregf<8>>(a)); }
		friend bool any(const mregi32 & a) { return !none(a); }

		friend bool all(const mregi32 & valid_mask, const mregi32 & b) { return all((~valid_mask) | b); }
		friend bool none(const mregi32 & valid_mask, const mregi32 & b) { return none(valid_mask & b); }
		friend bool any(const mregi32 & valid_mask, const mregi32 & b) { return any(valid_mask & b); };


		/* printing */

		friend std::ostream & operator<<(std::ostream &os, const mregi32 &reg) {
			return os << reg[0] << reg[1] << reg[2] << reg[3] << reg[4] << reg[5] << reg[6] << reg[7];
		}

	};



	template <>
	struct mregd<2>
	{
		__m128d raw;

		mregd() : raw(_mm_setzero_pd()) {}
		mregd(const mregd &) = default;
		mregd & operator=(const mregd &) = default;

		mregd(__m128d m128d) : raw(m128d) {}
		mregd(double d) : raw(_mm_set1_pd(d)) {}

		explicit operator mregf<4>() const;
		explicit operator mregi32<4>() const;
	};



	template <>
	struct mregd<4>
	{
		__m256d raw;

		mregd() : raw(_mm256_setzero_pd()) {}
		mregd(const mregd &) = default;
		mregd & operator=(const mregd &) = default;

		mregd(__m256d m256d) : raw(m256d) {}
		mregd(double d) : raw(_mm256_set1_pd(d)) {}

		explicit operator mregf<8>() const;
		explicit operator mregi32<8>() const;
	};



	/* trait check */

	static_assert(std::is_trivial_v<mregb<2>> && std::is_standard_layout_v<mregb<2>>);
	static_assert(std::is_trivial_v<mregb<2>> && std::is_standard_layout_v<mregb<4>>);
	static_assert(std::is_trivial_v<mregb<2>> && std::is_standard_layout_v<mregb<8>>);

	static_assert(std::is_trivial_v<mregf<4>> && std::is_standard_layout_v<mregf<4>>);
	static_assert(std::is_trivial_v<mregf<8>> && std::is_standard_layout_v<mregf<8>>);

	static_assert(std::is_trivial_v<mregi32<4>> && std::is_standard_layout_v<mregi32<4>>);
	static_assert(std::is_trivial_v<mregi32<8>> && std::is_standard_layout_v<mregi32<8>>);



	/* type casting implementation */

	// 128 bits

	mregf<4>::operator mregi32<4>() const { return _mm_castps_si128(raw); }
	mregf<4>::operator mregd<2>() const { return _mm_castps_pd(raw); }

	mregi32<4>::operator mregf<4>() const { return _mm_castsi128_ps(raw); }
	mregi32<4>::operator mregd<2>() const { return _mm_castsi128_pd(raw); }

	mregd<2>::operator mregf<4>() const { return _mm_castpd_ps(raw); }
	mregd<2>::operator mregi32<4>() const { return _mm_castpd_si128(raw); }

	// 256 bits

	mregf<8>::operator mregi32<8>() const { return _mm256_castps_si256(raw); }
	mregf<8>::operator mregd<4>() const { return _mm256_castps_pd(raw); }

	mregi32<8>::operator mregf<8>() const { return _mm256_castsi256_ps(raw); }
	mregi32<8>::operator mregd<4>() const { return _mm256_castsi256_pd(raw); }

	mregd<4>::operator mregf<8>() const { return _mm256_castpd_ps(raw); }
	mregd<4>::operator mregi32<8>() const { return _mm256_castpd_si256(raw); }



	template <int W>
	mregf<W> abs(const mregf<W> &a) { return a & static_cast<mregf<W>>(mregi32<W>(~mregf<W>::sign_mask)); };



	template <typename T>
	concept simdable = std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, bool>;

	template <simdable T>
	struct avx;

	template <>
	struct avx<float> {
		static constexpr isize stride = 8;
		using reg = mregf<8>;
	};

	template <>
	struct avx<int> {
		static constexpr isize stride = 8;
		using reg = mregi32<8>;
	};

	template <>
	struct avx<bool> {
		constexpr static isize stride = 8;
		using reg = mregb<8>;
	};

	template <simdable T>
	struct sse;

	template <>
	struct sse<float> {
		static constexpr isize stride = 4;
		using reg = mregf<4>;
	};

	template <>
	struct sse<int> {
		static constexpr isize stride = 4;
		using reg = mregi32<4>;
	};

	template <>
	struct sse<bool> {
		constexpr static isize stride = 4;
		using reg = mregb<4>;
	};

	template <simdable T>
	using avx_reg = typename avx<T>::reg;

	template <simdable T>
	using sse_reg = typename sse<T>::reg;

	template <simdable T>
	inline constexpr isize avx_stride = avx<T>::stride;

	template <simdable T>
	inline constexpr isize sse_stride = sse<T>::stride;

}