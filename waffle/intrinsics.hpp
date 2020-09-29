#pragma once

#include <immintrin.h>
#include <cassert>
#include <iostream>

#include "integer.hpp"



namespace waffle {

	template <usize S> struct mregf {};
	template <usize S> struct mregi32 {};
	template <usize S> struct mregd {};

	template <> struct mregf<4>;
	template <> struct mregf<8>;

	template <> struct mregi32<4>;
	template <> struct mregi32<8>;

	template <> struct mregd<2>;
	template <> struct mregd<4>;



	template <>
	struct mregf<4>
	{

		/* member and ctor */

		__m128 raw;

		mregf() : raw(_mm_setzero_ps()) {}
		mregf(const mregf &) = default;
		mregf & operator=(const mregf &) = default;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xF;


		/* type casting between raw register and wrapper */

		mregf(__m128 m128) : raw(m128) {}
		operator __m128() const { return raw; }
		operator __m128 &() { return raw; }


		/* loading / writing / accessing */

		mregf(float f) : raw(_mm_set1_ps(f)) {}
		mregf(const float src[4]) : raw(_mm_loadu_ps(src)) {}

		void storeu(float dest[4]) { _mm_storeu_ps(dest, raw); }

		float operator[](int i) const { assert(i < 4); return raw.m128_f32[i]; }


		/* (explicit) type casting between register types */

		explicit operator mregi32<4>() const;
		explicit operator mregd<2>() const;


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

		mregf operator~() const { return (*this) ^ (*this == *this); }

		mregf and_not(const mregf &a) const { return _mm_andnot_ps(a, raw); }


		/* comparator */

		friend mregf operator==(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_EQ_OS); }
		friend mregf operator!=(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_NEQ_OS); }
		friend mregf operator<=(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_LE_OS); }
		friend mregf operator>=(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_GE_OS); }
		friend mregf operator<(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_LT_OS); }
		friend mregf operator>(const mregf & a, const mregf & b) { return _mm_cmp_ps(a, b, _CMP_GT_OS); }


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

		/* member and ctor */

		__m256 raw;

		mregf() : raw(_mm256_setzero_ps()) {}
		mregf(const mregf &) = default;
		mregf & operator=(const mregf &) = default;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xFF;


		/* type casting between raw register and wrapper */

		mregf(__m256 m256) : raw(m256) {}
		operator __m256() const { return raw; }
		operator __m256 &() { return raw; }


		/* loading / storing / accessing */

		mregf(float f) : raw(_mm256_set1_ps(f)) {}
		mregf(const float src[4]) : raw(_mm256_loadu_ps(src)) {}

		void storeu(float dest[4]) { _mm256_storeu_ps(dest, raw); }

		float operator[](int i) const { assert(i < 8); return raw.m256_f32[i]; }


		/* (explicit) type casting between register types */

		explicit operator mregi32<8>() const;
		explicit operator mregd<4>() const;


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

		mregf operator~() const { return (*this) ^ (*this == *this); }

		mregf and_not(const mregf & a) const { return _mm256_andnot_ps(a, raw); }


		/* comparator */

		friend mregf operator==(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OS); }
		friend mregf operator!=(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OS); }
		friend mregf operator<=(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
		friend mregf operator>=(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS); }
		friend mregf operator<(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
		friend mregf operator>(const mregf & a, const mregf & b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS); }


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

		/* member and ctor */

		__m128i raw;

		mregi32() : raw(_mm_setzero_si128()) {}
		mregi32(const mregi32 &) = default;
		mregi32 & operator=(const mregi32 &) = default;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xF;


		/* type casting between raw register and wrapper */

		mregi32(__m128i m128i) : raw(m128i) {}

		operator __m128i() const { return raw; }
		operator __m128i &() { return raw; }


		/* loading / storing / accessing */

		mregi32(int i) : raw(_mm_set1_epi32(i)) {}
		mregi32(const int src[4]) : raw(_mm_loadu_si128((__m128i *)src)) {}

		void storeu(int dest[4]) { _mm_storeu_si128((__m128i *)dest, raw); }

		int operator[](int i) const { assert(i < 4); return raw.m128i_i32[i]; }


		/* (explicit) type casting between register types */

		explicit operator mregf<4>() const;
		explicit operator mregd<2>() const;


		/* arithmetic op */

		mregi32 & operator+=(const mregi32 &rhs) { raw = _mm_add_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator-=(const mregi32 &rhs) { raw = _mm_sub_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator*=(const mregi32 &rhs) { raw = _mm_mul_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator/=(const mregi32 &rhs) { raw = _mm_div_epi32(raw, rhs.raw); return *this; }

		friend mregi32 operator+(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) += rhs; }
		friend mregi32 operator-(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) -= rhs; }
		friend mregi32 operator*(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) *= rhs; }
		friend mregi32 operator/(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) /= rhs; }

		friend mregi32 abs(const mregi32 &a) { return _mm_abs_epi32(a); };


		/* bitwise op */

		mregi32 & operator&=(const mregi32 &rhs) { raw = _mm_and_si128(raw, rhs.raw); return *this; }
		mregi32 & operator|=(const mregi32 &rhs) { raw = _mm_or_si128(raw, rhs.raw); return *this; }
		mregi32 & operator^=(const mregi32 &rhs) { raw = _mm_xor_si128(raw, rhs.raw); return *this; }

		friend mregi32 operator&(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) &= rhs; }
		friend mregi32 operator|(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) |= rhs; }
		friend mregi32 operator^(const mregi32 &lhs, const mregi32 &rhs) { return mregi32(lhs) ^= rhs; }

		mregi32 operator~() const { return (*this) ^ (*this == *this); }

		mregi32 and_not(const mregi32 &a) const { return _mm_andnot_si128(a, raw); }


		/* comparator */

		friend mregi32 operator==(const mregi32 & a, const mregi32 & b) { return _mm_cmpeq_epi32(a, b); }
		friend mregi32 operator!=(const mregi32 & a, const mregi32 & b) { return ~(a == b); }
		friend mregi32 operator<=(const mregi32 & a, const mregi32 & b) { return ~(a > b); }
		friend mregi32 operator>=(const mregi32 & a, const mregi32 & b) { return (a == b) | (a > b); }
		friend mregi32 operator<(const mregi32 & a, const mregi32 & b) { return ~(a >= b); }
		friend mregi32 operator>(const mregi32 & a, const mregi32 & b) { return _mm_cmpgt_epi32(a, b); }


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

		mregi32() : raw(_mm256_setzero_si256()) {}
		mregi32(const mregi32 &) = default;
		mregi32 & operator=(const mregi32 &) = default;


		/* constexpr flag */

		static constexpr int sign_mask = 0x80000000;
		static constexpr int all_mask = 0xFF;


		/* type casting between raw register and wrapper */

		mregi32(__m256i m256i) : raw(m256i) {}

		operator __m256i() const { return raw; }
		operator __m256i &() { return raw; }


		/* loading / storing / accessing */

		mregi32(int i) : raw(_mm256_set1_epi32(i)) {}
		mregi32(const int src[8]) : raw(_mm256_loadu_si256((__m256i *)src)) {}

		void storeu(int dest[8]) { _mm256_storeu_si256((__m256i *)dest, raw); }

		int operator[](int i) const { assert(i < 8); return raw.m256i_i32[i]; }


		/* (explicit) type casting between register types */

		explicit operator mregf<8>() const;
		explicit operator mregd<4>() const;


		/* arithmetic op */

		mregi32 & operator+=(const mregi32 & rhs) { raw = _mm256_add_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator-=(const mregi32 & rhs) { raw = _mm256_sub_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator*=(const mregi32 & rhs) { raw = _mm256_mul_epi32(raw, rhs.raw); return *this; }
		mregi32 & operator/=(const mregi32 & rhs) { raw = _mm256_div_epi32(raw, rhs.raw); return *this; }

		friend mregi32 operator+(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) += rhs; }
		friend mregi32 operator-(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) -= rhs; }
		friend mregi32 operator*(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) *= rhs; }
		friend mregi32 operator/(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) /= rhs; }

		friend mregi32 abs(const mregi32 & a) { return _mm256_abs_epi32(a); };


		/* bitwise op */

		mregi32 & operator&=(const mregi32 & rhs) { raw = _mm256_and_si256(raw, rhs.raw); return *this; }
		mregi32 & operator|=(const mregi32 & rhs) { raw = _mm256_or_si256(raw, rhs.raw); return *this; }
		mregi32 & operator^=(const mregi32 & rhs) { raw = _mm256_xor_si256(raw, rhs.raw); return *this; }

		friend mregi32 operator&(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) &= rhs; }
		friend mregi32 operator|(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) |= rhs; }
		friend mregi32 operator^(const mregi32 & lhs, const mregi32 & rhs) { return mregi32(lhs) ^= rhs; }

		mregi32 operator~() const { return (*this) ^ (*this == *this); }

		mregi32 and_not(const mregi32 & a) const { return _mm256_andnot_si256(a, raw); }


		/* comparator */

		friend mregi32 operator==(const mregi32 & a, const mregi32 & b) { return _mm256_cmpeq_epi32(a, b); }
		friend mregi32 operator!=(const mregi32 & a, const mregi32 & b) { return ~(a == b); }
		friend mregi32 operator<=(const mregi32 & a, const mregi32 & b) { return ~(a > b); }
		friend mregi32 operator>=(const mregi32 & a, const mregi32 & b) { return (a == b) | (a > b); }
		friend mregi32 operator<(const mregi32 & a, const mregi32 & b) { return ~(a >= b); }
		friend mregi32 operator>(const mregi32 & a, const mregi32 & b) { return _mm256_cmpgt_epi32(a, b); }


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
	concept simdable = std::is_same_v<T, float> || std::is_same_v<T, int>;

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

	template <simdable T>
	using avx_reg = typename avx<T>::reg;

	template <simdable T>
	using sse_reg = typename sse<T>::reg;

	template <simdable T>
	inline constexpr isize avx_stride = avx<T>::stride;

	template <simdable T>
	inline constexpr isize sse_stride = sse<T>::stride;

}