#pragma once

#include <type_traits>
#include <concepts>

#include <cassert>

#include <iostream>

#include "arithmetics.hpp"
#include "intrinsics.hpp"



namespace waffle
{

	/* declaration */



	// array

	template <typename T, isize S>
	struct array;



	// ndarray

	namespace detail {
		template <typename T, isize ... Dims>
		struct ndarray_helper;

		template <typename T, isize InnerDim, isize ... Dims>
		struct ndarray_helper<T, InnerDim, Dims...> {
			using type = typename ndarray_helper<array<T, InnerDim>, Dims...>::type;
		};

		template <typename T, isize OutestDim>
		struct ndarray_helper<T, OutestDim> {
			using type = array<T, OutestDim>;
		};
	}

	template <typename T, isize ... Dims>
	requires (sizeof...(Dims) > 0)
	using ndarray = typename detail::ndarray_helper<T, Dims...>::type;



	// type traits of _array_

	template <typename InstType>
	struct is_array : std::false_type {};

	template <typename T, isize S>
	struct is_array<array<T, S>> : std::true_type {};

	template <typename InstType>
	inline constexpr bool is_array_v = is_array<InstType>::value;

	template <typename InstType>
	concept arrayname = is_array_v<InstType>;



	template <arrayname T>
	struct unwrap_array;

	namespace detail {
		template <typename T>
		struct unwrap_helper {
			using primitive = T;
			static constexpr isize depth = 0;
		};

		template <typename T, isize S>
		struct unwrap_helper<array<T, S>> {
			using primitive = typename unwrap_helper<T>::primitive;
			static constexpr isize depth = 1 + unwrap_helper<T>::depth;
		};
	}

	template <typename T, isize S>
	struct unwrap_array<array<T, S>> {
		using element = T;
		static constexpr isize size = S;
		using primitive = typename detail::unwrap_helper<array<T, S>>::primitive;
		static constexpr isize depth = detail::unwrap_helper<array<T, S>>::depth;
	};

	template <arrayname T>
	using array_element_t = typename unwrap_array<T>::element;

	template <arrayname T>
	inline constexpr isize array_size_v = unwrap_array<T>::size;

	template <arrayname T>
	using array_primitive_t = typename unwrap_array<T>::primitive;

	template <arrayname T>
	inline constexpr isize array_depth_v = unwrap_array<T>::depth;



	template <typename From, typename To>
	struct is_broadcastable_from : std::false_type {};

	template <typename Prim>
	requires (!arrayname<Prim>)
	struct is_broadcastable_from<Prim, Prim> : std::true_type {};
	
	template <typename Prim, typename Elem, isize Size>
	requires (std::is_same_v<Prim, array_primitive_t<array<Elem, Size>>>)
	struct is_broadcastable_from<Prim, array<Elem, Size>> : std::true_type {};

	template <typename FromType, isize FromSize, typename ToType, isize ToSize>
	struct is_broadcastable_from<array<FromType, FromSize>, array<ToType, ToSize>> {
		static constexpr bool value =
			is_broadcastable_from<array<FromType, FromSize>, ToType>::value
			|| (FromSize == ToSize && is_broadcastable_from<FromType, ToType>::value);
	};

	template <typename From, typename To>
	inline constexpr bool is_broadcastable_from_v = is_broadcastable_from<From, To>::value;

	template <typename From, typename To>
	concept broadcastable_to = is_broadcastable_from_v<From, To>;



	template <typename From, typename To>
	struct is_directly_broadcastable_from : std::false_type {};

	template <typename Inst>
	struct is_directly_broadcastable_from<Inst, Inst> : std::true_type {};

	template <typename From, typename ToElem, isize ToSize>
	requires (!std::is_same_v<From, array<ToElem, ToSize>>)
	struct is_directly_broadcastable_from<From, array<ToElem, ToSize>> {
		static constexpr bool value = is_directly_broadcastable_from<From, ToElem>::value;
	};

	template <typename From, typename To>
	inline constexpr bool is_directly_broadcastable_from_v = is_directly_broadcastable_from<From, To>::value;

	template <typename From, typename To>
	concept directly_broadcastable_to = is_directly_broadcastable_from_v<From, To>;



	template <typename From, typename To>
	struct is_tightly_broadcastable_from : std::false_type {};

	template <typename Inst>
	struct is_tightly_broadcastable_from<Inst, Inst> : std::true_type {};

	template <typename From, typename ToElem, isize ToSize>
	requires (!std::is_same_v<From, array<ToElem, ToSize>>)
	struct is_tightly_broadcastable_from<From, array<ToElem, ToSize>> {
		static constexpr bool value =
			is_broadcastable_from_v<From, array<ToElem, ToSize>>
			&& !(is_broadcastable_from_v<From, ToElem>);
	};

	template <typename From, typename To>
	inline constexpr bool is_tightly_broadcastable_from_v = is_tightly_broadcastable_from<From, To>::value;

	template <typename From, typename To>
	concept tightly_broadcastable_to = is_tightly_broadcastable_from_v<From, To>;



	// map, for loop fusion

	template <typename Fn, typename Scalar, typename ... Args>
	requires
		(!is_array_v<Scalar>)
		&& (true && ... && std::is_same_v<Scalar, Args>)
	void map(Fn fn, Scalar &scalar, Args ... args)
	{
		fn(scalar, args...);
	}

	namespace detail {
		template <typename To, typename From>
		auto load_wrt(const From &from, isize i) {
			if constexpr (is_tightly_broadcastable_from_v<From, To>) return from[i];
			else return from;
		}

		template <typename From>
		auto load_simd_reg(const From &from, isize offset) {
			if constexpr (is_array_v<From>) return from.data + offset;
			else return from;
		}
	}

	template <typename Fn, typename Scalar, isize Size, typename ... Froms>
	requires
		simdable<Scalar>
		&& (true && ... && is_broadcastable_from_v<Froms, array<Scalar, Size>>)
	void map(Fn fn, array<Scalar, Size> &to, Froms ... froms)
	{
		for (isize i = 0; i < Size / avx_stride<Scalar>; ++i) {
			avx_reg<Scalar> mto(to.data + i * avx_stride<Scalar>);
			fn(mto, avx_reg<Scalar>(detail::load_simd_reg(froms, i * avx_stride<Scalar>))...);
			mto.storeu(to.data + i * avx_stride<Scalar>);
		}

		static_assert(avx_stride<Scalar> == 2 * sse_stride<Scalar>);
		for (isize i = 2 * (Size / avx_stride<Scalar>); i < Size / sse_stride<Scalar>; ++i) {
			sse_reg<Scalar> mto(to.data + i * sse_stride<Scalar>);
			fn(mto, sse_reg<Scalar>(detail::load_simd_reg(froms, i * sse_stride<Scalar>))...);
			mto.storeu(to.data + i * sse_stride<Scalar>);
		}

		for (isize i = sse_stride<Scalar> * (Size / sse_stride<Scalar>); i < Size; ++i) {
			fn(to[i], detail::load_wrt<array<Scalar, Size>>(froms, i)...);
		}
	}

	template <typename Fn, typename ToElem, isize ToSize, typename ... Froms>
	requires
		(is_array_v<ToElem> || (!simdable<ToElem>))		// an array, of arrays or non-simd-able primitives
		&& (true && ... && is_broadcastable_from_v<Froms, array<ToElem, ToSize>>)
		void map(Fn fn, array<ToElem, ToSize> &to, Froms ... froms)
	{
		for (isize i = 0; i < ToSize; ++i)
			map<Fn>(fn, to[i], detail::load_wrt<array<ToElem, ToSize>, Froms>(froms, i)...);
	}

	

	/* implementation */



	// array

	template <typename T, isize S>
	struct array
	{
		
		/* standard layout */

		T data[S];

		
		/* waffle::array is trivial as long as its primitive is trivial */
		
		array() = default;
		array(const array &) = default;
		array & operator=(const array &) = default;
		array(array &&) = default;
		array & operator=(array &&) = default;
		~array() = default;


		/* ctor / assignment when the argument is broadcastable to T */
		
		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		explicit array(const From &from) {
			auto copy = [] <typename T> (T & a, const T & b) { a = b; };
			map(copy, *this, from);
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		array & operator=(const From &from) {
			auto copy = [] <typename T> (T & a, const T & b) { a = b; };
			map(copy, *this, from);
		}


		/* binary operation */

		// add

		template <broadcastable_to<array> From>
		array & operator+=(const From &from) {
			auto add_assign = [] <typename T> (T & a, const T & b) { a += b; };
			map(add_assign, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator+(const From &from, const array &to) {
			auto add_copy = [] <typename T> (T & a, const T & b, const T & c) { a = b + c; };
			array result;
			map(add_copy, result, from, to);
			return result;
		}
		
		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator+(const array &to, const From &from) {
			return from + to;
		}

		// sub

		template <broadcastable_to<array> From>
		array & operator-=(const From &from) {
			auto binary_sub = [] <typename T> (T & a, const T & b) { a -= b; };
			map(binary_sub, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator-(const From &from, const array &to) {
			auto ternary_sub = [] <typename T> (T & a, const T & b, const T & c) { a = b - c; };
			array result;
			map(ternary_sub, result, from, to);
			return result;
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator-(const array &to, const From &from) {
			auto ternary_sub = [] <typename T> (T & a, const T & b, const T & c) { a = b - c; };
			array result;
			map(ternary_sub, result, to, from);
			return result;
		}

		// mul

		template <broadcastable_to<array> From>
		array & operator*=(const From &from) {
			auto binary_mul = [] <typename T> (T & a, const T & b) { a *= b; };
			map(binary_mul, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator*(const From &from, const array &to) {
			auto ternary_mul = [] <typename T> (T & a, const T & b, const T & c) { a = b * c; };
			array result;
			map(ternary_mul, result, from, to);
			return result;
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator*(const array &to, const From &from) {
			return from * to;
		}

		// div

		template <broadcastable_to<array> From>
		array & operator/=(const From &from) {
			auto binary_div = [] <typename T> (T & a, const T & b) { a /= b; };
			map(binary_div, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator/(const From &from, const array &to) {
			auto ternary_div = [] <typename T> (T & a, const T & b, const T & c) { a = b / c; };
			array result;
			map(ternary_div, result, from, to);
			return result;
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator/(const array &to, const From &from) {
			auto ternary_div = [] <typename T> (T & a, const T & b, const T & c) { a = b / c; };
			array result;
			map(ternary_div, result, to, from);
			return result;
		}


		/* data access */

		T & operator[](isize i) { assert(i < S); return data[i]; }
		const T & operator[](isize i) const { assert(i < S); return data[i]; }


		/* output */

		friend std::ostream & operator<<(std::ostream &os, const array &A) {
			// dim 1: [ ___ ,\t ___ ,\t ___ ]
			// dim n: [ ___ ,\n ___ ,\n ___ ]
			os << "[" << A[0];
			for (isize i = 1; i < S; ++i) os << ',' << (array_depth_v<array> == 1 ? '\t' : '\n') << A[i];
			os << "]";
			return os;
		}

	};

}