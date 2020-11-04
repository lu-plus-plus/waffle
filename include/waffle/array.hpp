#pragma once

#include <type_traits>
// #include <concepts>

#include <cassert>

#include <iostream>

#include "arithmetics.hpp"
#include "intrinsics.hpp"



namespace waffle
{

	/* declaration */



	// array

	template <typename T, usize S>
	struct array;



	// ndarray

	namespace detail {
		template <typename T, usize ... Dims>
		struct ndarray_helper;

		template <typename T, usize InnerDim, usize ... Dims>
		struct ndarray_helper<T, InnerDim, Dims...> {
			using type = typename ndarray_helper<array<T, InnerDim>, Dims...>::type;
		};

		template <typename T, usize OutestDim>
		struct ndarray_helper<T, OutestDim> {
			using type = array<T, OutestDim>;
		};
	}

	template <typename T, usize ... Dims>
	requires (sizeof...(Dims) > 0)
	using ndarray = typename detail::ndarray_helper<T, Dims...>::type;



	// type traits of _array_

	template <typename InstType>
	struct is_array : std::false_type {};

	template <typename T, usize S>
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
			static constexpr usize depth = 0;
		};

		template <typename T, usize S>
		struct unwrap_helper<array<T, S>> {
			using primitive = typename unwrap_helper<T>::primitive;
			static constexpr usize depth = 1 + unwrap_helper<T>::depth;
		};
	}

	template <typename T, usize S>
	struct unwrap_array<array<T, S>> {
		using element = T;
		static constexpr usize size = S;
		using primitive = typename detail::unwrap_helper<array<T, S>>::primitive;
		static constexpr usize depth = detail::unwrap_helper<array<T, S>>::depth;
	};

	template <arrayname T>
	using array_element_t = typename unwrap_array<T>::element;

	template <arrayname T>
	inline constexpr usize array_size_v = unwrap_array<T>::size;

	template <arrayname T>
	using array_primitive_t = typename unwrap_array<T>::primitive;

	template <arrayname T>
	inline constexpr usize array_depth_v = unwrap_array<T>::depth;



	template <typename From, typename To>
	struct is_broadcastable_from : std::false_type {};

	template <typename Prim>
	requires (!arrayname<Prim>)
	struct is_broadcastable_from<Prim, Prim> : std::true_type {};
	
	template <typename Prim, typename Elem, usize Size>
	requires (std::is_same_v<Prim, array_primitive_t<array<Elem, Size>>>)
	struct is_broadcastable_from<Prim, array<Elem, Size>> : std::true_type {};

	template <typename FromType, usize FromSize, typename ToType, usize ToSize>
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

	template <typename From, typename ToElem, usize ToSize>
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

	template <typename From, typename ToElem, usize ToSize>
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

	template <typename Fn, typename To, typename ... Froms>
	void map(Fn fn, To &to, Froms ... froms);

	namespace detail {
		template <typename To, typename From>
		auto subscript_proxy(const From &from, usize i) {
			if constexpr (is_tightly_broadcastable_from_v<From, To>) return from[i];
			else return from;
		}

		template <typename From>
		auto simd_data_proxy(const From &from, usize offset) {
			if constexpr (is_array_v<From>) return from.data + offset;
			else return from;
		}
	}

	template <typename Fn, typename To, typename ... Froms>
	void map(Fn fn, To &to, Froms ... froms)
	{
		if constexpr (is_array_v<To>) {
			// nested array, or non-simd-able 1D array
			if constexpr (array_depth_v<To> > 1 || !simdable<array_element_t<To>>) {
				for (usize i = 0; i < array_size_v<To>; ++i)
					map(fn, to[i], detail::subscript_proxy<To>(froms, i)...);
			}
			// simd-able 1D array
			else {
				constexpr usize Size = array_size_v<To>;
				using Prim = array_element_t<To>;
				static_assert(avx_stride<Prim> == 2 * sse_stride<Prim>);

				for (usize i = 0; i < Size / avx_stride<Prim>; ++i) {
					avx_reg<Prim> mto(to.data + i * avx_stride<Prim>);
					fn(mto, avx_reg<Prim>(detail::simd_data_proxy(froms, i * avx_stride<Prim>))...);
					mto.storeu(to.data + i * avx_stride<Prim>);
				}
				
				for (usize i = 2 * (Size / avx_stride<Prim>); i < Size / sse_stride<Prim>; ++i) {
					sse_reg<Prim> mto(to.data + i * sse_stride<Prim>);
					fn(mto, sse_reg<Prim>(detail::simd_data_proxy(froms, i * sse_stride<Prim>))...);
					mto.storeu(to.data + i * sse_stride<Prim>);
				}

				for (usize i = sse_stride<Prim> * (Size / sse_stride<Prim>); i < Size; ++i) {
					fn(to[i], detail::subscript_proxy<array<Prim, Size>>(froms, i)...);
				}
			}
		}
		else {
			// scalar
			fn(to, froms...);
		}
	}



	/* reduce, for horizontal operations */

	template <arrayname Src, broadcastable_to<Src> Dest, typename Fn>
	void reduce_inplace(const Src &src, Dest &dest, Fn fn)
	{
		if constexpr (array_depth_v<Src> == 1) {
			if constexpr (is_array_v<Dest>)
				fn(dest, src);
			else
				for (usize i = 0; i < array_size_v<Src>; ++i) fn(dest, src[i]);
		}
		else {
			if constexpr (is_tightly_broadcastable_from_v<Dest, Src>)
				for (usize i = 0; i < array_size_v<Src>; ++i) reduce_inplace(src[i], dest[i], fn);
			else
				for (usize i = 0; i < array_size_v<Src>; ++i) reduce_inplace(src[i], dest, fn);
		}
	}

	template <arrayname Src, broadcastable_to<Src> Dest, typename Fn>
	Dest reduce(const Src &src, const Dest &init, Fn fn)
	{
		Dest dest(init);
		reduce_inplace(src, dest, fn);
		return dest;
	}



	/* commonly used horizontal operations */

	namespace inplace {
		struct plus {
			template <typename T>
			void operator()(T &dest, const T &src) { dest += src; }
		};
	}

	template <arrayname Src, broadcastable_to<Src> Dest>
	void sum_inplace(const Src &src, Dest &dest)
	{
		reduce_inplace(src, dest, inplace::plus());
	}

	template <arrayname Src, broadcastable_to<Src> Dest>
	Dest sum(const Src &src, const Dest &init)
	{
		return reduce(src, init, inplace::plus());
	}

	template <arrayname Src>
	requires requires { array_primitive_t<Src>(0); }
	array_primitive_t<Src> sum(const Src &src)
	{
		return sum(src, array_primitive_t<Src>(0));
	}

	

	/* implementation */



	// array

	template <typename T, usize S>
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
			auto copy = [] <typename P> (P & a, const P & b) { a = b; };
			map(copy, *this, from);
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		array & operator=(const From &from) {
			auto copy = [] <typename P> (P & a, const P & b) { a = b; };
			map(copy, *this, from);
		}


		/* binary operation */

		// add

		template <broadcastable_to<array> From>
		array & operator+=(const From &from) {
			auto add_assign = [] <typename P> (P & a, const P & b) { a += b; };
			map(add_assign, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator+(const From &from, const array &to) {
			auto add_copy = [] <typename P> (P & a, const P & b, const P & c) { a = b + c; };
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
			auto binary_sub = [] <typename P> (P & a, const P & b) { a -= b; };
			map(binary_sub, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator-(const From &from, const array &to) {
			auto ternary_sub = [] <typename P> (P & a, const P & b, const P & c) { a = b - c; };
			array result;
			map(ternary_sub, result, from, to);
			return result;
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator-(const array &to, const From &from) {
			auto ternary_sub = [] <typename P> (P & a, const P & b, const P & c) { a = b - c; };
			array result;
			map(ternary_sub, result, to, from);
			return result;
		}

		// mul

		template <broadcastable_to<array> From>
		array & operator*=(const From &from) {
			auto binary_mul = [] <typename P> (P & a, const P & b) { a *= b; };
			map(binary_mul, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator*(const From &from, const array &to) {
			auto ternary_mul = [] <typename P> (P & a, const P & b, const P & c) { a = b * c; };
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
			auto binary_div = [] <typename P> (P & a, const P & b) { a /= b; };
			map(binary_div, *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator/(const From &from, const array &to) {
			auto ternary_div = [] <typename P> (P & a, const P & b, const P & c) { a = b / c; };
			array result;
			map(ternary_div, result, from, to);
			return result;
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		friend array operator/(const array &to, const From &from) {
			auto ternary_div = [] <typename P> (P & a, const P & b, const P & c) { a = b / c; };
			array result;
			map(ternary_div, result, to, from);
			return result;
		}


		/* data access */

		T & operator[](usize i) { assert(i < S); return data[i]; }
		const T & operator[](usize i) const { assert(i < S); return data[i]; }


		/* output */

		friend std::ostream & operator<<(std::ostream &os, const array &A) {
			// dim 1: [ ___ ,\t ___ ,\t ___ ]
			// dim n: [ ___ ,\n ___ ,\n ___ ]
			os << "[" << A[0];
			for (usize i = 1; i < S; ++i) os << ',' << (array_depth_v<array> == 1 ? '\t' : '\n') << A[i];
			os << "]";
			return os;
		}

	};

}