#pragma once

#include <cstdint>
#include <type_traits>
#include <concepts>
#include <cassert>

#include <iostream>

namespace waffle
{
	using isize = std::int32_t;
	using usize = std::uint32_t;

	
	
	template <typename T, isize S>
	struct array;

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

		template <isize ... Dims>
		concept non_empty_isize_pack = (sizeof...(Dims) > 0);
	}

	template <typename T, isize ... Dims>
	requires detail::non_empty_isize_pack<Dims...>
	using ndarray = typename detail::ndarray_helper<T, Dims...>::type;



	template <typename InstType>
	struct is_array : std::false_type {};

	template <typename T, isize S>
	struct is_array<array<T, S>> : std::true_type {};

	template <typename InstType>
	inline constexpr bool is_array_v = is_array<InstType>::value;

	template <typename InstType>
	concept arrayname = is_array_v<InstType>;



	namespace detail {
		template <typename T>
		struct unwrap_helper {
			using primitive_type = T;
			static constexpr isize depth = 0;
		};

		template <typename T, isize S>
		struct unwrap_helper<array<T, S>> {
			using primitive_type = typename unwrap_helper<T>::primitive_type;
			static constexpr isize depth = 1 + unwrap_helper<T>::depth;
		};
	}

	template <arrayname T>
	struct unwrap_array;

	template <typename T, isize S>
	struct unwrap_array<array<T, S>> {
		using element_type = T;
		static constexpr isize size = S;
		using primitive_type = typename detail::unwrap_helper<array<T, S>>::primitive_type;
		static constexpr isize depth = detail::unwrap_helper<array<T, S>>::depth;
	};

	template <arrayname T>
	using get_array_element_type = typename unwrap_array<T>::element_type;

	template <arrayname T>
	inline constexpr isize get_array_size = unwrap_array<T>::size;

	template <arrayname T>
	using get_array_primitive_type = typename unwrap_array<T>::primitive_type;

	template <arrayname T>
	inline constexpr isize get_array_depth = unwrap_array<T>::depth;

	// simple test
	//using t1 = get_array_element_type<ndarray<float, 3, 4>>;
	//using pt1 = get_array_primitive_type<ndarray<float, 3, 4>>;
	//bool s1 = get_array_size<ndarray<float, 3, 4>>;
	//isize d1 = get_array_depth<ndarray<float, 3, 4>>;



	template <typename From, typename To>
	struct is_broadcastable_from : std::false_type {};

	template <typename Prim>
	requires (!arrayname<Prim>)
	struct is_broadcastable_from<Prim, Prim> : std::true_type {};
	
	template <typename Prim, typename Elem, isize Size>
	requires (std::is_same_v<Prim, get_array_primitive_type<array<Elem, Size>>>)
	struct is_broadcastable_from<Prim, array<Elem, Size>> : std::true_type {};

	template <typename FromType, isize FromSize, typename ToType, isize ToSize>
	struct is_broadcastable_from<array<FromType, FromSize>, array<ToType, ToSize>> {
		static constexpr bool value =
			is_broadcastable_from<array<FromType, FromSize>, ToType>::value
			|| (FromSize == ToSize && is_broadcastable_from<FromType, ToType>::value);
	};

	template <typename From, typename To>
	inline constexpr bool is_broadcastable_from_v = is_broadcastable_from<From, To>::value;

	template <typename To, typename From>
	concept broadcastable_from = is_broadcastable_from_v<From, To>;

	// simple test
	bool bc1 = is_broadcastable_from_v<float, float>;
	bool bc2 = is_broadcastable_from_v<float, ndarray<float, 3, 4>>;
	bool bc3 = is_broadcastable_from_v<ndarray<float, 3, 5>, ndarray<float, 2, 3, 4, 5, 6>>;
	bool bc4 = is_broadcastable_from_v<ndarray<float, 5, 4>, ndarray<float, 2, 3, 4, 5, 6>>;



	/* implementation */

	template <typename T, isize S>
	struct array
	{
		using primitive_type = get_array_primitive_type<array<T, S>>;

		T data[S];

		array() = default;
		array(const array &) = default;
		array & operator=(const array &) = default;

		array(const primitive_type &prim) {
			for (isize i = 0; i < S; ++i) data[i] = T(prim);
		}

		T & operator[](isize i) { assert(i < S); return data[i]; }
		const T & operator[](isize i) const { assert(i < S); return data[i]; }
	};

	template <typename T, isize S>
	std::ostream & operator<<(std::ostream &os, const array<T, S> &A)
	{
		// dim 1: "[\t" ",\t" ... ",\t" "]\n"
		// dim n: "[\t" "\n\t" ... "\n\t" "]\n"
		os << "[" << A[0];
		for (isize i = 1; i < S; ++i) {
			os << (get_array_depth<array<T, S>> == 1 ? ",\t" : "\n") << A[i];
		}
		os << "]";
		return os;
	}



	/* broadcasted evaluation */

	template <typename TyA, typename TyB>
	struct Eval;

	template <typename Prim>
	requires (!arrayname<Prim>)
	struct Eval<Prim, Prim> {
		Prim &a, &b;
		Eval(Prim &a, Prim &b) : a(a), b(b) {}

		void operator()(Prim &result) { result = a + b; }
	};

	template <typename Prim, typename Elem, isize Size>
	requires (std::is_same_v<Prim, get_array_primitive_type<array<Elem, Size>>>)
	struct Eval<Prim, array<Elem, Size>> {
		Prim &p;
		array<Elem, Size> &A;
		Eval(Prim &p, array<Elem, Size> &A) : p(p), A(A) {}

		void operator()(array<Elem, Size> &result) {
			for (isize i = 0; i < Size; ++i) result[i] = p + A[i];
		}
	};

	//template <typename Prim, typename Elem, isize Size>
	//requires (!arrayname<Prim>)
	//struct Eval<Prim, array<Elem, Size>> {
	//	Prim &prim;
	//	array<Elem, Size> &arr;

	//	void operator()(array<Elem, Size> &result) {
	//		for (isize i = 0; i < Size; ++i) Eval(prim, arr[i])(result[i]);
	//	}
	//};

	template <typename FromElem, isize FromSize, typename ToElem, isize ToSize>
	struct Eval<array<FromElem, FromSize>, array<ToElem, ToSize>> {
		array<FromElem, FromSize> &from;
		array<ToElem, ToSize> &to;
		Eval(array<FromElem, FromSize> &from, array<ToElem, ToSize> &to) : from(from), to(to) {}

		void operator()(array<ToElem, ToSize> &result) {
			//if constexpr (FromSize == ToSize) {
			//	for (isize i = 0; i < ToSize; ++i) Eval<FromElem, ToElem>(from[i], to[i])(result[i]);
			//}
			//else {
			//	for (isize i = 0; i < ToSize; ++i) Eval<array<FromElem, FromSize>, ToElem>(from, to[i])(result[i]);
			//}
			for (isize i = 0; i < ToSize; ++i) {
				if constexpr (FromSize == ToSize && !is_broadcastable_from_v<array<FromElem, FromSize>, ToElem>) {
					Eval<FromElem, ToElem>(from[i], to[i])(result[i]);
				}
				else {
					Eval<array<FromElem, FromSize>, ToElem>(from, to[i])(result[i]);
				}
			}
		}
	};
}