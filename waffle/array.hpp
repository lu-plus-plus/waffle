#pragma once

#include <type_traits>
#include <concepts>
#include <functional>

#include <cassert>

#include <iostream>

#include "integer.hpp"
#include "intrinsics.hpp"



namespace waffle
{

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

	namespace test {
		using array_et = get_array_element_type<ndarray<float, 3, 4>>;
		using array_pt = get_array_primitive_type<ndarray<float, 3, 4>>;
		bool array_s = get_array_size<ndarray<float, 3, 4>>;
		isize array_d = get_array_depth<ndarray<float, 3, 4>>;
	}



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

	template <typename From, typename To>
	concept broadcastable_to = is_broadcastable_from_v<From, To>;

	namespace test {
		bool bc1 = is_broadcastable_from_v<float, float>;
		bool bc2 = is_broadcastable_from_v<float, ndarray<float, 3, 4>>;
		bool bc3 = is_broadcastable_from_v<ndarray<float, 3, 5>, ndarray<float, 2, 3, 4, 5, 6>>;
		bool bc4 = is_broadcastable_from_v<ndarray<float, 5, 4>, ndarray<float, 2, 3, 4, 5, 6>>;
	}


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

	namespace test {
		bool dbc0 = is_directly_broadcastable_from_v<float, float>;
		bool dbc1 = is_directly_broadcastable_from_v<ndarray<float, 3, 4>, ndarray<float, 3, 4, 5>>;
		bool dbc2 = is_directly_broadcastable_from_v<ndarray<float, 3, 5>, ndarray<float, 3, 4, 5>>;
	}



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

	namespace test {
		bool tbc0 = is_tightly_broadcastable_from_v<float, float>;
		bool tbc1 = is_tightly_broadcastable_from_v<ndarray<float, 3, 5>, ndarray<float, 3, 4, 5>>;
		bool tbc2 = is_tightly_broadcastable_from_v<ndarray<float, 3, 5>, ndarray<float, 3, 4, 5, 5>>;
	}



	/* declaration of operations */

	template <typename From, typename To, template <typename> typename Op>
	requires is_broadcastable_from_v<From, To>
	struct assign_base;

	namespace assign_with {
		template <typename T>
		struct copy {
			void operator()(const T &from, T &to) const { to = from; }
		};

		template <typename T>
		struct add {
			void operator()(const T &from, T &to) const { to += from; }
		};

		template <typename T>
		struct sub {
			void operator()(const T &from, T &to) const { to -= from; }
		};
	}

	template <typename From, typename To>
	using copy_assign = assign_base<From, To, assign_with::copy>;

	template <typename From, typename To>
	using add_assign = assign_base<From, To, assign_with::add>;

	template <typename From, typename To>
	using sub_assign = assign_base<From, To, assign_with::sub>;

	

	/* implementation */

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
		array(const From &from) : data() {
			// std::cout << "copy ctor with dim = " << S << std::endl;
			copy_assign<From, array>()(from, *this);
		}

		template <broadcastable_to<array> From>
		requires (!std::is_same_v<From, array>)
		array & operator=(const From &from) {
			std::cout << "assign ctor with dim = " << S << std::endl;
			copy_assign<From, array>()(from, *this);
			return *this;
		}

		template <broadcastable_to<array> From>
		array & operator+=(const From &from) {
			// std::cout << "add-assign with dim = " << S << std::endl;
			add_assign<From, array>()(from, *this);
			return *this;
		}

		template <broadcastable_to<array> From>
		array & operator-=(const From &from) {
			std::cout << "sub-assign with dim = " << S << std::endl;
			sub_assign<From, array>()(from, *this);
			return *this;
		}

		T & operator[](isize i) { assert(i < S); return data[i]; }
		const T & operator[](isize i) const { assert(i < S); return data[i]; }
	};

	//template <isize S>
	//struct array<float, S>
	//{	
	//	float data[S];

	//	array() = default;
	//	array(const array &) = default;
	//	array & operator=(const array &) = default;

	//	array(const float &f) : data() {
	//		for (isize i = 0; i < S; ++i) data[i] = f;
	//	}

	//	float & operator[](isize i) { assert(i < S); return data[i]; }
	//	const float & operator[](isize i) const { assert(i < S); return data[i]; }

	//	array & operator+=(const array &rhs) {
	//		for (isize i = 0; i < S; i += 8) {
	//			mregf<8> a(this->data + i);
	//			mregf<8> b(rhs.data + i);
	//			(a + b).storeu(this->data + i);
	//		}
	//		for (isize i = (S / 8) * 8; i < S; i += 4) {
	//			mregf<4> a(this->data + i);
	//			mregf<4> b(rhs.data + i);
	//			(a + b).storeu(this->data + i);
	//		}
	//		for (isize i = (S / 4) * 4; i < S; ++i) {
	//			this->data[i] += rhs.data[i];
	//		}
	//	}
	//};

	// scalar to 1-d array: SIMD or fallback?
	// dispatch in constexpr if

	template <typename Prim, isize ToSize, template <typename> typename Op>
	requires (!is_array_v<Prim>)
	struct assign_base<Prim, array<Prim, ToSize>, Op>
	{
		void operator()(const Prim &from, array<Prim, ToSize> &to) const {
			if constexpr (simdable<Prim>) {
				for (isize i = 0; i + avx_stride<Prim> <= ToSize; i += avx_stride<Prim>) {
					avx_reg<Prim> mfrom(from);
					avx_reg<Prim> mto(to.data);
					Op<avx_reg<Prim>>()(mfrom, mto);
					mto.storeu(to.data + i);
					//std::cout << "AVX call" << std::endl;
				}
				for (isize i = (ToSize / 8) * 8; i + sse_stride<Prim> <= ToSize; i += sse_stride<Prim>) {
					sse_reg<Prim> mfrom(from);
					sse_reg<Prim> mto(to.data + i);
					Op<sse_reg<Prim>>()(mfrom, mto);
					mto.storeu(to.data + i);
					//std::cout << "SSE call" << std::endl;
				}
				for (isize i = (ToSize / 4) * 4; i < ToSize; ++i) {
					Op<Prim>()(from, to[i]);
					//std::cout << "fallback to scalar" << std::endl;
				}
			}
			else {
				for (isize i = 0; i < ToSize; ++i) Op<Prim>()(from, to[i]);
			}
		}
	};

	// 1-d array to 1-d array: SIMD or fallback?
	// dispatch in constexpr if

	template <typename Prim, isize Size, template <typename> typename Op>
	requires (!is_array_v<Prim>)
		struct assign_base<array<Prim, Size>, array<Prim, Size>, Op>
	{
		void operator()(const array<Prim, Size> &from, array<Prim, Size> &to) const {
			if constexpr (simdable<Prim>) {
				// ... ...
				assert(false);
			}
			else {
				for (isize i = 0; i < Size; ++i) Op<Prim>()(from[i], to[i]);
			}
		}
	};

	// problem reduction: scalar to n-d array, n > 1

	template <typename FromPrim, typename ToElem, isize ToSize, template <typename> typename Op>
	requires (
		!is_array_v<FromPrim>
		&& !std::is_same_v<FromPrim, ToElem>
		&& std::is_same_v<FromPrim, get_array_primitive_type<array<ToElem, ToSize>>>
		)
	struct assign_base<FromPrim, array<ToElem, ToSize>, Op>
	{
		void operator()(const FromPrim &from, array<ToElem, ToSize> &to) {
			for (isize i = 0; i < ToSize; ++i) assign_base<FromPrim, ToElem, Op>()(from, to[i]);
		}
	};

	// problem reduction: m-d array to n-d array

	template <typename FromElem, isize FromSize, typename ToElem, isize ToSize, template <typename> typename Op>
	requires (
		!is_array_v<FromElem>
		&& is_broadcastable_from_v<array<FromElem, FromSize>, array<ToElem, ToSize>>
		)
		struct assign_base<array<FromElem, FromSize>, array<ToElem, ToSize>, Op>
	{
		void operator()(const array<FromElem, FromSize> &from, array<ToElem, ToSize> &to) {
			if constexpr (is_tightly_broadcastable_from_v<array<FromElem, FromSize>, array<ToElem, ToSize>>) {
				static_assert(FromSize == ToSize);
				for (isize i = 0; i < ToSize; ++i) assign_base<FromElem, ToElem, Op>()(from[i], to[i]);
			}
			else {
				for (isize i = 0; i < ToSize; ++i) assign_base<array<FromElem, FromSize>, ToElem, Op>()(from, to[i]);
			}
		}
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
			for (isize i = 0; i < Size; ++i) Eval<Prim, Elem>(p, A[i])(result[i]);
		}
	};

	template <typename FromElem, isize FromSize, typename ToElem, isize ToSize>
	struct Eval<array<FromElem, FromSize>, array<ToElem, ToSize>> {
		array<FromElem, FromSize> &from;
		array<ToElem, ToSize> &to;
		Eval(array<FromElem, FromSize> &from, array<ToElem, ToSize> &to) : from(from), to(to) {}

		void operator()(array<ToElem, ToSize> &result) {
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


	
	template <typename From, typename To>
	struct ArrayExpr;

	/* primitive => primitive[N] */
	template <typename Prim, isize ToSize>
	requires (!arrayname<Prim>)
	struct ArrayExpr<Prim, array<Prim, ToSize>>
	{
		const Prim &ref;
		ArrayExpr(const Prim &prim) : ref(prim) {}

		const Prim & operator[](isize i) const { assert(i < ToSize); return ref; }
	};

	/* primititve => primitive[N1]...[Nm] */
	template <typename Prim, typename ToElem, isize ToSize>
	requires (!arrayname<Prim> && !std::is_same_v<Prim, ToElem>)
	struct ArrayExpr<Prim, array<ToElem, ToSize>>
	{
		const Prim &ref;
		ArrayExpr(const Prim &prim) : ref(prim) {}

		ArrayExpr<Prim, ToElem> operator[](isize i) const {
			assert(i < ToSize);
			return ArrayExpr<Prim, ToElem>(ref);
		}
	};

	/* n-d array => n-d array, totally the same */
	//template <typename Elem, isize Size>
	//struct ArrayExpr<array<Elem, Size>, array<Elem, Size>>
	//{
	//	array<Elem, Size> &ref;
	//	ArrayExpr(array<Elem, Size> &arr) : ref(arr) {}
	//	Elem & operator[](isize i) const {
	//		assert(i < Size);
	//		return ref[i];
	//	}
	//};

	template <typename FromElem, typename ToElem, isize Size>
	requires is_tightly_broadcastable_from_v<array<FromElem, Size>, array<ToElem, Size>>
	struct ArrayExpr<array<FromElem, Size>, array<ToElem, Size>>
	{
		const array<FromElem, Size> &ref;
		ArrayExpr(const array<FromElem, Size> &arr) : ref(arr) {}

		ArrayExpr<FromElem, ToElem> operator[](isize i) const {
			assert(i < Size);
			return ArrayExpr<FromElem, ToElem>(ref[i]);
		}
	};

	template <typename FromElem, isize FromSize, typename ToElem, isize ToSize>
	requires (!is_tightly_broadcastable_from_v<array<FromElem, FromSize>, array<ToElem, ToSize>>)
	struct ArrayExpr<array<FromElem, FromSize>, array<ToElem, ToSize>>
	{
		const array<FromElem, FromSize> &ref;
		ArrayExpr(const array<FromElem, FromSize> &arr) : ref(arr) {}
		
		ArrayExpr<array<FromElem, FromSize>, ToElem> operator[](isize i) const {
			assert(i < ToSize);
			return ArrayExpr<array<FromElem, FromSize>, ToElem>(ref);
		}
	};



	template <typename From, typename ToElem, isize ToSize>
	array<ToElem, ToSize> operator+(
		const ArrayExpr<From, array<ToElem, ToSize>> &expr_a,
		const array<ToElem, ToSize> &b)
	{
		array<ToElem, ToSize> c;
		for (isize i = 0; i < ToSize; ++i) c[i] = expr_a[i] + b[i];
		return c;
	}

	template <typename From, arrayname To>
	requires is_broadcastable_from_v<From, To>
	To operator+(const From &from, const To &to)
	{
		return ArrayExpr<From, To>(from) + to;
	}

	template <typename From, arrayname To>
	requires (!is_broadcastable_from_v<To, From> && is_broadcastable_from_v<From, To>)
	To operator+(const To &to, const From &from)
	{
		return ArrayExpr<From, To>(from) + to;
	}

}