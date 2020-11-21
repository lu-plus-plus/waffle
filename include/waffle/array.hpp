#pragma once

#include <type_traits>
#include <utility>

#include <cassert>

#include <iostream>
#include <sstream>

#include "arithmetics.hpp"
#include "intrinsics.hpp"
#include "utils.hpp"



namespace waffle
{

	/* declaration */



	//	array

	template <typename T, usize S>
	struct array;



	//	ndarray

	template <typename T, usize ... Dims>
	using ndarray = left_fold_t<array, T, Dims...>;



	//	type trait: is it an array?

	template <typename InstType>
	struct is_array : std::false_type {};

	template <typename T, usize S>
	struct is_array<array<T, S>> : std::true_type {};

	template <typename InstType>
	inline constexpr bool is_array_v = is_array<InstType>::value;

	template <typename InstType>
	concept arrayname = is_array_v<InstType>;



	template <typename T>
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

	template <typename T>
	struct unwrap_array {
		using primitive = T;
		static constexpr usize depth = 0;
	};

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

	template <typename T>
	using array_primitive_t = typename unwrap_array<T>::primitive;

	template <typename T>
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



	template <typename From, typename To>
	inline constexpr bool is_properly_broadcastable_from_v = is_broadcastable_from_v<From, To> && (!std::is_same_v<From, To>);

	template <typename From, typename To>
	concept properly_broadcastable_to = is_properly_broadcastable_from_v<From, To>;



	/*
		When T is NOT an waffle::ndarray, replace_primitive<T, P> returns P.
		When T == ndarray<sth, Dims...>, replace_primitive<T, NewP> returns ndarray<P, Dims...>
	*/
	template <typename T, typename NewPrim>
	struct replace_primitive;

	namespace detail {
		template <typename T, typename NewPrim>
		struct replace_primitive_helper {
			using type = NewPrim;
		};

		template <typename Elem, usize Size, typename NewPrim>
		struct replace_primitive_helper<array<Elem, Size>, NewPrim> {
			using type = array<typename replace_primitive_helper<Elem, NewPrim>::type, Size>;
		};
	}

	template <typename T, typename NewPrim>
	struct replace_primitive {
		using type = typename detail::replace_primitive_helper<T, NewPrim>::type;
	};

	template <typename T, typename NewPrim>
	using replace_primitive_t = typename replace_primitive<T, NewPrim>::type;



	/* map, for loop fusion */

	template <typename Fn, typename To, typename ... Froms>
	requires valid_arguments_to<Fn, array_primitive_t<To>, array_primitive_t<Froms>...>
	// at least, the fallback (scalar) mode should be supported
	void map(Fn fn, To &to, const Froms & ... froms)
	{
		auto load_element = [] <typename From> (const From & from, usize i) {
			if constexpr (is_tightly_broadcastable_from_v<replace_primitive_t<From, void *>, replace_primitive_t<To, void *>>)
				return from[i];
			else
				return from;
		};

		auto load_packet = [] <typename From> (const From & from, usize offset) {
			if constexpr (is_array_v<From>) return from.data + offset;
			else return from;
		};

		if constexpr (is_array_v<To>) {
			// To:		nested array, or non-simd-able 1D array,
			// Froms:	some of them have non-simd-able primitives
			if constexpr (array_depth_v<To> > 1 || !(simdable<array_element_t<To>> && (simdable<array_primitive_t<Froms>> && ...))) {
				for (usize i = 0; i < array_size_v<To>; ++i)
					map(fn, to[i], load_element(froms, i)...);
			}
			// To:		simd-able 1D array
			// Froms:	all have simd-able primitives
			// However, at here we are still not sure whether Fn takes (simd_reg<To's prim>, simd_reg<Froms' prims>...)
			else {
				constexpr usize ToSize = array_size_v<To>;
				using ToPrim = array_element_t<To>;

				// test whether Fn is invocable in the SIMD way
				constexpr bool avxable = valid_arguments_to<Fn, avx_reg<ToPrim>, avx_reg<array_primitive_t<Froms>>...>;
				constexpr bool sseable = valid_arguments_to<Fn, sse_reg<ToPrim>, sse_reg<array_primitive_t<Froms>>...>;

				constexpr usize avx_begin = 0;
				constexpr usize avx_end = avxable ? (ToSize / avx_stride<ToPrim>) : avx_begin;

				constexpr usize sse_begin = avxable ? ((ToSize / avx_stride<ToPrim>) << 1) : 0;
				constexpr usize sse_end = sseable ? (ToSize / sse_stride<ToPrim>) : sse_begin;

				constexpr usize fallback_begin = (avxable | sseable)
					? (sseable
						? (sse_stride<ToPrim> * (ToSize / sse_stride<ToPrim>))
						: (avx_stride<ToPrim> *(ToSize / avx_stride<ToPrim>)))
					: 0;
				constexpr usize fallback_end = ToSize;

				if (avx_begin != avx_end || sse_begin != sse_end)
					std::cout << "vector mode" << std::endl;

				for (usize i = avx_begin; i < avx_end; ++i) {
					avx_reg<ToPrim> mto(to.data + i * avx_stride<ToPrim>);
					fn(mto, avx_reg<array_primitive_t<Froms>>(load_packet(froms, i * avx_stride<array_primitive_t<Froms>>))...);
					mto.storeu(to.data + i * avx_stride<ToPrim>);
				}
				
				for (usize i = sse_begin; i < sse_end; ++i) {
					sse_reg<ToPrim> mto(to.data + i * sse_stride<ToPrim>);
					fn(mto, sse_reg<array_primitive_t<Froms>>(load_packet(froms, i * sse_stride<array_primitive_t<Froms>>))...);
					mto.storeu(to.data + i * sse_stride<ToPrim>);
				}

				for (usize i = fallback_begin; i < fallback_end; ++i) {
					fn(to[i], load_element(froms, i)...);
				}
			}
		}
		else {
			// To:		scalar
			// Froms:	scalars
			std::cout << "scalar mode" << std::endl;
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

	namespace functional::in_place {

		struct plus {
			template <typename T>
			requires requires (T t) { t += t; }
			void operator()(T &dest, const T &src) { dest += src; }
		};

		struct minus {
			template <typename T>
			requires requires (T t) { t -= t; }
			void operator()(T &dest, const T &src) { dest -= src; }
		};

		struct multiplies {
			template <typename T>
			requires requires (T t) { t *= t; }
			void operator()(T &dest, const T &src) { dest *= src; }
		};

		struct divides {
			template <typename T>
			requires requires (T t) { t /= t; }
			void operator()(T &dest, const T &src) { dest /= src; }
		};

		struct modulus {
			template <typename T>
			requires requires (T t) { t %= t; }
			void operator()(T &dest, const T &src) { dest %= src; }
		};

	}

	namespace functional {
		
		struct copy {
			template <typename T>
			T operator()(const T &src) { return src; }

			template <typename T>
			void operator()(T &dest, const T &src) { dest = src; }
		};

		struct plus {
			template <typename T>
			requires requires (T t) { t + t; }
			T operator()(const T &lhs, const T &rhs) { return lhs + rhs; }

			template <typename T>
			requires requires (T t) { t = t + t; }
			void operator()(T &dest, const T &lhs, const T &rhs) { dest = lhs + rhs; }
		};

		struct minus {
			template <typename T>
			requires requires (T t) { t - t; }
			T operator()(const T &lhs, const T &rhs) { return lhs - rhs; }

			template <typename T>
			requires requires (T t) { t = t - t; }
			void operator()(T &dest, const T &lhs, const T &rhs) { dest = lhs - rhs; }
		};

		struct multiplies {
			template <typename T>
			requires requires (T t) { t * t; }
			T operator()(const T &lhs, const T &rhs) { return lhs * rhs; }

			template <typename T>
			requires requires (T t) { t = t * t; }
			void operator()(T &dest, const T &lhs, const T &rhs) { dest = lhs * rhs; }
		};

		struct divides {
			template <typename T>
			requires requires (T t) { t / t; }
			T operator()(const T &lhs, const T &rhs) { return lhs / rhs; }

			template <typename T>
			requires requires (T t) { t = t / t; }
			void operator()(T &dest, const T &lhs, const T &rhs) { dest = lhs / rhs; }
		};

		struct modulus {
			template <typename T>
			requires requires (T t) { t % t; }
			T operator()(const T &lhs, const T &rhs) { return lhs % rhs; }

			template <typename T>
			requires requires (T t) { t = t % t; }
			void operator()(T &dest, const T &lhs, const T &rhs) { dest = lhs % rhs; }
		};

		struct equal_to {
			template <typename S>
			requires requires (S s) { s == s; }
			auto operator()(const S &lhs, const S &rhs) { return lhs == rhs; }

			template <typename D, typename S>
			requires requires (D dest, S src) { dest = src == src; }
			void operator()(D &dest, const S &lhs, const S &rhs) { dest = lhs == rhs; }
		};

	}



	// sum

	template <arrayname Src, broadcastable_to<Src> Dest>
	void sum_inplace(const Src &src, Dest &dest)
	{
		reduce_inplace(src, dest, functional::in_place::plus());
	}

	template <arrayname Src, broadcastable_to<Src> Dest>
	Dest sum(const Src &src, const Dest &init)
	{
		return reduce(src, init, functional::in_place::plus());
	}

	template <arrayname Src>
	requires requires { array_primitive_t<Src>(0); }
	array_primitive_t<Src> sum(const Src &src)
	{
		return sum(src, array_primitive_t<Src>(0));
	}

	// prod

	template <arrayname Src, broadcastable_to<Src> Dest>
	void prod_inplace(const Src &src, Dest &dest)
	{
		reduce_inplace(src, dest, functional::in_place::multiplies());
	}

	template <arrayname Src, broadcastable_to<Src> Dest>
	Dest prod(const Src &src, const Dest &init)
	{
		return reduce(src, init, functional::in_place::multiplies());
	}

	template <arrayname Src>
	requires requires { array_primitive_t<Src>(0); }
	array_primitive_t<Src> prod(const Src &src)
	{
		return prod(src, array_primitive_t<Src>(0));
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


		/* primitive initializer list */

		using prim_t = array_primitive_t<array<T, S>>;

		array(std::initializer_list<prim_t> il) {
			assert(il.size() <= S);
			std::copy_n(il.begin(), std::min(il.size(), S * sizeof(T) / sizeof(prim_t)), data);
		}


		/* ctor / assignment when the argument is broadcastable to T */
		
		template <properly_broadcastable_to<array> From>
		explicit array(const From &from) {
			map(functional::copy(), *this, from);
		}

		template <properly_broadcastable_to<array> From>
		array & operator=(const From &from) {
			map(functional::copy(), *this, from);
		}


		/* arithmetic operation */

		// add

		template <broadcastable_to<array> From>
		array & operator+=(const From &from) {
			map(functional::in_place::plus(), *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator+(const From &from, const array &to) {
			array result;
			map(functional::plus(), result, from, to);
			return result;
		}
		
		template <properly_broadcastable_to<array> From>
		friend array operator+(const array &to, const From &from) {
			return from + to;
		}

		// sub

		template <broadcastable_to<array> From>
		array & operator-=(const From &from) {
			map(functional::in_place::minus(), *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator-(const array &to, const From &from) {
			array result;
			map(functional::minus(), result, to, from);
			return result;
		}

		// mul

		template <broadcastable_to<array> From>
		array & operator*=(const From &from) {
			map(functional::in_place::multiplies(), *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator*(const From &from, const array &to) {
			array result;
			map(functional::multiplies(), result, from, to);
			return result;
		}

		template <properly_broadcastable_to<array> From>
		friend array operator*(const array &to, const From &from) {
			return from * to;
		}

		// div

		template <broadcastable_to<array> From>
		array & operator/=(const From &from) {
			map(functional::in_place::divides(), *this, from);
			return *this;
		}

		template <broadcastable_to<array> From>
		friend array operator/(const array &to, const From &from) {
			array result;
			map(functional::divides(), result, to, from);
			return result;
		}


		/* compare */

		template <broadcastable_to<array> From>
		friend auto operator==(const From &from, const array &to) {
			replace_primitive_t<array, bool> result;
			map(functional::equal_to(), result, from, to);
			return result;
		}

		template <properly_broadcastable_to<array> From>
		friend auto operator==(const array &to, const From &from) {
			return from == to;
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


		/* vectorization */

		constexpr static usize packet_avx() requires simdable<T>;
		constexpr static usize packet_sse() requires simdable<T>;

	};



	namespace slice
	{

		struct all_t {
			explicit constexpr all_t(int) {}
		};
		
		inline constexpr all_t all{ 0 };


		struct end_t {
			explicit constexpr end_t(int) {}
		};
		
		inline constexpr end_t end{ 0 };

		
		struct mid_range_t {
			const usize b, e;
		private:
			mid_range_t(usize b, usize e) : b(b), e(e) {}
			friend mid_range_t range(usize begin, usize end);
		};

		mid_range_t range(usize begin, usize end) { return mid_range_t(begin, end); }


		struct end_range_t {
			usize b;
		private:
			end_range_t(usize b) : b(b) {}
			friend end_range_t range(usize begin, end_t);
		};

		end_range_t range(usize begin, end_t) { assert(begin != 0); return end_range_t(begin); }


		
	}



	template <typename T, usize ND>
	struct varray
	{
	private:
		array<usize, ND> siz;

		T *raw;

		static void copy_raw(varray &self, const varray &other) {
			std::copy(self.raw, other.raw, prod(other.siz));
		}

		// remember to reset (*this) and std::forward the argument
		static void move_raw(varray &self, varray &&other) {
			delete[] self.raw;
			self.raw = other.raw;
			other.raw = nullptr;
		}

	public:

		/* initializing */

		varray(const array<usize, ND> &siz) : siz(siz), raw(new T[prod(siz)]) {}

		varray(const array<usize, ND> &siz, const T &val) : varray(siz) {
			std::fill(raw, raw + prod(siz), val);
		}

		varray(const varray &other) : siz(other.siz), raw(new T[prod(other.siz)]) {
			copy_raw(*this, other);
		}

		varray(varray &&other) : siz(other.siz), raw(nullptr) {
			move_raw(*this, std::forward<varray>(other));
		}

		~varray() { delete[] raw; }

		struct unaligned_assignment : std::exception
		{
			std::string s;
			array<usize, ND> a, b;

			unaligned_assignment(const array<usize, ND> &a, const array<usize, ND> &b) : a(a), b(b), s() {
				std::ostringstream oss;
				oss << "assigning varray(" << b << ") to varray(" << a << ")";
				s = oss.str();
			}

			virtual const char * what() const noexcept override {
				return s.c_str();
			}
		};

		//varray & operator=(const varray &other) {
		//	if (siz != other.siz)	throw unaligned_assignment(siz, other.siz);
		//	else					copy_raw(*this, other);
		//}
		//varray & operator=(varray &&other) {
		//	if (siz != other.siz)	throw unaligned_assignment(siz, other.siz);
		//	else					move_raw(*this, std::forward<varray>(other));
		//}

	};

	//template <arrayname Array, usize OuterDim, usize ... Dims>
	//struct static_view
	//{
	//	Array &arr;

	//	using elem_t = array_element_t<Array>;
	//	static_assert(OuterDim <= array_size_v<Array>);
	//};

	namespace dynamic_view
	{
	
		template <typename Prim, usize ND>
		struct base
		{
			const array<usize, ND> dims;
			
			base(const array<usize, ND> &dims) : dims(dims) {}

			template <usize N>
			requires (N == ND)
			base(const usize (&dims)[N]) : dims(dims) {}

			template <typename ... Args>
			requires (sizeof...(Args) == ND && (... && std::is_integral_v<Args>))
			base(Args ... args) : base({ usize(args)... }) {}
		};

	}

}