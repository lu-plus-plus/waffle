#pragma once

#include <cstdint>

namespace waffle
{

	using isize = std::int64_t;
	using usize = std::uint64_t;

	using byte = std::uint8_t;

	namespace literals
	{
		constexpr isize operator "" _is(unsigned long long int i) { return i; }
		constexpr usize operator "" _us(unsigned long long int i) { return i; }
	}



	template <typename T>
	struct scalar {

		T value;

		scalar() = default;
		scalar(const scalar &) = default;
		scalar & operator=(const scalar &) = default;

		scalar(const T &val) : value(val) {}
		scalar & operator=(const T &val) { value = val; return *this; }

		operator T() const { return value; }
		operator T &() { return value; }

	};
	
	namespace detail {
		template <typename T, template <typename> typename Container>
		struct soa_base : public Container<T>
		{
			using Container<T>::Container;
		};
	}

	template <template <typename> typename Container = scalar>
	struct f32 : public detail::soa_base<float, Container>
	{
		using detail::soa_base<float, Container>::soa_base;
	};

	template <template <typename> typename Container = scalar>
	struct f64 : public detail::soa_base<double, Container>
	{
		using detail::soa_base<double, Container>::soa_base;
	};

}