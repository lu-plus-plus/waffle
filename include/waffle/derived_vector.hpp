#pragma once

#include "arithmetics.hpp"
#include "array.hpp"

#include <vector>
#include <cassert>

#include <type_traits>



namespace waffle
{
	template <typename Base>
	struct derived_vector
	{
	private:

		std::vector<byte> bytes;
		std::vector<usize> elem_inds;

		template <typename Derived>
		void reserve_for() {
			elem_inds.push_back(bytes.size());
			bytes.resize(bytes.size() + sizeof(Derived));
		}

	public:

		derived_vector() = default;


		usize size() const { return elem_inds.size(); }


		const Base * operator[](usize i) const {
			assert(i < size());
			return reinterpret_cast<const Base *>(bytes.data() + elem_inds[i]);
		}
		Base * operator[](usize i) {
			assert(i < size());
			return reinterpret_cast<Base *>(bytes.data() + elem_inds[i]);
		}

		const Base * back() const { return (*this)[elem_inds.size() - 1]; }
		Base * back() { return (*this)[elem_inds.size() - 1]; }


		template <typename Derived>
			requires std::is_base_of_v<Base, Derived> && std::is_copy_constructible_v<Derived>
		void push_back(const Derived &derived) {
			reserve_for<Derived>();
			new (this->back()) Derived(derived);
		}

		template <typename Derived>
			requires std::is_base_of_v<Base, Derived> && std::is_move_constructible_v<Derived>
		void push_back(Derived &&derived) {
			reserve_for<Derived>();
			new (this->back()) Derived(std::move(derived));
		}

		template <typename Derived, typename ... Args>
			requires std::is_base_of_v<Base, Derived> && std::is_constructible_v<Derived, Args...>
		void emplace_back(Args && ... args)
		{
			reserve_for<Derived>();
			new (this->back()) Derived(std::forward<Args>(args)...);
		}

		
	};
}