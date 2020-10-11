
#include <iostream>

#include "array.hpp"
#include "dynamic.hpp"
#include "vector.hpp"



template <typename T>
using arraytest = waffle::ndarray<T, 2, 3>;

int main(void)
{
	waffle::ndarray<float, 13, 2, 2> a(5.f);
	waffle::ndarray<float, 13> b(1.f);
	auto c = b / a;
	std::cout << c << std::endl;

	float f1 = 1.f;
	float f2 = 2.f;
	waffle::Vector3f<arraytest> dv(1.f, f1, std::move(f2));
}