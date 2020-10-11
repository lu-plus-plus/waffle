
#include <iostream>

#include "array.hpp"
#include "dynamic.hpp"
#include "vector.hpp"



int main(void)
{
	waffle::ndarray<float, 13, 2, 2> a(5.f);
	waffle::ndarray<float, 13> b(1.f);
	auto c = b / a;
	std::cout << c << std::endl;

	waffle::ndarray<float, 13> d(0.f);
	auto my_plus = [] <typename T> (T & dest, const T & src) { dest += src; };
	waffle::reduce(d, c, my_plus);
	std::cout << d << std::endl;

	std::cout << waffle::reduce(0.f, d, my_plus) << std::endl;
}