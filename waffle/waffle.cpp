
#include <iostream>

#include "array.hpp"
#include "dynamic.hpp"
#include "vector.hpp"



int main(void)
{
	waffle::ndarray<float, 13, 2> A(5.f);
	waffle::ndarray<float, 13> Ainit(5.f);
	std::cout << waffle::sum(A, Ainit) << std::endl;

	waffle::ndarray<waffle::scalar<float>, 13, 2> B(waffle::scalar<float>(10.f));
	std::cout << waffle::sum(B, waffle::scalar<float>(5.f)) << std::endl;
}