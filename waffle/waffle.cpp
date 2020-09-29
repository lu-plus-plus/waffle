
#include <iostream>
#include "array.hpp"

template <template <typename> typename Container = waffle::scalar>
struct rgb {
	Container<float> r, g, b;
};


template <typename T>
struct Test {
	T val;
	Test(const T &v) : val(v) {}
};


template <typename T>
using vec4 = waffle::ndarray<T, 4>;

int main(void)
{
	waffle::ndarray<float, 13, 2> A(1.f);
	waffle::ndarray<float, 13, 2> B(2.f);

	auto C1 = A + B;
	auto C2 = B + A + 2.f;

	std::cout << C1 << std::endl;
	std::cout << C2 << std::endl;

	auto D = C2 - C1;
	std::cout << D << std::endl;

	auto E = D * waffle::ndarray<float, 13>(3.f);
	std::cout << E << std::endl;

	auto F = waffle::ndarray<float, 13, 2, 2>(6.f) / E;
	std::cout << F << std::endl;

	waffle::f32<> single;
	std::cout << single << std::endl;

	//waffle::f32<vec4> testvec(2.5f);
	//std::cout << testvec[0];

	Test test = 1.f;
}