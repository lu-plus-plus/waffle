
#include <iostream>
#include "array.hpp"



template <typename T, template <typename> typename Op>
T eval(const T &a, const T &b)
{
	return Op<T>()(a, b);
}

int main(void)
{
	//waffle::ndarray<float, 3> x(1.f);
	//waffle::ndarray<float, 4, 3> y(2.f);

	//auto z1 = x + y;
	//std::cout << z1;

	//auto z2 = y + x;
	//std::cout << z2;

	waffle::ndarray<double, 2, 3, 4> A(1.0);
	A += 2.0;
	A -= 4.0;
	std::cout << A;
}