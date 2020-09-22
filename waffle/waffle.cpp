
#include <iostream>
#include "array.hpp"

int main(void)
{
	waffle::ndarray<float, 3, 4> x(1.f);
	waffle::ndarray<float, 3, 4, 5> y(2.f);
	
	waffle::Eval<decltype(x), decltype(y)> eval(x, y);
	
	decltype(y) z;
	eval(z);

	//for (waffle::isize i = 0; i < 4; ++i) {
	//	for (waffle::isize j = 0; j < 3; ++j) {
	//		std::cout << z[i][j] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	std::cout << z;
}