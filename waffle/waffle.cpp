
#include <iostream>

#include "array.hpp"

#include "derived_vector.hpp"

#include "adjoint.hpp"



int main(void)
{
	{
		waffle::ndarray<float, 13, 2> A(5.f);
		waffle::ndarray<float, 13> Ainit(5.f);
		std::cout << waffle::sum(A, Ainit) << std::endl;

		waffle::ndarray<waffle::scalar<float>, 13, 2> B(waffle::scalar<float>(10.f));
		std::cout << waffle::sum(B, waffle::scalar<float>(5.f)) << std::endl;
	}

	{
		waffle::adjoint<float> a(2.f);
		waffle::adjoint<float> b(3.f);
		auto c = a * b;
		c.grad() = 4.f;

		c.backward();
		
		std::cout << "grad A = " << a.grad() << std::endl;
		std::cout << "grad B = " << b.grad() << std::endl;
	}
	
}


