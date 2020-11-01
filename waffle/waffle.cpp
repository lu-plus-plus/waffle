
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
		auto d = c * c;

		d.grad() = 10.f;
		d.backward();
		
		std::cout << "backward:" << std::endl;
		std::cout << "a' = " << a.grad() << std::endl;
		std::cout << "b' = " << b.grad() << std::endl;
		std::cout << "c' = " << c.grad() << std::endl;
		std::cout << std::endl;
	}

	{
		waffle::adjoint<float> a(2.f);
		waffle::adjoint<float> b(3.f);
		auto c = a * b;
		auto d = c * c;

		a.grad() = 10.f;
		a.forward();

		std::cout << "forward:" << std::endl;
		std::cout << "b' = " << b.grad() << std::endl;
		std::cout << "c' = " << c.grad() << std::endl;
		std::cout << "d' = " << d.grad() << std::endl;
		std::cout << std::endl;
	}
	
}


