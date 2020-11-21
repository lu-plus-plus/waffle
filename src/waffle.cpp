
#include <iostream>

#include "waffle/arithmetics.hpp"
using namespace waffle::literals;

#include "waffle/array.hpp"

#include "waffle/derived_vector.hpp"

#include "waffle/adjoint.hpp"



int main(void)
{

	{
		waffle::ndarray<float, 13, 2> A(5.f);
		waffle::ndarray<float, 13> Ainit(5.f);
		std::cout << waffle::sum(A, Ainit) << std::endl;

		waffle::ndarray<float, 13> B(10.f);
		float b = 10.f;

		std::cout << std::boolalpha << (B == b) << std::endl;

		auto C = waffle::sum(B, 5.f);
		std::cout << C << std::endl;
		
		std::cout << std::endl;
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

	{
		waffle::array<float, 2> A{ 1,2 };
		waffle::array<float, 2> B = { 3,4 };
		waffle::array<float, 2> C({ 5,6 });
		std::cout << A << std::endl
			<< B << std::endl
			<< C << std::endl;
	}

}


