#pragma once

#include "arithmetics.hpp"
#include "derived_vector.hpp"

#include <vector>
#include <queue>
#include <unordered_set>



namespace waffle
{

	struct comp_graph {

		/*
			interface: arbitrary edge _without_type_ in computation graph
		*/
		struct edge {

			// from_idx refers to an instance in node::records
			usize from_idx;

			// to_idx refers to an instance in node::records
			usize to_idx;

			inline static derived_vector<edge> records;


			/* ctor */

			edge(usize _from_idx_, usize _to_idx_) : from_idx(_from_idx_), to_idx(_to_idx_) {}


			/* optimization */

			virtual void forward() = 0;
			virtual void backward() = 0;

		};



		/*
			interface: arbitrary node _without_type_ in computation graph
		*/
		struct node {

			inline static derived_vector<node> records;

			// in_edge_inds refers to instances among edge::records
			std::vector<usize> in_edge_inds;

			// out_edge_inds refers to instances among edge::records
			std::vector<usize> out_edge_inds;


			/* ctor */

			node() = default;


			/* optimization */

			void forward() {
				for (usize in_edge_idx : in_edge_inds) {
					edge::records[in_edge_idx]->forward();
				}
			}

			void backward() {
				for (usize out_edge_idx : out_edge_inds) {
					edge::records[out_edge_idx]->backward();
				}
			}

		};



		template <typename T>
		struct tape
		{
			inline static std::vector<T> records;

			static usize push_record(const T &rec) {
				tape<T>::records.push_back(rec);
				return tape<T>::records.size() - 1;
			}

			static usize push_record(T &&rec) {
				tape<T>::records.push_back(std::move(rec));
				return tape<T>::records.size() - 1;
			}

			template <typename ... Args>
				requires std::is_constructible_v<T, Args...>
			static usize emplace_record(Args && ... args) {
				tape<T>::records.emplace_back(std::forward<Args>(args)...);
				return tape<T>::records.size() - 1;
			}
		};



		template <typename T>
		struct node_impl : public node
		{
			// grad_idx refers to an instance in tape<T>::records
			usize grad_idx;
			
			const T & grad() const { return tape<T>::records[grad_idx]; }
			T & grad() { return tape<T>::records[grad_idx]; }


			/* ctor */

			node_impl() : node(), grad_idx(tape<T>::emplace_record(0.f)) {}

			static usize create() {
				node::records.emplace_back<node_impl<T>>();
				return node::records.size() - 1;
			}

		};



		template <typename From, typename To>
		struct edge_impl : edge
		{
			// grad_idx refers to an instance in tape<To>::records
			usize grad_idx;

			const To & grad() const { return tape<To>::records[grad_idx]; }
			To & grad() { return tape<To>::records[grad_idx]; }


			/* ctor */

			edge_impl(usize _from_idx_, usize _to_idx_, const To &grad) :
				edge(_from_idx_, _to_idx_), grad_idx(tape<To>::emplace_record(grad)) {}

			static usize link(usize from_idx, usize to_idx, const To &grad) {
				edge::records.emplace_back<edge_impl<From, To>>(from_idx, to_idx, grad);
				const usize this_ind = edge::records.size() - 1;
				static_cast<node_impl<From> *>(node::records[from_idx])->out_edge_inds.push_back(this_ind);
				static_cast<node_impl<To> *>(node::records[to_idx])->in_edge_inds.push_back(this_ind);
				return this_ind;
			}

			static usize link(usize from_idx, usize to_idx, To &&grad) {
				edge::records.emplace_back<edge_impl<From, To>>(from_idx, to_idx, std::move(grad));
				const usize this_ind = edge::records.size() - 1;
				static_cast<node_impl<From> *>(node::records[from_idx])->out_edge_inds.push_back(this_ind);
				static_cast<node_impl<To> *>(node::records[to_idx])->in_edge_inds.push_back(this_ind);
				return this_ind;
			}


			/* optimization */

			virtual void forward() override {
				auto from = static_cast<node_impl<From> &>(*node::records[from_idx]);
				auto to = static_cast<node_impl<To> &>(*node::records[to_idx]);

				from.grad() += to.grad() * this->grad();

				if (!traversed.contains(from_idx)) {
					schedule.emplace(from_idx);
					traversed.emplace(from_idx);
				}
			}

			virtual void backward() override {
				auto from = static_cast<node_impl<From> &>(*node::records[from_idx]);
				auto to = static_cast<node_impl<To> &>(*node::records[to_idx]);

				if constexpr (std::is_same_v<From, To>)
					to.grad() += from.grad() * this->grad();
				else
					to.grad() += sum(from.grad(), To(0.f)) * this->grad();

				if (!traversed.contains(to_idx)) {
					schedule.emplace(to_idx);
					traversed.emplace(to_idx);
				}
			}

		};



		inline static std::queue<usize> schedule;
		inline static std::unordered_set<usize> traversed;

		static void backward() {
			while (!schedule.empty()) {
				usize node_idx = schedule.front();
				schedule.pop();

				node::records[node_idx]->backward();
			}

			traversed.clear();
		}

		static void forward() {
			while (!schedule.empty()) {
				usize node_idx = schedule.front();
				schedule.pop();

				node::records[node_idx]->forward();
			}

			traversed.clear();
		}

	};



	template <typename T>
	struct adjoint
	{
		T value;
		
		usize node_idx;
		const T & grad() const {
			assert(node_idx < comp_graph::node::records.size());
			return static_cast<comp_graph::node_impl<T> *>(comp_graph::node::records[node_idx])->grad();
		}
		T & grad() {
			return const_cast<T &>(const_cast<const adjoint *>(this)->grad());
		}

		adjoint() : value(), node_idx(comp_graph::node_impl<T>::create()) {}
		adjoint(const T &v) : value(v), node_idx(comp_graph::node_impl<T>::create()) {}

		void backward() {
			comp_graph::schedule.emplace(node_idx);
			comp_graph::traversed.emplace(node_idx);
			comp_graph::backward();
		}

		void forward() {
			comp_graph::schedule.emplace(node_idx);
			comp_graph::traversed.emplace(node_idx);
			comp_graph::forward();
		}
	};

	template <typename T>
	adjoint<T> operator*(const adjoint<T> &a, const adjoint<T> &b)
	{
		adjoint<T> result(a.value * b.value);
		comp_graph::edge_impl<T, T>::link(result.node_idx, b.node_idx, a.value);
		comp_graph::edge_impl<T, T>::link(result.node_idx, a.node_idx, b.value);
		return result;
	}

}