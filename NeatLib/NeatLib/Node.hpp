#pragma once
#include <vector>

namespace NEAT {
	struct ConnectionGene;

	struct Node {
		int number;
		int layer;
		float inputSum;
		float outputValue;
		float (*activation) (const float&);
		std::vector<ConnectionGene> output_connections = std::vector<ConnectionGene>();

		Node() = default;
		Node(int ino, float (*func)(const float&));

		static Node Clone(const Node&);

		bool IsConnectedTo(const Node& other);
		void Engage();
	};
}