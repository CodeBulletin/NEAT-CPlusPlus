#pragma once
#include <vector>

namespace NEAT {
	struct ConnectionGene;

	struct Node {
		int m_number;
		int m_layer;
		float m_inputSum;
		float m_outputValue;
		float (*m_activation) (const float&);
		std::vector<ConnectionGene> m_outputConnections = std::vector<ConnectionGene>();

		Node();
		Node(int ino, float (*func)(const float&));

		static Node clone(const Node&);

		bool isConnectedTo(const Node& other);
		void engage(std::vector<Node>&);
	};
}