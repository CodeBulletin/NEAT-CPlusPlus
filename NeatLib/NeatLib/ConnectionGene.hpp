#pragma once

namespace NEAT {
	struct Node;
	struct ConnectionGene {
		Node *toNode, *fromNode;
		bool enabled;
		float weight, weight_mutation_ratio;
		int innovationNo;

		ConnectionGene() = default;
		ConnectionGene(Node&, Node&, float, int, float);

		static ConnectionGene Clone(const ConnectionGene&, Node&, Node&);

		void MutateWeight();
	};
}