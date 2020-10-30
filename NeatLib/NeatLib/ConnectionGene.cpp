#include "ConnectionGene.hpp"
#include "Node.hpp"
#include "extras.hpp"
#include <vector>

namespace NEAT {
	ConnectionGene::ConnectionGene(Node& _fromNode, Node& _toNode, float Weight, int ino, float wmr) :
		fromNode(&_fromNode), toNode(&_toNode), weight(Weight), innovationNo(ino), weight_mutation_ratio(wmr), enabled(true) {}

	ConnectionGene ConnectionGene::Clone(const ConnectionGene& gene, Node& fromNode, Node& toNode) {
		return ConnectionGene(fromNode, toNode, gene.weight, gene.innovationNo, gene.weight_mutation_ratio);
	}

	void ConnectionGene::MutateWeight() {
		float r = Random::random();
		if (weight_mutation_ratio < r) {
			weight = Random::randomRange();
		}
		else {
			weight += Random::randomRange() / 50.0f;
			if (weight >= 1.0f) weight = 1;
			if (weight <= -1.0f) weight = -1;
		}
	}
}