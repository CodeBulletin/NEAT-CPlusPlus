#include "ConnectionGene.hpp"
#include "Node.hpp"
#include "extras.hpp"
#include <vector>

namespace NEAT {
	ConnectionGene::ConnectionGene() : m_fromNode{ 0 }, m_fromNodeNumber{ 0 }, m_toNode{0},m_toNodeNumber{ 0 },
		m_weight{ 0.0f }, m_innovationNo{ 0 }, m_weightMutationRatio{ 0 }, m_enabled{ true } {}

	ConnectionGene::ConnectionGene(int _from_node , int _from_node_number, int _to_node, int _to_node_number,
		float _weight, int _innovation_number, float _weight_mutation_number) :
		m_fromNode{ _from_node }, m_fromNodeNumber{ _from_node_number }, m_toNode{ _to_node },
		m_toNodeNumber{ _to_node }, m_weight{ _weight }, m_innovationNo{ _innovation_number },
		m_weightMutationRatio{ _weight_mutation_number }, m_enabled{ true } {}

	ConnectionGene ConnectionGene::clone(const ConnectionGene& _gene) {
		return ConnectionGene(_gene.m_fromNode, _gene.m_fromNodeNumber, _gene.m_toNode, _gene.m_toNodeNumber,
			_gene.m_weight, _gene.m_innovationNo, _gene.m_weightMutationRatio);
	}

	void ConnectionGene::mutateWeight() {
		float r = Random::Random();
		if (m_weightMutationRatio < r) {
			m_weight = Random::RandomRange();
		}
		else {
			m_weight += Random::RandomRange() / 50.0f;
			if (m_weight >= 1.0f) m_weight = 1;
			if (m_weight <= -1.0f) m_weight = -1;
		}
	}
}