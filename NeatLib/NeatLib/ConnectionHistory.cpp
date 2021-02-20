#include "ConnectionGene.hpp"
#include "Node.hpp"
#include "ConnectionHistory.hpp"
#include "Genome.hpp"
#include <vector>
#include <algorithm>

namespace NEAT {
	ConnectionHistory::ConnectionHistory() : m_fromNode{ 0 }, m_toNode{ 0 }, m_innovationNumber{ 0 },
		m_innovationNumbers{} {}

	ConnectionHistory::ConnectionHistory(int _from_node, int _to_node, int _innovation_number,
		std::vector<int>& _innovation_numbers) : m_fromNode{ _from_node }, m_toNode{ _to_node },
		m_innovationNumber{ _innovation_number }, m_innovationNumbers{ _innovation_numbers } {}

	bool ConnectionHistory::matches(Genome& _genome, const Node& _from_node, const Node& _to_node) {
		if (_genome.m_genes.size() == m_innovationNumbers.size()) {
			if (_from_node.m_number == m_fromNode && _to_node.m_number == m_toNode) {

				for (const ConnectionGene& gene : _genome.m_genes) {

					if (!std::count(m_innovationNumbers.begin(),
						m_innovationNumbers.end(), gene.m_innovationNo)) {
						return false;
					}

				}

				return true;
			}
		}
		return false;
	}
}