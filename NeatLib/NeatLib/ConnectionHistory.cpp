#include "ConnectionGene.hpp"
#include "Node.hpp"
#include "ConnectionHistory.hpp"
#include "Genome.hpp"
#include <vector>
#include <algorithm>

namespace NEAT {
	ConnectionHistory::ConnectionHistory(int frm, int to, int in_no, std::vector<int>& in_nos):
		fromNode(frm), toNode(to), innovationNumber(in_no), innovationNumbers(in_nos) {}

	bool ConnectionHistory::Matches(Genome& genome, const Node& from, const Node& to) {
		if (genome.genes.size() == innovationNumbers.size()) {
			if (from.number == fromNode && to.number == toNode) {
				for (const ConnectionGene& gene : genome.genes)
					if (!std::count(innovationNumbers.begin(), innovationNumbers.end(), gene.innovationNo))
						return false;
				return true;
			}
		}
		return false;
	}
}