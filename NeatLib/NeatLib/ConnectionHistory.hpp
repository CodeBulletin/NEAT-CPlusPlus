#pragma once
#include <vector>

namespace NEAT {
	struct Genome;
	struct Node;
	struct ConnectionHistory {
		int m_fromNode, m_toNode, m_innovationNumber;
		std::vector<int> m_innovationNumbers;

		ConnectionHistory();
		ConnectionHistory(int, int, int, std::vector<int>&);

		bool matches(Genome&, const Node&, const Node&);
	};
}