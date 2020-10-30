#pragma once
#include <vector>

namespace NEAT {
	struct Genome;
	struct Node;
	struct ConnectionHistory {
		int fromNode, toNode, innovationNumber;
		std::vector<int> innovationNumbers;

		ConnectionHistory() = default;
		ConnectionHistory(int, int, int, std::vector<int>&);

		bool Matches(Genome&, const Node&, const Node&);
	};
}