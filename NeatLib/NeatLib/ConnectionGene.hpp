#pragma once

namespace NEAT {
	struct Node;
	struct ConnectionGene {
		int m_fromNode, m_fromNodeNumber;
		int m_toNode, m_toNodeNumber;
		bool m_enabled;
		float m_weight, m_weightMutationRatio;
		int m_innovationNo;

		ConnectionGene();
		ConnectionGene(int, int, int, int, float, int, float);

		static ConnectionGene clone(const ConnectionGene&);

		void mutateWeight();
	};
}