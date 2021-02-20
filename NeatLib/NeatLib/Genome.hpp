#pragma once
#include <vector>
#include <unordered_map>

namespace NEAT {
	struct Node;
	struct ConnectionHistory;
	struct ConnectionGene;
	struct Genome {
		int m_inputNodes, m_outputNodes;
		std::vector<Node> m_nodes;
		std::vector<int> m_network;
		std::vector<ConnectionGene> m_genes;
		std::unordered_map<std::string, float> m_settings;
		int m_nextNode, m_layers, m_biasNode;

		Genome();
		Genome(const std::unordered_map<std::string, float>&);

		static Genome clone(Genome&);

		void connectNodes();
		void generateNetwork();
		std::vector<float> feedForward(std::vector<float>&);

		int getInnovationNumber(std::vector<ConnectionHistory>&, const Node&, const Node&);
		bool isFullyConnected();
		bool randomConnectionsAreBad(int, int);
		void addConnection(std::vector<ConnectionHistory>&);
		void addNode(std::vector<ConnectionHistory>&);
		void fullyConnect(std::vector<ConnectionHistory>&);
		void partialConnect(std::vector<ConnectionHistory>&);
		void minimumConnect(std::vector<ConnectionHistory>&);
		void mutate(std::vector<ConnectionHistory>&);
		
		Genome crossover(const Genome&);

		static int matchingGene(const Genome&, int);
	};
}