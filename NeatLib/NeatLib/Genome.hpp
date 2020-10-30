#pragma once
#include <vector>
#include <unordered_map>

namespace NEAT {
	struct Node;
	struct ConnectionHistory;
	struct ConnectionGene;
	struct Genome {
		int input_nodes, output_nodes;
		std::vector<Node> nodes;
		std::vector<Node*> network;
		std::vector<ConnectionGene> genes;
		std::unordered_map<std::string, float> settings;
		int next_node, layers, bias_node;

		Genome() = default;
		Genome(const std::unordered_map<std::string, float>&);

		static Genome Clone(const Genome&);

		void ConnectNodes();
		void GenerateNetwork();
		std::vector<float> FeedForward(std::vector<float>&);

		Node& GetNode(int);
		int GetInnovationNumber(std::vector<ConnectionHistory>&, const Node&, const Node&);
		bool IsFullyConnected();
		bool RandomConnectionsAreBad(int, int);
		void AddConnection(std::vector<ConnectionHistory>&);
		void AddNode(std::vector<ConnectionHistory>&);
		void FullyConnect(std::vector<ConnectionHistory>&);
		void PartialConnect(std::vector<ConnectionHistory>&);
		void MinimumConnect(std::vector<ConnectionHistory>&);
		void Mutate(std::vector<ConnectionHistory>&);
		
		Genome Crossover(Genome&);

		static int MatchingGene(Genome&, int);
	};
}