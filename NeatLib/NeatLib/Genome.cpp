#include "Genome.hpp"
#include "Node.hpp"
#include "ConnectionGene.hpp"
#include "ConnectionHistory.hpp"
#include "extras.hpp"

#include <string>
#include <iostream>

namespace NEAT {
	Genome::Genome() : m_settings{}, m_nextNode{ 0 }, m_layers{ 0 }, m_biasNode{ 0 }, m_inputNodes{ 0 },
		m_outputNodes{ 0 }, m_network{}, m_nodes{}, m_genes{} {}

	Genome::Genome(const std::unordered_map<std::string, float>& _settings):
		m_settings{ _settings }, m_nextNode{ 0 }, m_layers{ 2 }, m_biasNode{ 0 }, m_network{}, m_genes{} {

		m_inputNodes = (int)m_settings["Genome Inputs"];
		m_outputNodes = (int)m_settings["Genome Outputs"];

		for (int i = 0; i < m_inputNodes; i++) {
			m_nodes.emplace_back(i, Activations::linear);
			m_nodes[m_nextNode].m_layer = 0;
			++m_nextNode;
		}

		for (int i = 0; i < m_outputNodes; i++) {
			m_nodes.emplace_back(i+m_inputNodes, Activations::Activations[(int)m_settings["Output Activation"]]);
			m_nodes[m_nextNode].m_layer = 1;
			++m_nextNode;
		}

		m_nodes.emplace_back(m_nextNode, Activations::linear);
		m_biasNode = m_nextNode;
		++m_nextNode;
	}

	Genome Genome::clone(Genome& _genome) {
		Genome new_genome;
		new_genome.m_settings = _genome.m_settings;
		new_genome.m_inputNodes = _genome.m_inputNodes;
		new_genome.m_outputNodes = _genome.m_outputNodes;
		new_genome.m_layers = _genome.m_layers;
		new_genome.m_nextNode = _genome.m_nextNode;
		new_genome.m_biasNode = _genome.m_biasNode;

		for (Node& node : _genome.m_nodes) {
			new_genome.m_nodes.push_back(Node::clone(node));
		}
		for (const ConnectionGene& gene : _genome.m_genes) {
			new_genome.m_genes.push_back(ConnectionGene::clone(gene));
		}

		new_genome.connectNodes();
		return new_genome;
	}

	void Genome::connectNodes() {
		for (int i = 0; i < m_nextNode; i++) {
			m_nodes[i].m_outputConnections.clear();
		}

		for (ConnectionGene& gene : m_genes) {
			std::cout << m_nodes.size() << " " << gene.m_fromNodeNumber << " " << gene.m_toNodeNumber << std::endl;
			m_nodes[gene.m_fromNodeNumber].m_outputConnections.push_back(gene);
		}
	}

	void Genome::generateNetwork() {
		connectNodes();

		m_network = std::vector<int>();

		for (int i = 0; i < m_layers; i++) {
			for (int j = 0; j < m_nextNode; j++) {
				if (m_nodes[j].m_layer == i) {
					m_network.push_back(j);
				}
			}
		}
	}

	std::vector<float> Genome::feedForward(std::vector<float>& _inputs) {
		for (int i = 0; i < m_inputNodes; i++) {
			m_nodes[i].m_outputValue = _inputs[i];
		}
		m_nodes[m_biasNode].m_outputValue = 1;

		for (int node_number : m_network) {
			m_nodes[node_number].engage(m_nodes);
		}

		std::vector<float> output;
		for (int i = 0; i < m_outputNodes; i++) {
			output.push_back(m_nodes[m_inputNodes + i].m_outputValue);
		}

		for (Node& node : m_nodes) {
			node.m_inputSum = 0;
		}
		return output;
	}

	int Genome::getInnovationNumber(std::vector<ConnectionHistory>& _innovation_histories,
		const Node& _from_node, const Node& _to_node) {
		bool is_new = true;

		for (ConnectionHistory& history : _innovation_histories) {
			if (history.matches(*this, _from_node, _to_node)) {
				is_new = false;
				return history.m_innovationNumber;
			}
		}

		if (is_new) {
			int connection_innovation_number = (int)m_settings["_nextConnectionNo"];
			std::vector<int> innovation_numbers;

			for (ConnectionGene& gene : m_genes) {
				innovation_numbers.push_back(gene.m_innovationNo);
			}

			_innovation_histories.emplace_back(_from_node.m_number, _to_node.m_number,
				connection_innovation_number, innovation_numbers);

			++m_settings["_nextConnectionNo"];
			return connection_innovation_number;
		}
		return 0;
	}

	bool Genome::isFullyConnected() {
		int max_connections = 0;
		std::vector<int> nodes_in_layer(m_layers, 0);

		for (Node& node : m_nodes) {
			++nodes_in_layer[node.m_layer];
		}

		for (int i = 0; i < m_layers - 1; i++) {
			int nodes_in_front = 0;
			for (int j = i + 1; j < m_layers; j++) {
				nodes_in_front += nodes_in_layer[j];
			}
			max_connections += nodes_in_layer[i] * nodes_in_front;
		}

		if (max_connections == m_genes.size()) return true;
		else return false;
	}
	
	bool Genome::randomConnectionsAreBad(int node1, int node2) {
		return (m_nodes[node1].m_layer == m_nodes[node2].m_layer) || m_nodes[node1].isConnectedTo(m_nodes[node2]);
	}

	void Genome::addConnection(std::vector<ConnectionHistory>& _innovation_histories) {
		if (isFullyConnected()) return;

		int random_node_1 = (int)std::floorf(Random::Random() * m_nextNode);
		int random_node_2 = (int)std::floorf(Random::Random() * m_nextNode);
		while (randomConnectionsAreBad(random_node_1, random_node_2)) {
			random_node_1 = (int)std::floorf(Random::Random() * m_nextNode);
			random_node_2 = (int)std::floorf(Random::Random() * m_nextNode);
		}

		if (m_nodes[random_node_1].m_layer > m_nodes[random_node_2].m_layer) {
			std::swap(random_node_1, random_node_2);
		}

		int connection_innovation_number = getInnovationNumber(_innovation_histories,
			m_nodes[random_node_1], m_nodes[random_node_2]);

		m_genes.emplace_back(m_nodes[random_node_1].m_number, random_node_1,
			m_nodes[random_node_2].m_number, random_node_2, Random::RandomRange(),
			connection_innovation_number, m_settings["Weight Mutation Ratio"]);

		connectNodes();
	}

	void Genome::addNode(std::vector<ConnectionHistory>& _innovation_histories) {
		if (m_genes.empty()) {
			addConnection(_innovation_histories);
			return;
		}

		int random_connection_number = (int)std::floorf(Random::Random() * m_genes.size());
		while (m_genes[random_connection_number].m_fromNode == m_biasNode && m_genes.size() != 1) {
			random_connection_number = (int)std::floorf(Random::Random() * m_genes.size());
		}
		m_genes[random_connection_number].m_enabled = false;

		int new_node_no = m_nextNode;
		m_nodes.emplace_back(new_node_no, Activations::Activations[(int)m_settings["Hidden Activation"]]);
		Node& node = m_nodes[new_node_no];
		m_nextNode++;

		float weight_mutation_ratio = m_settings["Weight Mutation Ratio"];


		int connection_innovation_number = getInnovationNumber(_innovation_histories,
			m_nodes[m_genes[random_connection_number].m_fromNode], node);

		m_genes.emplace_back(m_genes[random_connection_number].m_fromNode,
			m_genes[random_connection_number].m_fromNodeNumber, new_node_no, node.m_number,
			1, connection_innovation_number, weight_mutation_ratio);


		connection_innovation_number = getInnovationNumber(_innovation_histories,
			node, m_nodes[m_genes[random_connection_number].m_toNode]);

		m_genes.emplace_back(new_node_no, node.m_number, m_genes[random_connection_number].m_toNode,
			m_genes[random_connection_number].m_toNodeNumber, m_genes[random_connection_number].m_weight,
			connection_innovation_number, weight_mutation_ratio);

		node.m_layer = m_nodes[m_genes[random_connection_number].m_fromNode].m_layer + 1;


		connection_innovation_number = getInnovationNumber(_innovation_histories, m_nodes[m_biasNode], node);
		
		m_genes.emplace_back(m_biasNode, m_nodes[m_biasNode].m_number, new_node_no, node.m_number, 0,
			connection_innovation_number, weight_mutation_ratio);


		if (node.m_layer == m_nodes[m_genes[random_connection_number].m_toNode].m_layer) {
			for (Node& current_node : m_nodes) {
				if (current_node.m_layer >= node.m_layer && current_node.m_number != new_node_no) {
					current_node.m_layer += 1;
				}
			}
			m_layers += 1;
		}
		connectNodes();
	}

	void Genome::fullyConnect(std::vector<ConnectionHistory>& _innovation_histories) {
		for (int i = 0; i < (m_inputNodes + 1) * m_outputNodes; i++) {
			addConnection(_innovation_histories);
		}
		connectNodes();
	}

	void Genome::partialConnect(std::vector<ConnectionHistory>& _innovation_histories) {
		for (int i = 0; i < ((m_inputNodes + 1) * m_outputNodes) / 2; i++) {
			addConnection(_innovation_histories);
		}
		connectNodes();
	}

	void Genome::minimumConnect(std::vector<ConnectionHistory>& _innovation_histories) {
		for (int i = 0; i < 3; i++) {
			addConnection(_innovation_histories);
		}
		connectNodes();
	}

	void Genome::mutate(std::vector<ConnectionHistory>& _innovation_histories) {
		if (Random::Random() < m_settings["Weight Mutate Percent"]) {
			for (ConnectionGene& gene : m_genes) {
				if (Random::Random() < m_settings["Weight Mutation Percent"]) {
					gene.mutateWeight();
				}
				if (Random::Random() < m_settings["Connection Toggle Percent"]) {
					gene.m_enabled = !gene.m_enabled;
				}
			}
		}

		if (Random::Random() < m_settings["Connection Mutation Percent"]) {
			addConnection(_innovation_histories);
		}

		if (Random::Random() < m_settings["Node Mutation Percent"]) {
			addNode(_innovation_histories);
		}
	}

	Genome Genome::crossover(const Genome& _other_parrent) {

		Genome child;
		child.m_inputNodes = m_inputNodes;
		child.m_outputNodes = m_outputNodes;
		child.m_settings = m_settings;
		child.m_layers = m_layers;
		child.m_nextNode = m_nextNode;
		child.m_biasNode = m_biasNode;

		std::vector<ConnectionGene> child_gene;
		std::vector<bool> is_enabled;

		for (size_t i = 0; i < m_genes.size(); i++) {
			bool set_enabled = true;

			int other_parrent_gene = matchingGene(_other_parrent, m_genes[i].m_innovationNo);

			if (other_parrent_gene != -1) {
				if (!(m_genes[i].m_enabled || _other_parrent.m_genes[i].m_enabled)) {
					if (Random::Random() < m_settings["Enable Percent"])
						set_enabled = false;
				}

				if (Random::Random() < m_settings["Parent Gene Percent"]) {
					child_gene.push_back(m_genes[i]);
				}
				else {
					child_gene.push_back(_other_parrent.m_genes[other_parrent_gene]);
				}
			}
			else {
				child_gene.push_back(m_genes[i]);
				set_enabled = m_genes[i].m_enabled;
			}

			is_enabled.push_back(set_enabled);
		}

		for (Node& node : m_nodes) {
			child.m_nodes.push_back(node);
		}

		for (size_t i = 0; i < child_gene.size(); i++) {
			child.m_genes.push_back(ConnectionGene::clone(child_gene[i]));
			child.m_genes[i].m_enabled = is_enabled[i];
		}

		child.connectNodes();
		return child;
	}

	int Genome::matchingGene(const Genome& _parent, int _innovation_number) {
		for (size_t i = 0; i < _parent.m_genes.size(); i++) {
			if (_parent.m_genes[i].m_innovationNo == _innovation_number) {
				return i;
			}
		}
		return -1;
	}
}