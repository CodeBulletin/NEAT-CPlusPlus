#include "Genome.hpp"
#include "Node.hpp"
#include "ConnectionGene.hpp"
#include "ConnectionHistory.hpp"
#include "extras.hpp"

#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

namespace NEAT {
	Genome::Genome(const std::unordered_map<std::string, float>& _settings): settings(_settings), next_node(0), layers(2), bias_node(0) {
		input_nodes = settings["Genome Inputs"];
		output_nodes = settings["Genome Outputs"];
		for (auto& it : Defaults::neat_defaults)
			if (settings.find(it.first) == settings.end())
				settings[it.first] = it.second;
		for (int i = 0; i < input_nodes; i++) {
			nodes.emplace_back(i, Activation_Functions::linear);
			nodes[next_node].layer = 0;
			++next_node;
		}
		for (int i = 0; i < output_nodes; i++) {
			nodes.emplace_back(i+input_nodes, Activation_Functions::Activation_functions[settings["Output Activation"]]);
			nodes[next_node].layer = 1;
			++next_node;
		}
		nodes.emplace_back(next_node, Activation_Functions::linear);
		bias_node = next_node;
		++next_node;
	}

	Genome Genome::Clone(const Genome& genome) {
		Genome newgenome;
		newgenome.settings = genome.settings;
		newgenome.input_nodes = genome.input_nodes;
		newgenome.output_nodes = genome.output_nodes;
		newgenome.layers = genome.layers;
		newgenome.next_node = genome.next_node;
		newgenome.bias_node = genome.bias_node;
		for (const Node& node : genome.nodes) {
			newgenome.nodes.push_back(Node::Clone(node));
		}
		for (const ConnectionGene& gene : genome.genes) {
			newgenome.genes.push_back(ConnectionGene::Clone(gene, newgenome.GetNode(gene.fromNode->number), newgenome.GetNode(gene.toNode->number)));
		}
		newgenome.ConnectNodes();
		return newgenome;
	}

	void Genome::ConnectNodes() {
		for (Node& node : nodes) node.output_connections.clear();
		for (ConnectionGene& gene : genes) {
			gene.fromNode->output_connections.push_back(gene);
		}
	}

	void Genome::GenerateNetwork() {
		ConnectNodes();
		network = std::vector<Node*>();
		for (int i = 0; i < layers; i++) for (Node& node : nodes) if (node.layer == i) network.push_back(&node);
	}

	std::vector<float> Genome::FeedForward(std::vector<float>& inputs) {
		for (int i = 0; i < input_nodes; i++) nodes[i].outputValue = inputs[i];
		nodes[bias_node].outputValue = 1;
		for (Node* node : network) node->Engage();
		std::vector<float> output;
		for (int i = 0; i < output_nodes; i++) output.push_back(nodes[input_nodes + i].outputValue);
		for (Node& node : nodes) node.inputSum = 0;
		return output;
	}

	Node& Genome::GetNode(int node_number) {
		for (Node& node : nodes) if (node.number == node_number) return node;
	}

	int Genome::GetInnovationNumber(std::vector<ConnectionHistory>& innovationHistories, const Node& _fromNode, const Node& _toNode) {
		bool isNew = true;
		for (ConnectionHistory& history : innovationHistories) {
			if (history.Matches(*this, _fromNode, _toNode)) {
				isNew = false;
				return history.innovationNumber;
			}
		}
		if (isNew) {
			int connection_innovation_number = settings["_nextConnectionNo"];
			std::vector<int> innovation_numbers;
			for (ConnectionGene& gene : genes) {
				innovation_numbers.push_back(gene.innovationNo);
			}
			innovationHistories.emplace_back(_fromNode.number, _toNode.number,
				connection_innovation_number, innovation_numbers);
			++settings["_nextConnectionNo"];
			return connection_innovation_number;
		}
	}

	bool Genome::IsFullyConnected() {
		int maxConnections = 0;
		std::vector<int> nodes_in_layer(layers, 0);
		for (Node& node : nodes) {
			++nodes_in_layer[node.layer];
		}
		for (int i = 0; i < layers - 1; i++) {
			int nodes_in_front = 0;
			for (int j = i + 1; j < layers; j++) nodes_in_front += nodes_in_layer[j];
			maxConnections += nodes_in_layer[i] * nodes_in_front;
		}
		if (maxConnections == genes.size()) return true;
		else return false;
	}
	
	bool Genome::RandomConnectionsAreBad(int node1, int node2) {
		return (nodes[node1].layer == nodes[node2].layer) || nodes[node1].IsConnectedTo(nodes[node2]);
	}

	void Genome::AddConnection(std::vector<ConnectionHistory>& innovationHistories) {
		if (IsFullyConnected()) return;
		int randomNode1 = (int)std::floorf(Random::random() * nodes.size());
		int randomNode2 = (int)std::floorf(Random::random() * nodes.size());
		while (RandomConnectionsAreBad(randomNode1, randomNode2)) {
			randomNode1 = (int)std::floorf(Random::random() * nodes.size());
			randomNode2 = (int)std::floorf(Random::random() * nodes.size());
		}
		if (nodes[randomNode1].layer > nodes[randomNode2].layer) std::swap(randomNode1, randomNode2);
		int ConnectionInnovationNumber = GetInnovationNumber(innovationHistories, nodes[randomNode1], nodes[randomNode2]);
		genes.emplace_back(nodes[randomNode1], nodes[randomNode2],
			Random::randomRange(), ConnectionInnovationNumber, settings["Weight Mutation Ratio"]);
		ConnectNodes();
	}

	void Genome::AddNode(std::vector<ConnectionHistory>& innovationHistories) {
		if (genes.empty()) {
			AddConnection(innovationHistories);
			return;
		}
		int rc = (int)std::floorf(Random::random() * genes.size());
		while (genes[rc].fromNode == &nodes[bias_node] && genes.size() != 1) rc = (int)std::floorf(Random::random() * genes.size());
		genes[rc].enabled = false;
		int NewNodeNo = next_node;
		std::cout << genes[1].fromNode << " " << genes[1].fromNode->number << " " << genes[1].innovationNo << " " <<
			genes[1].fromNode->output_connections.size() << " " << &nodes[0] << std::endl;
		nodes.emplace_back(NewNodeNo, Activation_Functions::Activation_functions[settings["Hidden Activation"]]);
		std::cout << genes[1].fromNode << " " << genes[1].fromNode->number << " " << genes[1].innovationNo << " " <<
			genes[1].fromNode->output_connections.size() << " " << &nodes[0] << std::endl;
		++next_node;
		Node node = GetNode(NewNodeNo);
		float wmr = settings["Weight Mutation Ratio"];
		int ConnectionInnovationNumber = GetInnovationNumber(innovationHistories, *genes[rc].fromNode, node);
		genes.emplace_back(*genes[rc].fromNode, node, 1, ConnectionInnovationNumber, wmr);

		ConnectionInnovationNumber = GetInnovationNumber(innovationHistories, node, *genes[rc].toNode);
		genes.emplace_back(node, *genes[rc].toNode, genes[rc].weight, ConnectionInnovationNumber, wmr);
		node.layer = genes[rc].fromNode->layer + 1;
		
		ConnectionInnovationNumber = GetInnovationNumber(innovationHistories, nodes[bias_node], node);
		genes.emplace_back(nodes[bias_node], node, 0, ConnectionInnovationNumber, wmr);

		if (node.layer == genes[rc].toNode->layer) {
			for (Node& Node : nodes) if (Node.layer >= node.layer && &Node != &node) Node.layer += 1;
			layers += 1;
		}
		ConnectNodes();
	}

	void Genome::FullyConnect(std::vector<ConnectionHistory>& innovationHistories) {
		for (int i = 0; i < (input_nodes + 1) * output_nodes; i++) AddConnection(innovationHistories);
		ConnectNodes();
	}

	void Genome::PartialConnect(std::vector<ConnectionHistory>& innovationHistories) {
		for (int i = 0; i < ((input_nodes + 1) * output_nodes)/2; i++) AddConnection(innovationHistories);
		ConnectNodes();
	}

	void Genome::MinimumConnect(std::vector<ConnectionHistory>& innovationHistories) {
		for (int i = 0; i < 3; i++) AddConnection(innovationHistories);
		ConnectNodes();
	}

	void Genome::Mutate(std::vector<ConnectionHistory>& innovationHistories) {
		if (Random::random() < settings["Weight Mutate Percent"])
			for (ConnectionGene& gene : genes) {
				if (Random::random() < settings["Weight Mutation Percent"])
					gene.MutateWeight();
				if (Random::random() < settings["Connection Toggle Percent"])
					gene.enabled = !gene.enabled;
			}
		if (Random::random() < settings["Connection Mutation Percent"])
			AddConnection(innovationHistories);
		if (Random::random() < settings["Node Mutation Percent"])
			AddNode(innovationHistories);
	}

	Genome Genome::Crossover(Genome& parrent2) {
		Genome child;
		child.input_nodes = input_nodes;
		child.output_nodes = output_nodes;
		child.settings = settings;
		child.layers = layers;
		child.next_node = next_node;
		child.bias_node = bias_node;
		std::vector<ConnectionGene> childGene;
		std::vector<bool> isEnabled;
		for (int i = 0; i < genes.size(); i++) {
			bool setEnabled = true;
			int parrent2Gene = MatchingGene(parrent2, genes[i].innovationNo);
			if (parrent2Gene != -1) {
				if (!(genes[i].enabled || parrent2.genes[i].enabled))
					if (Random::random() < settings["Enable Percent"])
						setEnabled = false;
				if (Random::random() < settings["Parent Gene Percent"]) childGene.push_back(genes[i]);
				else childGene.push_back(parrent2.genes[i]);
			}
			else {
				childGene.push_back(genes[i]);
				setEnabled = genes[i].enabled;
			}
			isEnabled.push_back(setEnabled);
		}
		for (Node& node : nodes) child.nodes.push_back(node);
		for (int i = 0; i < childGene.size(); i++) {
			child.genes.push_back(ConnectionGene::Clone(childGene[i],
				child.GetNode(childGene[i].fromNode->number), child.GetNode(childGene[i].toNode->number)));
			child.genes[i].enabled = isEnabled[i];
		}
		child.ConnectNodes();
		return child;
	}

	int Genome::MatchingGene(Genome& parent, int innovationNumber) {
		for (int i = 0; i < parent.genes.size(); i++) if (parent.genes[i].innovationNo == innovationNumber) return i;
		return -1;
	}
}