#include "Node.hpp"
#include "ConnectionGene.hpp"
#include <vector>

namespace NEAT {
	Node::Node(int ino, float (*func)(const float&)) : number(ino), layer(0), inputSum(0), outputValue(0), activation(func) {}

	Node Node::Clone(const Node& node) {
		Node newnode;
		newnode.number = node.number;
		newnode.layer = node.layer;
		newnode.activation = node.activation;
		return newnode;
	}

	bool Node::IsConnectedTo(const Node& other) {
		if (layer == other.layer) return false;
		else if (layer > other.layer) {
			for (const ConnectionGene& connection : other.output_connections) {
				if (connection.toNode == this) {
					return true;
				}
			}
		}
		else {
			for (const ConnectionGene& connection : output_connections) {
				if (connection.toNode == &other) {
					return true;
				}
			}
		}
		return false;
	}

	void Node::Engage() {
		if (layer != 0)
			outputValue = activation(inputSum);
		for (ConnectionGene& gene : output_connections) if (gene.enabled) gene.toNode->inputSum += gene.weight * outputValue;
	}
}