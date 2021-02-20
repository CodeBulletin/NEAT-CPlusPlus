#include "Node.hpp"
#include "ConnectionGene.hpp"
#include <vector>

namespace NEAT {
	Node::Node(): m_number{ 0 }, m_layer{ 0 }, m_inputSum{ 0 }, m_outputValue{ 0 }, m_activation{} {}

	Node::Node(int _innovation_number, float (*_activation_function)(const float&)) :
		m_number{ _innovation_number }, m_layer{ 0 }, m_inputSum{ 0 }, m_outputValue{ 0 },
		m_activation{ _activation_function } {}

	Node Node::clone(const Node& _node) {
		Node new_node;
		new_node.m_number = _node.m_number;
		new_node.m_layer = _node.m_layer;
		new_node.m_activation = _node.m_activation;
		return new_node;
	}

	bool Node::isConnectedTo(const Node& _other) {
		if (m_layer == _other.m_layer) {
			return false;
		}

		else if (m_layer > _other.m_layer) {
			for (const ConnectionGene& connection : _other.m_outputConnections) {
				if (connection.m_toNodeNumber == m_number) {
					return true;
				}
			}
		}

		else {
			for (const ConnectionGene& connection : m_outputConnections) {
				if (connection.m_toNodeNumber == _other.m_number) {
					return true;
				}
			}
		}

		return false;
	}

	void Node::engage(std::vector<Node>& _nodes) {
		if (m_layer != 0) {
			m_outputValue = m_activation(m_inputSum);
		}

		for (ConnectionGene& gene : m_outputConnections) {
			if (gene.m_enabled) {
				_nodes[gene.m_toNode].m_inputSum += gene.m_weight * m_outputValue;
			}
		}
	}
}