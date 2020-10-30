#include "Genome.hpp"
#include "ConnectionHistory.hpp"
#include "Node.hpp"
#include "ConnectionGene.hpp"
#include "extras.hpp"
#include <cmath>
#include <iostream>

std::ostream& operator<<(std::ostream& os, NEAT::ConnectionGene gene) {
	os << "<fromNode: " << gene.fromNode->number << " to node: " <<gene.toNode->number <<
		" | " << (gene.enabled ? " enabled" : "disabled") << " | " << gene.innovationNo << " | " << gene.weight << ">";
	return os;
}

int main() {
	std::unordered_map<std::string, float> settings {
		{"Genome Inputs", 4},
		{"Genome Outputs", 4},
		{"Node Mutation Percent", 0.9}
	};
	std::vector<float> inputs = { 0.5f, 0.7f, 1.f, 0.5f };
	NEAT::Genome Gene1(settings), Gene2(settings), child;
	std::vector<NEAT::ConnectionHistory> history;

	Gene1.PartialConnect(history);
	Gene2.PartialConnect(history);
	while (true) {
		std::cout << "---Parrent1---" << std::endl;
		for (auto& gene : Gene1.genes) std::cout << gene << std::endl;
		std::cout << "---Parrent2---" << std::endl;
		for (auto& gene : Gene2.genes) std::cout << gene << std::endl;

		child = Gene1.Crossover(Gene2);
		std::cout << "---child---" << std::endl;
		for (auto& gene : child.genes) std::cout << gene << std::endl;
		Gene1.Mutate(history);
		Gene2.Mutate(history);
		system("pause");
		system("cls");
	}

	//child.GenerateNetwork();
	//auto a = child.FeedForward(inputs);
	//std::cout << a[0] << " | " << a[1] << " | " << a[2] << " | " << a[3] << std::endl;
	//std::cin.get();
}