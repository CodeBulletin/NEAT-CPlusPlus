#include "Population.hpp"

std::ostream& operator<<(std::ostream& os, NEAT::ConnectionGene gene) {
	os << "<fromNode: " << gene.m_fromNode << " to node: " <<gene.m_toNode <<
		" | " << (gene.m_enabled ? " enabled" : "disabled") << " | " << gene.m_innovationNo << " | " << gene.m_weight << ">";
	return os;
}

using namespace NEAT;

struct MasterPlayer
{
	MasterPlayer() : m_fitness{ 0.0f }, m_score{ 0.0f }, m_bestScore{ 0 }, m_settings{},
		m_brain{}, m_gen{ 0 }, m_vision{}, m_isDead{ false } {}

	MasterPlayer(const Player& clone) : m_fitness{ clone.m_fitness }, m_score{ clone.m_score }, m_bestScore{ clone.m_bestScore },
		m_settings{ clone.m_settings }, m_brain{ clone.m_brain }, m_gen{ clone.m_gen }, m_vision{ clone.m_vision }, m_isDead{ clone.m_isDead } {};

	MasterPlayer(std::unordered_map<std::string, float>& _settings) : m_fitness{ 0.f }, m_score{ 0.f },
		m_bestScore{ 0 }, m_gen{ 0 }, m_vision{}, m_isDead{ false } {
		m_settings = _settings;
		m_brain = Genome(m_settings);
	}

	MasterPlayer clone() {
		Player clone = Player(m_settings);
		clone.m_brain = NEAT::Genome::clone(m_brain);
		clone.m_fitness = m_fitness;
		clone.m_brain.generateNetwork();
		clone.m_gen = m_gen;
		clone.m_bestScore = m_bestScore;
		return clone;
	}
	MasterPlayer cloneForReplay() {
		Player clone = Player(m_settings);
		clone.m_brain = NEAT::Genome::clone(m_brain);
		clone.m_fitness = m_fitness;
		clone.m_brain.generateNetwork();
		clone.m_gen = m_gen;
		clone.m_bestScore = m_bestScore;
		return clone;
	}
	MasterPlayer crossover(Player& _other_parrent) {
		Player child = Player(m_settings);
		child.m_brain = m_brain.crossover(_other_parrent.m_brain);
		child.m_brain.generateNetwork();
		return child;
	}
	void calculateFitness() {
		m_score = 1.0f;
		m_fitness = m_score;
	}
);

int main() {
	std::unordered_map<std::string, float> settings = {
		{"Debug", 1},
		{"Connection Mutation Percent", 0.08},
		{"Node Mutation Percent", 0.02},
		{"Genome Inputs", 5},
		{"Genome Outputs", 5},
		{"Initial Connections", 1}
	};

	Population<Player> populations(10, settings);

	while (true) {
		//system("cls");
		populations.naturalSelection();
		std::cout << std::endl << std::endl;
		//std::cin.get();
		//system("pause");
	}
}