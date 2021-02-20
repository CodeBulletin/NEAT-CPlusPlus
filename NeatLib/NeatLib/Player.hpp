#pragma once

#include "Genome.hpp"
#include "ConnectionGene.hpp"
#include "Node.hpp"
#include "ConnectionHistory.hpp"
#include <string>

namespace NEAT {
	struct Genome;
	struct Player {
		float m_fitness, m_score, m_bestScore;
		std::unordered_map<std::string, float> m_settings;
		Genome m_brain;
		int m_gen;
		std::vector<float> m_vision;
		bool m_isDead;

		Player() : m_fitness{ 0.0f }, m_score{ 0.0f }, m_bestScore{ 0 }, m_settings{},
			m_brain{}, m_gen{ 0 }, m_vision{}, m_isDead{ false } {}

		Player(const Player& clone) : m_fitness{ clone.m_fitness }, m_score{ clone.m_score }, m_bestScore{ clone.m_bestScore },
			m_settings{ clone.m_settings }, m_brain{ clone.m_brain }, m_gen{ clone.m_gen }, m_vision{ clone.m_vision }, m_isDead{ clone.m_isDead } {};

		Player(std::unordered_map<std::string, float>& _settings) : m_fitness{ 0.f }, m_score{ 0.f },
			m_bestScore{ 0 }, m_gen{ 0 }, m_vision{}, m_isDead{ false } {
			m_settings = _settings;
			m_brain = Genome(m_settings);
		}

		Player clone() {
			Player clone = Player(m_settings);
			clone.m_brain = NEAT::Genome::clone(m_brain);
			clone.m_fitness = m_fitness;
			clone.m_brain.generateNetwork();
			clone.m_gen = m_gen;
			clone.m_bestScore = m_bestScore;
			return clone;
		}
		Player cloneForReplay() {
			Player clone = Player(m_settings);
			clone.m_brain = NEAT::Genome::clone(m_brain);
			clone.m_fitness = m_fitness;
			clone.m_brain.generateNetwork();
			clone.m_gen = m_gen;
			clone.m_bestScore = m_bestScore;
			return clone;
		}
		Player crossover(Player& _other_parrent) {
			Player child = Player(m_settings);
			child.m_brain = m_brain.crossover(_other_parrent.m_brain);
			child.m_brain.generateNetwork();
			return child;
		}
		void calculateFitness() {
			m_score = 1.0f;
			m_fitness = m_score;
		}
	};
}