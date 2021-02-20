#pragma once
#include "extras.hpp"
#include "Player.hpp"
#include <cmath>

namespace NEAT {
	template<typename Type>
	struct Species
	{
		std::unordered_map<std::string, float> m_settings;
		std::vector<int> m_players;
		float m_averageFitness, m_bestFitness;
		int m_staleness;
		float m_excessCoef, m_weightDiffCoef, m_compatabilityThreshold;
		Genome m_best;
		Type m_champ;

		Species() : m_settings{}, m_players{}, m_averageFitness{ 0 }, m_bestFitness{ 0 }, m_staleness{ 0 },
			m_excessCoef{ 0 }, m_weightDiffCoef{ 0 }, m_compatabilityThreshold{ 0 }, m_best{}, m_champ{} {}

		Species(std::unordered_map<std::string, float>& _settings, int _player = -1, std::vector<Type>& _populations = {}) :
			m_settings{ _settings }, m_players{}, m_averageFitness{ 0 }, m_staleness{ 0 } {

			m_excessCoef = m_settings["excessCoeff"];
			m_weightDiffCoef = m_settings["weightDiffCoeff"];
			m_compatabilityThreshold = m_settings["compatibilityThreshold"];

			if (_player != -1) {
				m_players.push_back(_player);
				m_bestFitness = _populations[m_players[0]].m_fitness;
				m_best = Genome::clone(_populations[m_players[0]].m_brain);
				m_champ = _populations[m_players[0]].cloneForReplay();
			}
			else {
				m_best = Genome();
				m_champ = Type();
				m_bestFitness = 0;
			}
		}

		void addToSpecies(int _player) {
			m_players.push_back(_player);
		}

		void sortSpecies(std::vector<Type>& _populations) {
			std::sort(m_players.begin(), m_players.end(), [&_populations](int _player1, int _player2) {
				return _populations[_player1].m_fitness > _populations[_player2].m_fitness;
			});
			if (m_players.size() == 0) {
				m_staleness = 200;
				return;
			}
			if (_populations[m_players[0]].m_fitness > m_bestFitness) {
				m_staleness = 0;
				m_bestFitness = _populations[m_players[0]].m_fitness;
				m_best = Genome::clone(_populations[m_players[0]].m_brain);
				m_champ = _populations[m_players[0]].cloneForReplay();
			}
			else {
				++m_staleness;
			}
		}

		void setAverage(std::vector<Type>& _populations) {
			float sum = 0;
			for (int& player : m_players) {
				sum += _populations[player].m_fitness;
			}
			m_averageFitness = sum / (float)m_players.size();
		}

		Type& selectPlayer(std::vector<Type>& _populations) {
			float fitness_sum = 0;
			for (int& player : m_players) {
				fitness_sum += _populations[player].m_fitness;
			}

			float random_value = Random::Random() * fitness_sum;
			float running_sum = 0;
			for (int& player : m_players) {
				running_sum += _populations[player].m_fitness;
				if (running_sum > random_value) {
					return _populations[player];
				}
			}
			return _populations[m_players[0]];
		}

		Type giveMeBaby(std::vector<ConnectionHistory>& _innovation_histories, std::vector<Type>& _populations) {
			Type baby;
			if (Random::Random() < m_settings["Cross Over Percent"]) {
				Type parent_1 = selectPlayer(_populations);
				Type parent_2 = selectPlayer(_populations);
				if (parent_1.m_fitness >= parent_2.m_fitness) {
					baby = parent_1.crossover(parent_2);
				}
				else {
					baby = parent_2.crossover(parent_1);
				}
			}
			else {
				baby = selectPlayer(_populations).clone();
			}
			baby.m_brain.mutate(_innovation_histories);
			return baby;
		}

		bool sameSpecies(Genome& _gene) {
			float excess_and_disjoint = getExcessAndDisjoint(_gene, m_best);
			float average_weight_diff = getAverageWeightDiff(_gene, m_best);
			float large_genome_normalizer = (float)_gene.m_genes.size() - m_settings["largeGenomeNormaliser"];
			if (large_genome_normalizer < 1) {
				large_genome_normalizer = 1;
			}
			float compatibility = (m_excessCoef * excess_and_disjoint / large_genome_normalizer) +
				(m_weightDiffCoef * average_weight_diff);

			return compatibility < m_compatabilityThreshold;
		}

		void cull() {
			if (m_players.size() > 2) {
				int i = (int)(m_players.size()/2);
				m_players.erase(m_players.begin() + i, m_players.end());
			}
		}

		void fitnessSharing(std::vector<Type>& _populations) {
			for (int& player : m_players) {
				_populations[player].m_fitness /= (float)m_players.size();
			}
		}

		static float getExcessAndDisjoint(Genome& _gene_1, Genome& _gene_2) {
			int matching = 0;
			for (ConnectionGene& gene_1 : _gene_1.m_genes) {
				for (ConnectionGene& gene_2 : _gene_2.m_genes) {
					if (gene_1.m_innovationNo == gene_2.m_innovationNo) {
						matching += 1;
						break;
					}
				}
			}

			return (float)_gene_1.m_genes.size() + (float)_gene_2.m_genes.size() - 2.0f * (float) matching;
		}

		static float getAverageWeightDiff(Genome& _gene_1, Genome& _gene_2) {
			if (_gene_1.m_genes.size() == 0 || _gene_2.m_genes.size() == 0) {
				return 0.0f;
			}
			float matching = 0.0f;
			float totalDiff = 0.0f;
			for (ConnectionGene& gene_1 : _gene_1.m_genes) {
				for (ConnectionGene& gene_2 : _gene_2.m_genes) {
					if (gene_1.m_innovationNo == gene_2.m_innovationNo) {
						matching += 1.0f;
						totalDiff += std::abs(gene_1.m_weight - gene_2.m_weight);
						break;
					}
				}
			}
			if (matching == 0.0f) {
				return 100.0f;
			}
			else {
				return totalDiff / matching;
			}
		}
	};
}