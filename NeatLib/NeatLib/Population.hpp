#pragma once

#include "Species.hpp"
#include <iostream>

namespace NEAT {
	template<typename Type>
	struct Population {
		int m_size, m_gen;
		std::unordered_map<std::string, float> m_settings;
		float m_bestScore;
		bool m_massExtinctionEvent, m_newStage;
		std::vector<Type> m_population;
		std::vector<Type> m_bestPlayers;
		std::vector<Species<Type>> m_species;
		std::vector<ConnectionHistory> m_innovationHistories;
		Type m_bestPlayer;

		Population() : m_size{ 0 }, m_gen{ 0 }, m_settings{}, m_bestScore{ 0 },
			m_massExtinctionEvent{ false }, m_newStage{ false }, m_population{}, m_bestPlayers{},
			m_species{}, m_innovationHistories{}, m_bestPlayer{} {}

		Population(int _size, std::unordered_map<std::string, float>& _settings) : m_size{ _size },
			m_gen{ 0 }, m_settings{_settings}, m_bestScore{ 0 }, m_massExtinctionEvent{ false },
			m_newStage{ false }, m_bestPlayers{}, m_species{}, m_innovationHistories{}, m_bestPlayer{} {
			for (auto& it : Defaults::NeatDefaults) {
				if (m_settings.find(it.first) == m_settings.end()) {
					m_settings[it.first] = it.second;
				}
			}

			for (int i = 0; i < m_size; i++) {
				m_population.emplace_back(m_settings);
				if ((int)m_settings["Initial Connections"] == 0);
				else if ((int)m_settings["Initial Connections"] == 1) {
					m_population[i].m_brain.minimumConnect(m_innovationHistories);
				} 
				else if ((int)m_settings["Initial Connections"] == 2) {
					m_population[i].m_brain.partialConnect(m_innovationHistories);
				}
				else if ((int)m_settings["Initial Connections"] == 3) {
					m_population[i].m_brain.fullyConnect(m_innovationHistories);
				}
				m_population[i].m_brain.mutate(m_innovationHistories);
				m_population[i].m_brain.generateNetwork();
			}
		}

		bool done() {
			for (int i = 0; i < m_population.size(); i++) {
				if (!m_population[i].m_isDead) {
					return false;
				}
			}
			return true;
		}

		void setBestPlayer() {
			Type& temp_best = m_population[m_species[0].m_players[0]];
			temp_best.m_gen = m_gen;
			if (temp_best.m_score > m_bestScore) {
				Type temp = temp_best.cloneForReplay();
				m_bestPlayers.push_back(temp);
				m_bestScore = temp.m_bestScore;
				m_bestPlayer = temp_best.cloneForReplay();
				m_bestPlayer.m_brain.generateNetwork();
			}
		}

		void speciate() {
			for (Species<Type>& species : m_species) {
				species.m_players.clear();
			}
			for (size_t i = 0; i < m_population.size(); i++) {
				bool species_found = false;
				for (Species<Type>& species : m_species) {
					if (species.sameSpecies(m_population[i].m_brain)) {
						species.addToSpecies(i);
						species_found = true;
						break;
					}
				}
				if (!species_found) {
					m_species.push_back(Species<Type>(m_settings, i, m_population));
				}
			}
		}

		void killEmptySpecies() {
			size_t i = 0;
			while (i < m_species.size()) {
				if (m_species[i].m_players.size() == 0) {
					m_species.erase(m_species.begin() + i);
					--i;
				}
				++i;
			}
		}

		void calculateFitness() {
			for (Type& player : m_population) {
				player.calculateFitness();
			}
		}

		void sortSpecies() {
			for (Species<Type>& species : m_species) {
				species.sortSpecies(m_population);
			}
			std::sort(m_species.begin(), m_species.end(), sortElement);
		}

		void massExtinction() {
			size_t i = 5;
			while (i < m_species.size()) {
				m_species.erase(m_species.begin() + i);
			}
		}

		void cullSpecies() {
			for (Species<Type>& species : m_species) {
				species.cull();
				species.fitnessSharing(m_population);
				species.setAverage(m_population);
			}
		}

		void killStaleSpecies() {
			size_t i = 2;
			while (i < m_species.size()) {
				if (m_species[i].m_staleness >=  m_settings["Stalness Factor"]) {
					m_species.erase(m_species.begin() + i);
					--i;
				}
				++i;
			}
		}

		float getAverageFitnessSum() {
			float sum_of_average_fitness = 0;
			for (Species<Type>& species : m_species) {
				sum_of_average_fitness += species.m_averageFitness;
			}
			return sum_of_average_fitness;
		}

		void killBadSpecies() {
			float average_sum = getAverageFitnessSum();
			size_t i = 1;
			while (i < m_species.size()) {
				if (m_species[i].m_averageFitness / average_sum * (float)m_population.size() < 1.0f) {
					m_species.erase(m_species.begin() + i);
					--i;
				}
				++i;
			}
		}

		void naturalSelection() {
			speciate();
			killEmptySpecies();
			calculateFitness();
			sortSpecies();

			if (m_massExtinctionEvent) {
				massExtinction();
				m_massExtinctionEvent = false;
			}

			cullSpecies();
			setBestPlayer();
			killStaleSpecies();
			killBadSpecies();

			float average_sum = getAverageFitnessSum();
			std::vector<Type> childerns;

			bool debug = m_settings["Debug"];
			if (debug) {
				std::cout << "Genrations : " << m_gen << " | " << "Number of mutations : " <<
					m_innovationHistories.size() << " | " << "Toatal species : " <<
					m_species.size() << std::endl;
			}

			int i = 0;

			for (Species<Type>& species : m_species) {

				if (debug) {
					std::cout << "Best unadjusted fitness of species " << i << " : " <<
						species.m_bestFitness << std::endl;
					for (size_t j = 0; j < species.m_players.size(); j++) {
						std::cout << "<Player : " << j << " | Fitness : " <<
							m_population[species.m_players[j]].m_fitness << " | Score : " <<
							m_population[species.m_players[j]].m_score << ">, ";
					}
					++i;
				}

				childerns.push_back(species.m_champ.cloneForReplay());

				int no_of_childerns = (int)((float)std::floor(species.m_averageFitness/average_sum) *
					m_population.size()) - 1;

				for (int j = 0; j < no_of_childerns; j++) {
					childerns.push_back(species.giveMeBaby(m_innovationHistories, m_population));
				}
			}
			
			while (childerns.size() < m_population.size()) {
				childerns.push_back(m_species[0].giveMeBaby(m_innovationHistories, m_population));
			}

			m_population.clear();
			m_population = childerns;
			m_gen += 1;
			for (Type& player : m_population) {
				player.m_brain.generateNetwork();
			}
		}

		static bool sortElement(Species<Type>& _species_1, Species<Type>& _species_2) {
			return _species_1.m_bestFitness > _species_2.m_bestFitness;
		}
 	};
}
