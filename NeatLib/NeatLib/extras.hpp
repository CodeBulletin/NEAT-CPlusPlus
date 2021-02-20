#pragma once
#include <unordered_map>
#include <random>
#include <functional>

namespace NEAT {

	namespace Activations {
		static const int TanH = 0;
		static const int Sigmoid = 1;
		static const int ReLu = 2;
		static const int ELU = 3;
		static const int SoftPlus = 4;

		float linear(const float&);
		float tanh(const float&);
		float sigmoid(const float&);
		float relu(const float&);
		float elu(const float&);
		float softPlus(const float&);

		static std::unordered_map <int, float (*)(const float&)> Activations = {
			{TanH, tanh},
			{Sigmoid, sigmoid},
			{ReLu, relu}, 
			{ELU, elu},
			{SoftPlus, softPlus},
		};
	}

	namespace Random {
		static auto Random = std::bind(std::uniform_real_distribution<float>(0.0f,
			std::nextafter(1.0f, FLT_MAX)), std::mt19937{ std::random_device{}() });

		static auto RandomRange = std::bind(std::uniform_real_distribution<float>(-1.0f,
			std::nextafter(1.0f, FLT_MAX)), std::mt19937{ std::random_device{}() });
	}

	namespace Defaults {
		static const int DontConnect = 0;
		static const int MinimumConnect = 1;
		static const int PartialConnect = 2;
		static const int FullyConnect = 3;

		static std::unordered_map<std::string, float> NeatDefaults = {
			{"Output Activation", Activations::TanH},
			{"Hidden Activation", Activations::TanH},
			{"Genome Inputs", 3},
			{"Genome Outputs", 3},
			{"Stalness Factor", 15},
			{"Initial Connections", DontConnect},
			{"Weight Mutation Ratio", 0.01},
			{"Enable Percent", 0.75},
			{"Parent Gene Percent", 0.5},
			{"Weight Mutate Percent", 0.8},
			{"Weight Mutation Percent", 0.99},
			{"Connection Mutation Percent", 0.01},
			{"Node Mutation Percent", 0.02},
			{"Connection Toggle Percent", 0.01},
			{"Cross Over Percent", 0.75},
			{"Debug", false},
			{"_nextConnectionNo", 1000},
			{"excessCoeff", 1.0},
			{"weightDiffCoeff", 0.5},
			{"compatibilityThreshold", 3.0},
			{"largeGenomeNormaliser", 20.0},
		};
	}
}