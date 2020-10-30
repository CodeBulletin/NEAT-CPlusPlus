#pragma once
#include <random>
#include <functional>
#include <unordered_map>
#include <string>

namespace NEAT {
	namespace Activation_Functions {
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
		float softplus(const float&);

		static std::unordered_map <int, float (*)(const float&)> Activation_functions = {
			{TanH, tanh},
			{Sigmoid, sigmoid},
			{ReLu, relu}, 
			{ELU, elu},
			{SoftPlus, softplus},
		};
	}

	namespace Connection_Type {
		static const int DontConnect = 0;
		static const int MinimumConnect = 1;
		static const int PartialConnect = 2;
		static const int FullyConnect = 3;
	}

	namespace Random {
		static auto random = std::bind(std::uniform_real_distribution<float>(0.0f, std::nextafter(1.0f, FLT_MAX)),
			std::mt19937{ std::random_device{}() });

		static auto randomRange = std::bind(std::uniform_real_distribution<float>(-1.0f, std::nextafter(1.0f, FLT_MAX)),
			std::mt19937{ std::random_device{}() });
	}

	namespace Defaults {
		static std::unordered_map<std::string, float> neat_defaults = {
			{"Output Activation", Activation_Functions::TanH},
			{"Hidden Activation", Activation_Functions::TanH},
			{"Genome Inputs", 3},
			{"Genome Outputs", 3},
			{"Stalness Factor", 15},
			{"Initial Connections", Connection_Type::DontConnect},
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
			{"largeGenomeNormaliser", 20.0}
		};
	}
}