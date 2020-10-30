#include "extras.hpp"
#include <cmath>

namespace NEAT {
	namespace Activation_Functions {
		float linear(const float& x) { return x; }
		float tanh(const float& x) { return std::tanhf(x); }
		float sigmoid(const float& x) { return 1.0f / (1.0f + std::expf(-x)); }
		float relu(const float& x) { return std::fmaxf(0, x); }
		float elu(const float& x) { return std::max(x, std::expf(-x) - 1); }
		float softplus(const float& x) { return std::logf(std::expf(x) + 1); }
	}
}