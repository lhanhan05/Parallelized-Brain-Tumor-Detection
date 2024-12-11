#include "LinearLayer.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

LinearLayer::LinearLayer(int indim, int outdim, int rand_seed) {
    indim = indim;
    outdim = outdim;
    weights_momentum = Tensor<float, 2>(indim, outdim);
    biases_momentum = Tensor<float, 2>(outdim, 1);

    std::default_random_engine generator(rand_seed);
    std::uniform_real_distribution<float> distribution(-1*std::sqrt(6.0f / (indim + outdim)), std::sqrt(6.0f / (indim + outdim)));

    weights = Tensor<float, 2>(indim, outdim);
    for (int i = 0; i < indim; ++i) {
        for (int j = 0; j < outdim; ++j) {
            weights(i, j) = distribution(generator);
        }
    }

    biases = Tensor<float, 2>(outdim, 1);  // Biases initialized to zero
}

Tensor<float, 2> LinearLayer::forward(const Tensor<float, 2>& inputs) {
    this->inputs = inputs;
    Tensor<float, 2> input_weight_prod = inputs.contract(weights, array<int, 2>({1, 0}));

    Eigen::array<int, 2> broadcast_dims = {0, static_cast<int>(inputs.dimension(0))};
    Tensor<float, 2> output = input_weight_prod + biases.broadcast(broadcast_dims);
    return output;
}

std::vector<Tensor<float, 2>> LinearLayer::backward(const Tensor<float, 2>& dloss) {
    Tensor<float, 2> grad_weights = inputs.contract(dloss, array<int, 2>({0, 0}));
    Tensor<float, 2> grad_inputs = dloss.contract(weights, array<int, 2>({1, 1}));
    Tensor<float, 2> grad_biases = dloss.sum(array<int, 1>({0}));  // Sum across the batch dimension

    return {grad_weights, grad_biases, grad_inputs};
}

void LinearLayer::update(float learning_rate, float momentum_coeff) {
    weights_momentum = momentum_coeff * weights_momentum + (inputs.contract(inputs, array<int, 2>({0, 0})) / inputs.dimension(0));
    biases_momentum = momentum_coeff * biases_momentum + (biases / inputs.dimension(0));

    weights -= learning_rate * weights_momentum;
    biases -= learning_rate * biases_momentum;
}

std::pair<Tensor<float, 2>, Tensor<float, 2>> LinearLayer::getWb() {
    return {weights, biases};
}

