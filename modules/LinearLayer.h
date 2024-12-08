#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

class LinearLayer {
public:
    LinearLayer(int indim, int outdim, int rand_seed = 0);

    Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2>& inputs);

    std::vector<Eigen::Tensor<float, 2>> backward(const Eigen::Tensor<float, 2>& dloss);

    void update(float learning_rate = 0.01f, float momentum_coeff = 0.5f);

    std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>> get_wb_fc();

private:
    int indim;   // Input dimension
    int outdim;  // Output dimension

    Eigen::Tensor<float, 2> weights;            // Weights tensor (indim, outdim)
    Eigen::Tensor<float, 2> weights_momentum;   // Momentum for weights
    Eigen::Tensor<float, 2> biases;             // Biases tensor (outdim, 1)
    Eigen::Tensor<float, 2> biases_momentum;    // Momentum for biases

    Eigen::Tensor<float, 2> inputs;
};