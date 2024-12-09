#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Im2Col.h"

using namespace Eigen;
using namespace Im2Col;

class Conv {
public:
    Conv(std::tuple<int, int, int>& input_shape, 
        std::tuple<int, int, int>& filter_shape, 
         int rand_seed = 0);

    Tensor<float, 4> forward(const Tensor<float, 4>& inputs, int stride = 1, int pad = 2);

    std::vector<Tensor<float, 4>> backward(const Tensor<float, 4>& dloss);

    void update(float learning_rate = 0.01, float momentum_coeff = 0.5);

    std::pair<Tensor<float, 4>, Tensor<float, 1>> get_wb_conv();

private:
    int num_filters, C, H, W, k_height, k_width, N;
    int out_width, out_height, pad, stride;
    Tensor<float, 4> weights, weights_momentum, grad_weights;
    Tensor<float, 1> biases, biases_momentum, grad_biases;
    Tensor<float, 4> X;
    Tensor<float, 4> grad_inputs;
    Tensor<float, 2> weights_reshape;

};