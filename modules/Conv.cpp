#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../helper_funcs/Im2Col.h"
#include "Conv.h"


using namespace Eigen;
using namespace Im2Col;


Conv::Conv(std::tuple<int, int, int>& input_shape, 
    std::tuple<int, int, int>& filter_shape, 
    int rand_seed = 0)
{
    // Unpacking input shape and filter shape
    auto [num_filters, k_height, k_width] = filter_shape;
    auto [C, H, W] = input_shape;

    // Initialize layer dimensions
    this->num_filters = num_filters;
    this->C = C;
    this->H = H;
    this->W = W;
    this->k_height = k_height;
    this->k_width = k_width;

    // random weight initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-std::sqrt(6.0 / ((num_filters + C) * k_height * k_width)), 
                                            std::sqrt(6.0 / ((num_filters + C) * k_height * k_width)));

    weights = Tensor<float, 4>(num_filters, C, k_height, k_width);
    for (int i = 0; i < num_filters; ++i) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < k_height; ++h) {
                for (int w = 0; w < k_width; ++w) {
                    weights(i, c, h, w) = dist(gen);
                }
            }
        }
    }

    weights_momentum = Tensor<float, 4>(num_filters, C, k_height, k_width);
    weights_momentum.setZero();
    biases = Tensor<float, 1>(num_filters);
    biases.setZero();
    biases_momentum = Tensor<float, 1>(num_filters);
    biases_momentum.setZero();
}

// Forward pass of convolution
Tensor<float, 4> Conv::forward(const Tensor<float, 4>& inputs, int stride = 1, int pad = 2)
{
    this->pad = pad;
    this->stride = stride;
    this->N = inputs.dimension(0);
    this->X = inputs;

    Tensor<float, 2> input_cols = im2col(inputs, k_height, k_width, pad, stride);

    weights_reshape(num_filters, C * k_height * k_width);
    for (int i = 0; i < num_filters; ++i) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < k_height; ++h) {
                for (int w = 0; w < k_width; ++w) {
                    weights_reshape(i, c * k_height * k_width + h * k_width + w) = weights(i, c, h, w);
                }
            }
        }
    }

    out_width = (W + 2 * pad - k_width) / stride + 1;
    out_height = (H + 2 * pad - k_height) / stride + 1;

    Tensor<float, 2> output = weights_reshape.contract(input_cols, array<Index, 2>{0, 1});
    output = output + biases.broadcast(array<Index, 1>{0});  // Broadcasting biases
    
    Tensor<float, 4> output_reshape(num_filters, out_height, out_width, N);
    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < num_filters; ++f) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    output_reshape(f, h, w, n) = output(f, n * out_height * out_width + h * out_width + w);
                }
            }
        }
    }
    
    return output_reshape;
}

// Backward pass of convolution
std::vector<Tensor<float, 4>> Conv::backward(const Tensor<float, 4>& dloss)
{
    Tensor<float, 4> dloss_transpose = dloss.shuffle(array<int, 4>{1, 2, 3, 0});
    Tensor<float, 2> dloss_reshape(num_filters, out_height * out_width * N);
    for (int f = 0; f < num_filters; ++f) {
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    dloss_reshape(f, n * out_height * out_width + h * out_width + w) = dloss_transpose(f, h, w, n);
                }
            }
        }
    }

    Tensor<float, 2> mul_output = weights_reshape.contract(dloss_reshape, array<Index, 2>{1, 0});
    DSizes<int, 4> dims({N, C, H, W});
    Tensor<float, 4> grad_inputs = im2col_bw(mul_output, dims, k_height, k_width, pad, stride);

    Tensor<float, 2> x_col = im2col(X, k_height, k_width, pad, stride);
    Tensor<float, 2> grad_weight_pre = dloss_reshape.contract(x_col, array<Index, 2>{1, 0});
    
    Tensor<float, 4> grad_weights(num_filters, C, k_height, k_width);
    for (int f = 0; f < num_filters; ++f) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < k_height; ++h) {
                for (int w = 0; w < k_width; ++w) {
                    grad_weights(f, c, h, w) = grad_weight_pre(f, c * k_height * k_width + h * k_width + w);
                }
            }
        }
    }

    Tensor<float, 1> grad_biases = dloss_reshape.sum(array<int, 1>{1});
    
    return {grad_weights, grad_biases, grad_inputs};
}

void Conv::update(float learning_rate = 0.01, float momentum_coeff = 0.5)
{
    weights_momentum = momentum_coeff * weights_momentum + grad_weights / N;
    biases_momentum = momentum_coeff * biases_momentum + grad_biases / N;

    weights -= learning_rate * weights_momentum;
    biases -= learning_rate * biases_momentum;
}

std::pair<Tensor<float, 4>, Tensor<float, 1>> Conv::get_wb_conv()
{
    return {weights, biases};
}
