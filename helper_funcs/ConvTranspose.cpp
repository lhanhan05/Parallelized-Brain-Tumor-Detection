#include "ConvTranspose.h"
#include "conv1d.h"
#include "conv2d.h"
#include "resampling.h"

// Constructor for ConvTranspose1d
ConvTranspose1d::ConvTranspose1d(int in_channels, int out_channels, int kernel_size, int upsampling_factor,
                                 std::function<void(Eigen::Tensor<double, 3>&)> weight_init_fn,
                                 std::function<void(Eigen::Tensor<double, 1>&)> bias_init_fn)
    : upsampling_factor(upsampling_factor),
      upsample1d(upsampling_factor),
      conv1d_stride1(in_channels, out_channels, kernel_size)
{
    // Initialize weights and biases if initialization functions are provided
    if (weight_init_fn) {
        conv1d_stride1.initialize_weights(weight_init_fn);
    }
    if (bias_init_fn) {
        conv1d_stride1.initialize_bias(bias_init_fn);
    }
}

// Forward pass for ConvTranspose1d
Eigen::Tensor<double, 3> ConvTranspose1d::forward(const Eigen::Tensor<double, 3> &A) {
    // Upsample the input
    Eigen::Tensor<double, 3> A_upsampled = upsample1d.forward(A);

    // Apply the Conv1d_stride1 layer
    Eigen::Tensor<double, 3> Z = conv1d_stride1.forward(A_upsampled);

    return Z;
}

// Backward pass for ConvTranspose1d
Eigen::Tensor<double, 3> ConvTranspose1d::backward(const Eigen::Tensor<double, 3> &dLdZ) {
    // Backward pass through Conv1d_stride1
    Eigen::Tensor<double, 3> delta_out = conv1d_stride1.backward(dLdZ);

    // Backward pass through upsampling
    Eigen::Tensor<double, 3> dLdA = upsample1d.backward(delta_out);

    return dLdA;
}

// Constructor for ConvTranspose2d
ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int upsampling_factor,
                                 std::function<void(std::vector<std::vector<std::vector<std::vector<double>>>>&)> weight_init_fn,
                                 std::function<void(std::vector<double>&)> bias_init_fn)
    : upsampling_factor(upsampling_factor),
      upsample2d(upsampling_factor),
      conv2d_stride1(in_channels, out_channels, kernel_size)
{
    // Initialize weights and biases if initialization functions are provided
    if (weight_init_fn) {
        conv2d_stride1.initialize_weights(weight_init_fn);
    }
    if (bias_init_fn) {
        conv2d_stride1.initialize_bias(bias_init_fn);
    }
}

// Forward pass for ConvTranspose2d
std::vector<std::vector<std::vector<std::vector<double>>>> ConvTranspose2d::forward(
    const std::vector<std::vector<std::vector<std::vector<double>>>> &A) {
    // Upsample the input
    auto A_upsampled = upsample2d.forward(A);

    // Apply the Conv2d_stride1 layer
    auto Z = conv2d_stride1.forward(A_upsampled);

    return Z;
}

// Backward pass for ConvTranspose2d
std::vector<std::vector<std::vector<std::vector<double>>>> ConvTranspose2d::backward(
    const std::vector<std::vector<std::vector<std::vector<double>>>> &dLdZ) {
    // Backward pass through Conv2d_stride1
    auto delta_out = conv2d_stride1.backward(dLdZ);

    // Backward pass through upsampling
    auto dLdA = upsample2d.backward(delta_out);

    return dLdA;
}
