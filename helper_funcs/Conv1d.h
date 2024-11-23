#ifndef CONV1D_H
#define CONV1D_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "resampling.h"  // Include the helper functions from your provided file

using namespace Eigen;

class Conv1d_stride1 {
public:
    // Constructor
    Conv1d_stride1(int in_channels, int out_channels, int kernel_size);

    // Forward and backward functions
    Tensor<double, 3> forward(const Tensor<double, 3> &A);
    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ);

    // Weight and bias initialization functions
    void initialize_weights(std::function<void(Tensor<double, 3>&)> weight_init_fn = nullptr);
    void initialize_bias(std::function<void(Tensor<double, 1>&)> bias_init_fn = nullptr);

private:
    int in_channels;
    int out_channels;
    int kernel_size;

    Tensor<double, 3> W;  // Weights
    Tensor<double, 1> b;  // Bias

    Tensor<double, 3> dLdW;  // Gradient of loss w.r.t weights
    Tensor<double, 1> dLdb;  // Gradient of loss w.r.t bias

    Tensor<double, 3> A;  // Input tensor from forward pass
};

class Conv1d {
public:
    // Constructor
    Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding = 0);

    // Forward and backward functions
    Tensor<double, 3> forward(const Tensor<double, 3> &A);
    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ);

private:
    int stride;
    int pad;

    Conv1d_stride1 conv1d_stride1;
    Downsample1d downsample1d;
};

#endif // CONV1D_H
