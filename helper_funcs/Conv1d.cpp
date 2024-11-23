#include "Conv1d.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

// Implementation of Conv1d_stride1 constructor
Conv1d_stride1::Conv1d_stride1(int in_channels, int out_channels, int kernel_size)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
      W(out_channels, in_channels, kernel_size), b(out_channels) {
    dLdW = Tensor<double, 3>(out_channels, in_channels, kernel_size);
    dLdb = Tensor<double, 1>(out_channels);
}

// Weight initialization (optional)
void Conv1d_stride1::initialize_weights(std::function<void(Tensor<double, 3>&)> weight_init_fn) {
    if (weight_init_fn) {
        weight_init_fn(W);
    } else {
        // Default initialization: normal distribution
        W.setRandom();
    }
}

// Bias initialization (optional)
void Conv1d_stride1::initialize_bias(std::function<void(Tensor<double, 1>&)> bias_init_fn) {
    if (bias_init_fn) {
        bias_init_fn(b);
    } else {
        // Default initialization: zeros
        b.setZero();
    }
}

// Forward function for Conv1d_stride1
Tensor<double, 3> Conv1d_stride1::forward(const Tensor<double, 3> &A) {
    this->A = A;
    int N = A.dimension(0);  // Batch size
    int Cout = W.dimension(0);  // Output channels
    int Wout = A.dimension(2) - W.dimension(2) + 1;  // Output width
    int k = kernel_size;

    Tensor<double, 3> Z(N, Cout, Wout);
    Z.setZero();

    for (int i = 0; i < Wout; ++i) {
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < Cout; ++co) {
                Z(n, co, i) = (A.chip(n, 0).slice(Eigen::array<long, 3>({0, 0, i}), Eigen::array<long, 3>({in_channels, 1, k})) * W.chip(co, 0)).sum() + b(co);
            }
        }
    }
    return Z;
}

// Backward function for Conv1d_stride1
Tensor<double, 3> Conv1d_stride1::backward(const Tensor<double, 3> &dLdZ) {
    int Wout = dLdZ.dimension(2);
    int k = W.dimension(2);
    int Win = Wout + k - 1;

    // Calculate dLdb
    dLdb.setZero();
    for (int n = 0; n < dLdZ.dimension(0); ++n) {
        for (int co = 0; co < out_channels; ++co) {
            dLdb(co) += dLdZ.chip(n, 0).chip(co, 1).sum();
        }
    }

    // Calculate dLdW
    dLdW.setZero();
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int co = 0; co < out_channels; ++co) {
            for (int i = 0; i < Wout; ++i) {
                dLdW.chip(co, 0).chip(ci, 0) += (dLdZ.chip(0, 0).chip(co, 1).slice(Eigen::array<long, 3>({0, i, 0}), Eigen::array<long, 3>({1, 1, k})));
            }
        }
    }

    // Calculate dLdA
    Tensor<double, 3> dLdA(dLdZ.dimension(0), in_channels, Win);
    dLdA.setZero();
    Tensor<double, 3> W_flipped = W.reverse(Eigen::array<bool, 3>({false, false, true}));

    for (int n = 0; n < dLdZ.dimension(0); ++n) {
        for (int ci = 0; ci < in_channels; ++ci) {
            for (int co = 0; co < out_channels; ++co) {
                for (int i = 0; i < Win; ++i) {
                    dLdA.chip(n, 0).chip(ci, 0) += (dLdZ.chip(n, 0).chip(co, 1).slice(Eigen::array<long, 3>({0, i, 0}), Eigen::array<long, 3>({1, 1, k})));
                }
            }
        }
    }

    return dLdA;
}

// Implementation of Conv1d constructor
Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : stride(stride), pad(padding), conv1d_stride1(in_channels, out_channels, kernel_size), downsample1d(stride) {
}

// Forward function for Conv1d
Tensor<double, 3> Conv1d::forward(const Tensor<double, 3> &A) {
    Tensor<double, 3> padded_A = A.pad(Eigen::array<long, 3>({0, 0, pad}));
    Tensor<double, 3> mask = conv1d_stride1.forward(padded_A);
    Tensor<double, 3> Z = downsample1d.forward(mask);
    return Z;
}

// Backward function for Conv1d
Tensor<double, 3> Conv1d::backward(const Tensor<double, 3> &dLdZ) {
    Tensor<double, 3> mask = downsample1d.backward(dLdZ);
    Tensor<double, 3> dLdA = conv1d_stride1.backward(mask);

    if (pad != 0) {
        dLdA = dLdA.slice(Eigen::array<long, 3>({0, 0, pad}), Eigen::array<long, 3>({dLdA.dimension(0), dLdA.dimension(1), dLdA.dimension(2) - 2 * pad}));
    }

    return dLdA;
}
