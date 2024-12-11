// conv2d.cpp
#include "conv2d.h"
#include <vector>
#include <numeric>
#include <algorithm>

// Helper function for zero-padding a 4D tensor
std::vector<std::vector<std::vector<std::vector<double>>>> pad_input(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& input,
    int pad) {
    
    int N = input.size();
    int Cin = input[0].size();
    int Hin = input[0][0].size();
    int Win = input[0][0][0].size();

    // Create padded tensor
    std::vector<std::vector<std::vector<std::vector<double>>>> padded_input(
        N, std::vector<std::vector<std::vector<double>>>(
            Cin, std::vector<std::vector<double>>(
                Hin + 2 * pad, std::vector<double>(Win + 2 * pad, 0.0))));

    // Copy original input into padded tensor
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cin; ++c) {
            for (int h = 0; h < Hin; ++h) {
                for (int w = 0; w < Win; ++w) {
                    padded_input[n][c][h + pad][w + pad] = input[n][c][h][w];
                }
            }
        }
    }
    
    return padded_input;
}

// Conv2d_stride1 constructor
Conv2d_stride1::Conv2d_stride1(int in_channels, int out_channels, int kernel_size)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size) {

    // Initialize weights and biases with small random values
    W = std::vector<std::vector<std::vector<std::vector<double>>>>(
        out_channels, std::vector<std::vector<std::vector<double>>>(
            in_channels, std::vector<std::vector<double>>(
                kernel_size, std::vector<double>(kernel_size, 0.01)))); // Small initialization
    
    b = std::vector<double>(out_channels, 0.0);

    // Initialize gradients
    dLdW = W;
    dLdb = b;
}

// Conv2d_stride1 forward pass
std::vector<std::vector<std::vector<std::vector<double>>>> Conv2d_stride1::forward(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& A) {

    this->A = A;  // Store input for backpropagation
    int N = A.size();
    int Cin = A[0].size();
    int Hin = A[0][0].size();
    int Win = A[0][0][0].size();
    int Hout = Hin - kernel_size + 1;
    int Wout = Win - kernel_size + 1;

    // Output tensor
    std::vector<std::vector<std::vector<std::vector<double>>>> Z(
        N, std::vector<std::vector<std::vector<double>>>(
            out_channels, std::vector<std::vector<double>>(
                Hout, std::vector<double>(Wout, 0.0))));

    // Convolution operation
    for (int n = 0; n < N; ++n) {
        for (int cout = 0; cout < out_channels; ++cout) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    double sum = b[cout];
                    for (int cin = 0; cin < in_channels; ++cin) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                sum += A[n][cin][h + kh][w + kw] * W[cout][cin][kh][kw];
                            }
                        }
                    }
                    Z[n][cout][h][w] = sum;
                }
            }
        }
    }

    return Z;
}

// Conv2d_stride1 backward pass
std::vector<std::vector<std::vector<std::vector<double>>>> Conv2d_stride1::backward(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& dLdZ) {

    int N = A.size();
    int Hout = dLdZ[0][0].size();
    int Wout = dLdZ[0][0][0].size();

    // Initialize gradients to zero
    dLdW = std::vector<std::vector<std::vector<std::vector<double>>>>(
        out_channels, std::vector<std::vector<std::vector<double>>>(
            in_channels, std::vector<std::vector<double>>(
                kernel_size, std::vector<double>(kernel_size, 0.0))));
    dLdb = std::vector<double>(out_channels, 0.0);

    // Gradient of input
    std::vector<std::vector<std::vector<std::vector<double>>>> dLdA(
        N, std::vector<std::vector<std::vector<double>>>(
            in_channels, std::vector<std::vector<double>>(
                A[0][0].size(), std::vector<double>(A[0][0][0].size(), 0.0))));

    // Calculate gradients
    for (int n = 0; n < N; ++n) {
        for (int cout = 0; cout < out_channels; ++cout) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    double grad_output = dLdZ[n][cout][h][w];
                    dLdb[cout] += grad_output;
                    for (int cin = 0; cin < in_channels; ++cin) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                dLdW[cout][cin][kh][kw] += A[n][cin][h + kh][w + kw] * grad_output;
                                dLdA[n][cin][h + kh][w + kw] += W[cout][cin][kh][kw] * grad_output;
                            }
                        }
                    }
                }
            }
        }
    }

    return dLdA;
}

// Accessor for gradients
const std::vector<std::vector<std::vector<std::vector<double>>>>& Conv2d_stride1::get_dLdW() const {
    return dLdW;
}

const std::vector<double>& Conv2d_stride1::get_dLdb() const {
    return dLdb;
}

// Conv2d constructor
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : stride(stride), pad(padding), conv2d_stride1(in_channels, out_channels, kernel_size) {}

// Conv2d forward pass
std::vector<std::vector<std::vector<std::vector<double>>>> Conv2d::forward(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& A) {

    // Pad the input
    std::vector<std::vector<std::vector<std::vector<double>>>> padded_input = pad_input(A, pad);

    // Forward pass using stride-1 convolution
    std::vector<std::vector<std::vector<std::vector<double>>>> Z = conv2d_stride1.forward(padded_input);

    // Downsample the output
    return downsample2d.forward(Z, stride);
}

// Conv2d backward pass
std::vector<std::vector<std::vector<std::vector<double>>>> Conv2d::backward(
    const std::vector<std::vector<std::vector<std::vector<double>>>>& dLdZ) {

    // Upsample the gradient
    std::vector<std::vector<std::vector<std::vector<double>>>> dLdZ_upsampled = downsample2d.backward(dLdZ, stride);

    // Backward pass using stride-1 convolution
    return conv2d_stride1.backward(dLdZ_upsampled);
}
