#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "im2col.h"

using namespace Eigen;
namespace Im2Col {
    Tensor<float, 2> im2col(const Tensor<float, 4>& X, int k_height, int k_width, int padding = 1, int stride = 1) {
        //  X shape: [N, C, H, W]
        int N = X.dimension(0); // Batch size
        int C = X.dimension(1); // Number of channels
        int H = X.dimension(2); // Height of input image
        int W = X.dimension(3); // Width of input image

        int out_height = (H + 2 * padding - k_height) / stride + 1;
        int out_width = (W + 2 * padding - k_width) / stride + 1;

        Tensor<float, 2> out(C * k_height * k_width, N * out_height * out_width);

        Tensor<float, 4> padded_X(N, C, H + 2 * padding, W + 2 * padding);
        padded_X.setZero();
        padded_X.slice(array<int, 4>{0, 0, padding, padding}, array<int, 4>{N, C, H, W}) = X;

        int col_index = 0;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        for (int kh = 0; kh < k_height; ++kh) {
                            for (int kw = 0; kw < k_width; ++kw) {
                                int pad_h = h * stride + kh - padding;
                                int pad_w = w * stride + kw - padding;
                                if (pad_h >= 0 && pad_h < H && pad_w >= 0 && pad_w < W) {
                                    out(c * k_height * k_width + kh * k_width + kw, col_index) = padded_X(n, c, pad_h, pad_w);
                                } else {
                                    out(c * k_height * k_width + kh * k_width + kw, col_index) = 0;
                                }
                            }
                        }
                        ++col_index;
                    }
                }
            }
        }
        
        return out;
    }

    Tensor<float, 4> im2col_bw(const Tensor<float, 2>& grad_X_col, const array<int, 4>& X_shape,
                            int k_height, int k_width, int padding = 1, int stride = 1) {
        // Return gradient w.r.t. input tensor
        int N = X_shape[0]; // Batch size
        int C = X_shape[1]; // Number of channels
        int H = X_shape[2]; // Height of input image
        int W = X_shape[3]; // Width of input image

        int out_height = (H + 2 * padding - k_height) / stride + 1;
        int out_width = (W + 2 * padding - k_width) / stride + 1;

        Tensor<float, 6> grad_reshape(C, k_height, k_width, out_height, out_width, N);
        grad_reshape.setZero(); 

        Tensor<float, 2> grad_X_col_reshaped_orig = grad_X_col.reshape(array<int, 2>{C * k_height * k_width, N * out_height * out_width});
        array<int, 2> transpose_shuffling({1, 0});
        
        Tensor<float, 2> grad_X_col_reshaped = grad_X_col_reshaped_orig.shuffle(transpose_shuffling);

        for (int i = 0; i < C * k_height * k_width; ++i) {
            for (int j = 0; j < N * out_height * out_width; ++j) {
                int c = i / (k_height * k_width);
                int kh = (i % (k_height * k_width)) / k_width;
                int kw = i % k_width;

                int n = j / (out_height * out_width);
                int h = (j % (out_height * out_width)) / out_width;
                int w = j % out_width;

                grad_reshape(c, kh, kw, h, w, n) = grad_X_col_reshaped(i, j);
            }
        }

        Tensor<float, 4> grad_input(N, C, H + 2 * padding, W + 2 * padding);
        grad_input.setZero();

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < k_height; ++kh) {
                    for (int kw = 0; kw < k_width; ++kw) {
                        for (int h = 0; h < out_height; ++h) {
                            for (int w = 0; w < out_width; ++w) {
                                int pad_h = h * stride + kh - padding;
                                int pad_w = w * stride + kw - padding;

                                if (pad_h >= 0 && pad_h < H && pad_w >= 0 && pad_w < W) {
                                    grad_input(n, c, pad_h + padding, pad_w + padding) += grad_reshape(c, kh, kw, h, w, n);
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor<float, 4> grad_input_no_pad(N, C, H, W);
        grad_input_no_pad.setZero();
        grad_input_no_pad.slice(array<int, 4>{0, 0, 0, 0}, array<int, 4>{N, C, H, W}) = grad_input.slice(array<int, 4>{0, 0, padding, padding}, array<int, 4>{N, C, H, W});

        return grad_input_no_pad;
    }
}