#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Conv {
public:
    // Constructor for the Convolution Layer
    Conv(const std::tuple<int, int, int>& input_shape, 
         const std::tuple<int, int, int>& filter_shape, 
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
    Tensor<float, 4> forward(const Tensor<float, 4>& inputs, int stride = 1, int pad = 2)
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
    std::vector<Tensor<float, 4>> backward(const Tensor<float, 4>& dloss)
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
        Tensor<float, 4> grad_inputs = im2col_bw(mul_output, array<int, 4>{N, C, H, W}, k_height, k_width, pad, stride);

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

    void update(float learning_rate = 0.01, float momentum_coeff = 0.5)
    {
        weights_momentum = momentum_coeff * weights_momentum + grad_weights / N;
        biases_momentum = momentum_coeff * biases_momentum + grad_biases / N;

        weights -= learning_rate * weights_momentum;
        biases -= learning_rate * biases_momentum;
    }

    std::pair<Tensor<float, 4>, Tensor<float, 1>> get_wb_conv() const
    {
        return {weights, biases};
    }

private:
    int num_filters, C, H, W, k_height, k_width, N;
    int out_width, out_height, pad, stride;
    Tensor<float, 4> weights, weights_momentum, grad_weights;
    Tensor<float, 1> biases, biases_momentum, grad_biases;
    Tensor<float, 4> X;
    Tensor<float, 4> grad_inputs;
    Tensor<float, 2> weights_reshape;

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

    Tensor<float, 4> im2col_bw(const Tensor<float, 2>& grad_X_col, const Eigen::array<int, 4>& X_shape,
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

        Tensor<float, 2> grad_X_col_reshaped_orig = grad_X_col.reshape(Eigen::array<int, 2>{C * k_height * k_width, N * out_height * out_width});
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


};