#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm> 
#include "Im2Col.h"

using namespace Eigen;
using namespace Im2Col;

class MaxPool {
public:
    int k_height, k_width, stride;
    int N, C, H, W;
    Tensor<float, 4> inputs; // Input tensor
    Tensor<float, 4> grad_mask; // Mask to track max positions
    int out_height, out_width; // Output dimensions

    MaxPool(int filter_height, int filter_width, int stride) {
        k_height = filter_height;
        k_width = filter_width;
        this->stride = stride;
    }

    Tensor<float, 4> forward(const Tensor<float, 4>& inputs) {
        // Extract input shape: [N, C, H, W]
        N = inputs.dimension(0); 
        C = inputs.dimension(1);
        H = inputs.dimension(2);
        W = inputs.dimension(3);

        this->inputs = inputs;

        out_height = (H - k_height) / stride + 1;
        out_width = (W - k_width) / stride + 1;

        Tensor<float, 2> cols = im2col(inputs, k_height, k_width, 0, stride); // Padding = 0

        Tensor<float, 6> cols_reshaped = cols.reshape(DSizes<int, 6>({k_height, k_width, C, out_height, out_width, N}));
        cols_reshaped = cols_reshaped.shuffle(array<int, 6>({1, 2, 0, 3, 4, 5}));

        Tensor<float, 2> max_forward = cols_reshaped.maximum(array<int, 1>({0}));

        grad_mask = (cols_reshaped == max_forward.reshape(DSizes<int, 1>({max_forward.dimension(0), 1})));

        Tensor<float, 4> output = max_forward.reshape(DSizes<int, 4>({C, out_height, out_width, N}));
        output = output.shuffle(array<int, 4>({3, 0, 1, 2}));
        
        return output;
    }

    Tensor<float, 4> backward(const Tensor<float, 4>& dloss) {
        Tensor<float, 2> cols = dloss.shuffle(array<int, 4>({1, 2, 3, 0}));
        cols = cols.reshape(DSizes<int, 2>({C, out_height * out_width * N}));

        Tensor<float, 2> cols_repeat = cols.replicate(DSizes<int, 2>({k_height * k_width, 1}));

        Tensor<float, 2> mask_reshape = grad_mask.shuffle(array<int, 4>({1, 0, 2, 3}));
        mask_reshape = mask_reshape.reshape(DSizes<int, 2>({cols_repeat.dimension(0), cols_repeat.dimension(1)}));

        Tensor<float, 2> grad_input = im2col_bw(cols_repeat * mask_reshape, DSizes<int, 4>({N, C, H, W}), k_height, k_width, 0, stride);

        return grad_input;
    }
};