#include "MaxPool.h"
#include <array>


using namespace Eigen;
using namespace Im2Col;


MaxPool::MaxPool(int filter_height, int filter_width, int stride) {
   k_height = filter_height;
   k_width = filter_width;
   this->stride = stride;
}

Tensor<float, 4> MaxPool::forward(const Tensor<float, 4>& inputs) {
   N = inputs.dimension(0);
   C = inputs.dimension(1);
   H = inputs.dimension(2);
   W = inputs.dimension(3);

   this->inputs = inputs;


   out_height = (H - k_height) / stride + 1;
   out_width = (W - k_width) / stride + 1;


   Tensor<float, 2> cols = im2col(inputs, k_height, k_width, 0, stride);


   Tensor<float, 6> cols_reshaped = cols.reshape(DSizes<int, 6>({k_height, k_width, C, out_height, out_width, N}));
   cols_reshaped = cols_reshaped.shuffle(array<int, 6>({1, 2, 0, 3, 4, 5}));


   Tensor<float, 5> max_forward = cols_reshaped.maximum(array<int, 1>({0})); // Result is 5D: {k_width, C, out_height, out_width, N}


   Tensor<float, 6> max_forward_broadcast = max_forward.reshape(DSizes<int, 6>({1, k_width, C, out_height, out_width, N}));
   grad_mask = (cols_reshaped == max_forward_broadcast.broadcast(DSizes<int, 6>({k_height, 1, 1, 1, 1, 1})));


   grad_mask = grad_mask.reshape(DSizes<int, 4>({N * C * out_height * out_width, k_height * k_width}));


   Tensor<float, 4> output = max_forward.reshape(DSizes<int, 4>({C, out_height, out_width, N}));
   output = output.shuffle(array<int, 4>({3, 0, 1, 2}));


   return output;
}


Tensor<float, 4> MaxPool::backward(const Tensor<float, 4>& dloss) {
    Tensor<float, 4> shuffled_dloss = dloss.shuffle(array<int, 4>({1, 2, 3, 0}));


    Tensor<float, 2> reshaped_dloss = shuffled_dloss.reshape(DSizes<int, 2>({C, out_height * out_width * N}));
    Tensor<float, 2> cols_repeat = reshaped_dloss.reshape(DSizes<int, 2>({C * k_height * k_width, out_height * out_width * N}));


    for (int i = 0; i < k_height * k_width; ++i) {
        cols_repeat.slice(array<int, 2>{i * C, 0}, array<int, 2>{C, out_height * out_width * N}) = reshaped_dloss;
    }


    Tensor<float, 2> mask_reshaped = grad_mask.reshape(DSizes<int, 2>({N * C * out_height * out_width, k_height * k_width}));


    DSizes<int, 4> dims({N, C, H, W});

    Tensor<float, 2> cols_output = cols_repeat * mask_reshaped;
    Tensor<float, 4> grad_input = im2col_bw(cols_output, dims, k_height, k_width, 0, stride);
    grad_input = grad_input.reshape(dims);


   return grad_input;
}