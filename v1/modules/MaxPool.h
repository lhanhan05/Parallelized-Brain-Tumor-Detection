#ifndef MAXPOOL_H
#define MAXPOOL_H


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include "../helper_funcs/Im2Col.h"
using namespace Eigen;


class MaxPool {
public:
   int k_height, k_width, stride;
   int N, C, H, W;
   Tensor<float, 4> inputs;  // Input tensor
   Tensor<float, 4> grad_mask;  // Mask to track max positions
   int out_height, out_width;  // Output dimensions


   MaxPool(int filter_height, int filter_width, int stride);


   Tensor<float, 4> forward(const Tensor<float, 4>& inputs);
   Tensor<float, 4> backward(const Tensor<float, 4>& dloss);

};


#endif // MAXPOOL_H
