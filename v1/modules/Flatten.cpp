#include "Flatten.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

Tensor<float, 2> Flatten::forward(const Tensor<float, 4>& x) {
    auto dims = x.dimensions();
    int N = dims[0];  // batch size
    int C = dims[1];  // channels
    int H = dims[2];  // height
    int W = dims[3];  // width

    shape = {N, C, H, W};

    Tensor<float, 2> output = x.reshape(Eigen::array<int, 2>{N, C * H * W});
    return output;
}

Tensor<float, 4> Flatten::backward(const Tensor<float, 2>& dloss) {
    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];

    Tensor<float, 4> grad_input = dloss.reshape(array<int, 4>{N, C, H, W});
    return grad_input;
}

