#include "ReLU.h"

using namespace Eigen;

Tensor<float, 4> ReLU::forward(const Tensor<float, 4>& x) {
    x_mult = (x > 0).cast<float>();
    return x * x_mult;
}

Tensor<float, 4> ReLU::backward(const Tensor<float, 4>& grad_wrt_out) {
    return grad_wrt_out * x_mult;
}
