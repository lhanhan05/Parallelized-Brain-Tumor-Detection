#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include "Sigmoid.h"

using namespace Eigen;

Sigmoid::Sigmoid() {}

Tensor<float, 4> Sigmoid::forward(const Tensor<float, 4>& x) {
    forward_out = ((-1*x).exp() + 1).inverse();
    return forward_out;
}

Tensor<float, 4> Sigmoid::backward(const Tensor<float, 4>& grad_wrt_out) {
    Tensor<float, 4> grad_input = forward_out * (1 - forward_out);
    return grad_input * grad_wrt_out;
}