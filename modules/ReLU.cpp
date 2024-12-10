#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "ReLU.h"

using namespace Eigen;
class ReLU {
public:
    ReLU() : x_mult(Tensor<float, 4>()) {}

    Tensor<float, 4> forward(const Tensor<float, 4>& x) {
        x_mult = (x > 0).cast<float>();
        return x * x_mult;
    }

    Tensor<float, 4> backward(const Tensor<float, 4>& grad_wrt_out) {
        return grad_wrt_out * x_mult;
    }

private:
    Tensor<float, 4> x_mult;
};