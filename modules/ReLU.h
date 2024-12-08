#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;

class ReLU {
public:
    ReLU() : x_mult(Tensor<float, 4>()) {}
    Tensor<float, 4> forward(const Tensor<float, 4>& x);
    Tensor<float, 4> backward(const Tensor<float, 4>& grad_wrt_out);

private:
    Tensor<float, 4> x_mult;
};