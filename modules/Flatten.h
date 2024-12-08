#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Flatten {
public:
    Flatten() = default;

    Tensor<float, 2> forward(const Tensor<float, 4>& x);
    Tensor<float, 4> backward(const Tensor<float, 2>& dloss);

private:
    std::array<int, 4> shape;
};