#ifndef SIGMOID_H
#define SIGMOID_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>

using namespace Eigen;
class Sigmoid {
public:
   Sigmoid();

   Tensor<float, 4> forward(const Tensor<float, 4>& x);
   Tensor<float, 4> backward(const Tensor<float, 4>& grad_wrt_out);

private:
   Tensor<float, 4> forward_out; 
};

#endif // SIGMOID_H
