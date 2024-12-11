#include "Sigmoid.h"

Sigmoid::Sigmoid() {}

Tensor<float, 4> Sigmoid::forward(const Tensor<float, 4>& x) {
   Tensor<float, 4> exp_part = (-1 * x).exp();
   Tensor<float, 4> sum_part = exp_part + Tensor<float, 4>::Constant(1.0f);  
   forward_out = sum_part.inverse();
   return forward_out;
}


Tensor<float, 4> Sigmoid::backward(const Tensor<float, 4>& grad_wrt_out) {
   Tensor<float, 4> grad_input = forward_out * (1 - forward_out);
   return grad_input * grad_wrt_out;
}
