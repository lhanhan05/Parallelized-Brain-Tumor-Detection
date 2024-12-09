#include "Conv1dModel.h"

Conv1dModel::Conv1dModel(int out_dim, 
                         std::tuple<int, int, int> input_shape, 
                         std::tuple<int, int, int> filter_shape)
    : out_dim(out_dim),
      conv(input_shape, filter_shape, 0),
      relu(),
      maxpool(2, 2, 2),
      flatten(),
      linear(std::get<0>(filter_shape) * std::get<1>(input_shape) * std::get<2>(input_shape) / 4, out_dim),
      loss() 
{
    auto [conv_c, conv_w, conv_h] = input_shape;
    auto [conv_k_c, conv_k_w, conv_k_h] = filter_shape;

    int conv_padding = 2, conv_stride = 1;
    int pool_stride = 2, pool_w = 2, pool_h = 2;

    conv_w_out = (conv_w + 2 * conv_padding - conv_k_w) / conv_stride + 1;
    conv_h_out = (conv_h + 2 * conv_padding - conv_k_h) / conv_stride + 1;

    pool_w_out = (conv_w_out - pool_w) / pool_stride + 1;
    pool_h_out = (conv_h_out - pool_h) / pool_stride + 1;
}

std::pair<float, Tensor<int, 1>> Conv1dModel::forward(const Tensor<float, 4>& inputs, 
                                                      const Tensor<float, 2>& y_labels) 
{
    Tensor<float, 4> conv_out = conv.forward(inputs);
    Tensor<float, 4> acti_out = relu.forward(conv_out);
    Tensor<float, 4> maxpool_out = maxpool.forward(acti_out);
    Tensor<float, 3> flatten_out = flatten.forward(maxpool_out);
    Tensor<float, 2> linear_out = linear.forward(flatten_out);

    float loss_val = loss.forward(linear_out, y_labels);
    Tensor<int, 1> preds = loss.getPreds();

    return {loss_val, preds};
}

void Conv1dModel::backward() {
    Tensor<float, 2> loss_grad = loss.backward();
    Tensor<float, 3> linear_grad = linear.backward(loss_grad)[0];
    Tensor<float, 4> flatten_grad = flatten.backward(linear_grad);
    Tensor<float, 4> maxpool_grad = maxpool.backward(flatten_grad);
    Tensor<float, 4> relu_grad = relu.backward(maxpool_grad);
    conv.backward(relu_grad);
}

void Conv1dModel::update(float learning_rate, float momentum_coeff) {
    linear.update(learning_rate, momentum_coeff);
    conv.update(learning_rate, momentum_coeff);
}
