#include <tuple>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "modules/Conv.h"
#include "modules/ReLU.h"
#include "modules/LinearLayer.h"
#include "modules/MaxPool.h"
#include "modules/SoftMaxCrossEntropyLoss.h"
#include "modules/Flatten.h"

using namespace Eigen;

class Conv1dModel {
public:
    Conv1dModel(int out_dim = 4, 
            std::tuple<int, int, int> input_shape = {3, 32, 32}, 
            std::tuple<int, int, int> filter_shape = {1, 5, 5}) 
    {
        this->out_dim = out_dim;
        auto [conv_c, conv_w, conv_h] = input_shape;
        auto [conv_k_c, conv_k_w, conv_k_h] = filter_shape;

        int conv_padding = 2, conv_stride = 1;
        int pool_stride = 2;
        int pool_w = 2, pool_h = 2;

        conv_w_out = (conv_w + 2 * conv_padding - conv_k_w) / conv_stride + 1;
        conv_h_out = (conv_h + 2 * conv_padding - conv_k_h) / conv_stride + 1;

        pool_w_out = (conv_w_out - pool_w) / pool_stride + 1;
        pool_h_out = (conv_h_out - pool_h) / pool_stride + 1;

        conv = Conv(input_shape, filter_shape);
        relu = ReLU();
        maxpool = MaxPool(pool_w, pool_h, pool_stride);
        flatten = Flatten();
        linear = LinearLayer(conv_k_c * pool_w_out * pool_h_out, out_dim);
        loss = SoftMaxCrossEntropyLoss();
    }

    std::pair<float, std::vector<int>> forward(const Tensor<double, 4>& inputs, 
                                               const Tensor<double, 2>& y_labels) 
    {
        Tensor<double, 4> conv_out = conv.forward(inputs);
        Tensor<double, 4> acti_out = relu.forward(conv_out);
        Tensor<double, 4> maxpool_out = maxpool.forward(acti_out);
        Tensor<double, 3> flatten_out = flatten.forward(maxpool_out);
        Tensor<double, 2> linear_out = linear.forward(flatten_out);

        float loss_val = loss.forward(linear_out, y_labels);
        std::vector<int> preds = loss.get_predictions(linear_out);

        return {loss_val, preds};
    }

    void backward(const Tensor<double, 4>& inputs, 
                  const Tensor<double, 2>& y_labels) 
    {
        Tensor<double, 2> loss_grad = loss.backward(inputs, y_labels);

        Tensor<double, 3> linear_grad = linear.backward(loss_grad)[2];
        Tensor<double, 4> flatten_grad = flatten.backward(linear_grad);
        Tensor<double, 4> maxpool_grad = maxpool.backward(flatten_grad);
        Tensor<double, 4> relu_grad = relu.backward(maxpool_grad);
        conv.backward(relu_grad);
    }

    void update(double learning_rate, double momentum_coeff) {
        linear.update(learning_rate, momentum_coeff);
        conv.update(learning_rate, momentum_coeff);
    }

private:
    int out_dim;
    int conv_w_out, conv_h_out, pool_w_out, pool_h_out;
    Conv conv;
    ReLU relu;
    MaxPool maxpool;
    Flatten flatten;
    LinearLayer linear;
    SoftMaxCrossEntropyLoss loss;
};
