#ifndef CONV1DMODEL_H
#define CONV1DMODEL_H

#include <tuple>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../modules/Conv.h"
#include "../modules/ReLU.h"
#include "../modules/MaxPool.h"
#include "../modules/Flatten.h"
#include "../modules/LinearLayer.h"
#include "../modules/SoftMaxCrossEntropyLoss.h"
#include "../modules/Sigmoid.h"

using namespace Eigen;

class Conv1dModel {
public:
    Conv1dModel(int out_dim = 4, 
                std::tuple<int, int, int> input_shape = {3, 32, 32}, 
                std::tuple<int, int, int> filter_shape = {1, 5, 5});

    std::pair<float, Tensor<int, 1>> forward(const Tensor<float, 4>& inputs, 
                                             const Tensor<int, 2>& y_labels);

    void backward();

    void update(float learning_rate, float momentum_coeff);

private:
    int out_dim, conv_w_out, conv_h_out, pool_w_out, pool_h_out;
    Conv conv;
    ReLU relu;
    Sigmoid sigmoid;
    MaxPool maxpool;
    Flatten flatten;
    LinearLayer linear;
    SoftMaxCrossEntropyLoss loss;
};

#endif
