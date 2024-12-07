#ifndef FLATTEN_H
#define FLATTEN_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class Flatten {
public:
    Flatten();
    
    Eigen::MatrixXd forward(const Eigen::Tensor<double, 3> &A);
    Eigen::Tensor<double, 3> backward(const Eigen::MatrixXd &dLdZ);

private:
    int N;    // Batch size
    int Cin;  // Input channels
    int Win;  // Input width
};

#endif
