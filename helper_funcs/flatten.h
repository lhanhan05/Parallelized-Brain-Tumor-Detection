#ifndef FLATTEN_H
#define FLATTEN_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;

class Flatten {
public:
    Flatten();
    
    MatrixXd forward(const Tensor<double, 3> &A);
    Tensor<double, 3> backward(const MatrixXd &dLdZ);

private:
    int N;    // Batch size
    int Cin;  // Input channels
    int Win;  // Input width
};

#endif
