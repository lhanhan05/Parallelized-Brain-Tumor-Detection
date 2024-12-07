#include "flatten.h"

Flatten::Flatten() : N(0), Cin(0), Win(0) {}

Eigen::MatrixXd Flatten::forward(const Eigen::Tensor<double, 3> &A) {
    /*
    Argument:
        A (Tensor<double, 3>): (batch_size, in_channels, in_width)
    Return:
        Z (MatrixXd): (batch_size, in_channels * in_width)
    */

    N = A.dimension(0);
    Cin = A.dimension(1);
    Win = A.dimension(2);

    int flattened_size = Cin * Win;
    Eigen::MatrixXd Z(N, flattened_size);

    for (int n = 0; n < N; ++n) {
        int index = 0; // Index for flattened row in MatrixXd
        for (int c = 0; c < Cin; ++c) {
            for (int w = 0; w < Win; ++w) {
                Z(n, index) = A(n, c, w);
                index++;
            }
        }
    }

    return Z;
}

Eigen::Tensor<double, 3> Flatten::backward(const Eigen::MatrixXd &dLdZ) {
    /*
    Argument:
        dLdZ (MatrixXd): (batch_size, in_channels * in_width)
    Return:
        dLdA (Tensor<double, 3>): (batch_size, in_channels, in_width)
    */

    Eigen::Tensor<double, 3> dLdA(N, Cin, Win);

    for (int n = 0; n < N; ++n) {
        int index = 0;  // Track the index in the flattened row
        for (int c = 0; c < Cin; ++c) {
            for (int w = 0; w < Win; ++w) {
                dLdA(n, c, w) = dLdZ(n, index);
                index++;
            }
        }
    }

    return dLdA;
}
