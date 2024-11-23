#include <Eigen/Dense>

using namespace Eigen;

class Flatten {
public:
    Flatten() : N(0), Cin(0), Win(0) {}

    MatrixXd forward(const Tensor<double, 3> &A) {
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
        MatrixXd Z(N, flattened_size);

        for (int n = 0; n < N; ++n) {
            Map<MatrixXd> flattened(&A(n, 0, 0), 1, flattened_size);
            Z.row(n) = flattened;
        }

        return Z;
    }

    Tensor<double, 3> backward(const MatrixXd &dLdZ) {
        /*
        Argument:
            dLdZ (MatrixXd): (batch_size, in_channels * in_width)
        Return:
            dLdA (Tensor<double, 3>): (batch_size, in_channels, in_width)
        */

        Tensor<double, 3> dLdA(N, Cin, Win);

        for (int n = 0; n < N; ++n) {
            Map<const MatrixXd> reshaped(dLdZ.row(n).data(), Cin, Win);
            for (int c = 0; c < Cin; ++c) {
                for (int w = 0; w < Win; ++w) {
                    dLdA(n, c, w) = reshaped(c, w);
                }
            }
        }

        return dLdA;
    }

private:
    int N;    // Batch size
    int Cin;  // Input channels
    int Win;  // Input width
};
