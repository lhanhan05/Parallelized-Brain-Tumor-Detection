#include "pool.h" // Include the header file

// MaxPool2d_stride1 Implementation
MaxPool2d_stride1::MaxPool2d_stride1(int kernel) : kernel(kernel) {}

Tensor<double, 4> MaxPool2d_stride1::forward(const Tensor<double, 4> &A) {
    this->A = A;
    int N = A.dimension(0);
    int Cin = A.dimension(1);
    int Hin = A.dimension(2);
    int Win = A.dimension(3);
    int k = this->kernel;
    int Wout = Win - k + 1;
    int Hout = Hin - k + 1;
    int Cout = Cin;

    Tensor<double, 4> Z(N, Cout, Hout, Wout);
    Z.setZero();

    for (int n = 0; n < N; ++n) {
        for (int ci = 0; ci < Cin; ++ci) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    Z(n, ci, h, w) = A(n, ci, h, w); // Default is max, just for the skeleton
                }
            }
        }
    }
    return Z;
}

Tensor<double, 4> MaxPool2d_stride1::backward(const Tensor<double, 4> &dLdZ) {
    int N = dLdZ.dimension(0);
    int Cout = dLdZ.dimension(1);
    int Hout = dLdZ.dimension(2);
    int Wout = dLdZ.dimension(3);
    int k = this->kernel;

    int Win = Wout + k - 1;
    int Hin = Hout + k - 1;
    int Cin = Cout;

    Tensor<double, 4> dLdA(N, Cin, Hin, Win);
    dLdA.setZero();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cin; ++c) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    // You would add code here to handle the actual gradient calculation
                }
            }
        }
    }
    return dLdA;
}

// MeanPool2d_stride1 Implementation
MeanPool2d_stride1::MeanPool2d_stride1(int kernel) : kernel(kernel) {}

Tensor<double, 4> MeanPool2d_stride1::forward(const Tensor<double, 4> &A) {
    this->A = A;
    int N = A.dimension(0);
    int Cin = A.dimension(1);
    int Hin = A.dimension(2);
    int Win = A.dimension(3);
    int k = this->kernel;
    int Wout = Win - k + 1;
    int Hout = Hin - k + 1;
    int Cout = Cin;

    Tensor<double, 4> Z(N, Cout, Hout, Wout);
    Z.setZero();

    for (int n = 0; n < N; ++n) {
        for (int ci = 0; ci < Cin; ++ci) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    Z(n, ci, h, w) = A(n, ci, h, w); // Default is mean, just for skeleton
                }
            }
        }
    }
    return Z;
}

Tensor<double, 4> MeanPool2d_stride1::backward(const Tensor<double, 4> &dLdZ) {
    int N = dLdZ.dimension(0);
    int Cout = dLdZ.dimension(1);
    int Hout = dLdZ.dimension(2);
    int Wout = dLdZ.dimension(3);
    int k = this->kernel;

    int Win = Wout + k - 1;
    int Hin = Hout + k - 1;
    int Cin = Cout;

    Tensor<double, 4> dLdA(N, Cin, Hin, Win);
    dLdA.setZero();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cin; ++c) {
            for (int h = 0; h < Hout; ++h) {
                for (int w = 0; w < Wout; ++w) {
                    // Backpropagate the mean gradients
                }
            }
        }
    }
    return dLdA;
}

// MaxPool2d Implementation
MaxPool2d::MaxPool2d(int kernel, int stride)
    : kernel(kernel), stride(stride),
      maxpool2d_stride1(kernel), downsample2d(stride) {}

Tensor<double, 4> MaxPool2d::forward(const Tensor<double, 4> &A) {
    Tensor<double, 4> mask = maxpool2d_stride1.forward(A);
    return downsample2d.forward(mask);
}

Tensor<double, 4> MaxPool2d::backward(const Tensor<double, 4> &dLdZ) {
    Tensor<double, 4> mask = downsample2d.backward(dLdZ);
    return maxpool2d_stride1.backward(mask);
}

// MeanPool2d Implementation
MeanPool2d::MeanPool2d(int kernel, int stride)
    : kernel(kernel), stride(stride),
      meanpool2d_stride1(kernel), downsample2d(stride) {}

Tensor<double, 4> MeanPool2d::forward(const Tensor<double, 4> &A) {
    Tensor<double, 4> mask = meanpool2d_stride1.forward(A);
    return downsample2d.forward(mask);
}

Tensor<double, 4> MeanPool2d::backward(const Tensor<double, 4> &dLdZ) {
    Tensor<double, 4> mask = downsample2d.backward(dLdZ);
    return meanpool2d_stride1.backward(mask);
}

