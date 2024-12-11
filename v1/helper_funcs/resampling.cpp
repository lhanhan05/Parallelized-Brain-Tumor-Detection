#include "resampling.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Upsample1d {
public:
    explicit Upsample1d(int upsampling_factor)
        : upsampling_factor(upsampling_factor) {}

    Tensor<double, 3> forward(const Tensor<double, 3> &A) {
        /*
        Argument:
            A (Tensor<double, 3>): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor<double, 3>): (batch_size, in_channels, output_width)
        */
        int W_in = A.dimension(2);
        int W_out = upsampling_factor * (W_in - 1) + 1;
        Tensor<double, 3> Z(A.dimension(0), A.dimension(1), W_out);
        Z.setZero();

        for (int n = 0; n < A.dimension(0); ++n) {
            for (int c = 0; c < A.dimension(1); ++c) {
                for (int w = 0; w < W_in; ++w) {
                    Z(n, c, w * upsampling_factor) = A(n, c, w);
                }
            }
        }
        return Z;
    }

    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ) {
        /*
        Argument:
            dLdZ (Tensor<double, 3>): (batch_size, in_channels, output_width)
        Return:
            dLdA (Tensor<double, 3>): (batch_size, in_channels, input_width)
        */
        int W_out = dLdZ.dimension(2);
        int W_in = ((W_out - 1) / upsampling_factor) + 1;
        Tensor<double, 3> dLdA(dLdZ.dimension(0), dLdZ.dimension(1), W_in);
        dLdA.setZero();

        for (int n = 0; n < dLdZ.dimension(0); ++n) {
            for (int c = 0; c < dLdZ.dimension(1); ++c) {
                for (int w = 0; w < W_in; ++w) {
                    dLdA(n, c, w) = dLdZ(n, c, w * upsampling_factor);
                }
            }
        }
        return dLdA;
    }

private:
    int upsampling_factor;
};

class Downsample1d {
public:
    explicit Downsample1d(int downsampling_factor)
        : downsampling_factor(downsampling_factor), W_in(0) {}

    Tensor<double, 3> forward(const Tensor<double, 3> &A) {
        /*
        Argument:
            A (Tensor<double, 3>): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor<double, 3>): (batch_size, in_channels, output_width)
        */
        W_in = A.dimension(2);
        int W_out = W_in / downsampling_factor;
        Tensor<double, 3> Z(A.dimension(0), A.dimension(1), W_out);

        for (int n = 0; n < A.dimension(0); ++n) {
            for (int c = 0; c < A.dimension(1); ++c) {
                for (int w = 0; w < W_out; ++w) {
                    Z(n, c, w) = A(n, c, w * downsampling_factor);
                }
            }
        }
        return Z;
    }

    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ) {
        /*
        Argument:
            dLdZ (Tensor<double, 3>): (batch_size, in_channels, output_width)
        Return:
            dLdA (Tensor<double, 3>): (batch_size, in_channels, input_width)
        */
        Tensor<double, 3> dLdA(dLdZ.dimension(0), dLdZ.dimension(1), W_in);
        dLdA.setZero();

        for (int n = 0; n < dLdZ.dimension(0); ++n) {
            for (int c = 0; c < dLdZ.dimension(1); ++c) {
                for (int w = 0; w < dLdZ.dimension(2); ++w) {
                    dLdA(n, c, w * downsampling_factor) = dLdZ(n, c, w);
                }
            }
        }
        return dLdA;
    }

private:
    int downsampling_factor;
    int W_in;
};

class Upsample2d {
public:
    explicit Upsample2d(int upsampling_factor)
        : upsampling_factor(upsampling_factor) {}

    Tensor<double, 4> forward(const Tensor<double, 4> &A) {
        /*
        Argument:
            A (Tensor<double, 4>): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (Tensor<double, 4>): (batch_size, in_channels, output_height, output_width)
        */
        int H_in = A.dimension(2), W_in = A.dimension(3);
        int H_out = upsampling_factor * (H_in - 1) + 1;
        int W_out = upsampling_factor * (W_in - 1) + 1;

        Tensor<double, 4> Z(A.dimension(0), A.dimension(1), H_out, W_out);
        Z.setZero();

        for (int n = 0; n < A.dimension(0); ++n) {
            for (int c = 0; c < A.dimension(1); ++c) {
                for (int h = 0; h < H_in; ++h) {
                    for (int w = 0; w < W_in; ++w) {
                        Z(n, c, h * upsampling_factor, w * upsampling_factor) = A(n, c, h, w);
                    }
                }
            }
        }
        return Z;
    }

    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ) {
        /*
        Argument:
            dLdZ (Tensor<double, 4>): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (Tensor<double, 4>): (batch_size, in_channels, input_height, input_width)
        */
        int H_out = dLdZ.dimension(2), W_out = dLdZ.dimension(3);
        int H_in = ((H_out - 1) / upsampling_factor) + 1;
        int W_in = ((W_out - 1) / upsampling_factor) + 1;

        Tensor<double, 4> dLdA(dLdZ.dimension(0), dLdZ.dimension(1), H_in, W_in);
        dLdA.setZero();

        for (int n = 0; n < dLdZ.dimension(0); ++n) {
            for (int c = 0; c < dLdZ.dimension(1); ++c) {
                for (int h = 0; h < H_in; ++h) {
                    for (int w = 0; w < W_in; ++w) {
                        dLdA(n, c, h, w) = dLdZ(n, c, h * upsampling_factor, w * upsampling_factor);
                    }
                }
            }
        }
        return dLdA;
    }

private:
    int upsampling_factor;
};

class Downsample2d():

    def __init__(self, downsampling_factor):
        """
        Initialize the downsample layer with a given downsampling factor.
        """
        self.downsampling_factor = downsampling_factor
        self.W_in = None
        self.H_in = None

    def forward(self, A):
        """
        Perform the downsampling operation.

        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)

        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.W_in = A.shape[3]  # Store the input width
        self.H_in = A.shape[2]  # Store the input height
        k = self.downsampling_factor  # Downsampling factor
        Z = A[..., ::k, ::k]  # Extract every k-th element along height and width

        return Z

    def backward(self, dLdZ):
        """
        Backpropagate through the downsampling operation.

        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)

        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        k = self.downsampling_factor  # Downsampling factor
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.H_in, self.W_in))  # Initialize gradient
        dLdA[..., ::k, ::k] = dLdZ  # Spread the gradients back to the corresponding elements

        return dLdA

