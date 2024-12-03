#ifndef RESAMPLING_H
#define RESAMPLING_H

#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;

// Class declarations
class Upsample1d {
public:
    explicit Upsample1d(int upsampling_factor);
    Tensor<double, 3> forward(const Tensor<double, 3> &A);
    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ);

private:
    int upsampling_factor;
};

class Downsample1d {
public:
    explicit Downsample1d(int downsampling_factor);
    Tensor<double, 3> forward(const Tensor<double, 3> &A);
    Tensor<double, 3> backward(const Tensor<double, 3> &dLdZ);

private:
    int downsampling_factor;
    int W_in;
};

class Upsample2d {
public:
    explicit Upsample2d(int upsampling_factor);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int upsampling_factor;
};

class Downsample2d {
public:
    explicit Downsample2d(int downsampling_factor);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int downsampling_factor;
    int W_in;
    int H_in;
};

#endif // RESAMPLING_H
