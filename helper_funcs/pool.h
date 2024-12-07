#ifndef POOL_H
#define POOL_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "resampling.h" // Include the resampling header

using namespace Eigen;

// MaxPool2d_stride1 class
class MaxPool2d_stride1 {
public:
    explicit MaxPool2d_stride1(int kernel);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int kernel;
    Tensor<double, 4> A;  // Store input for backward pass
};

// MeanPool2d_stride1 class
class MeanPool2d_stride1 {
public:
    explicit MeanPool2d_stride1(int kernel);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int kernel;
    Tensor<double, 4> A;
};

// MaxPool2d class
class MaxPool2d {
public:
    explicit MaxPool2d(int kernel, int stride);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int kernel, stride;
    MaxPool2d_stride1 maxpool2d_stride1;
    Downsample2d downsample2d;
};

// MeanPool2d class
class MeanPool2d {
public:
    explicit MeanPool2d(int kernel, int stride);
    Tensor<double, 4> forward(const Tensor<double, 4> &A);
    Tensor<double, 4> backward(const Tensor<double, 4> &dLdZ);

private:
    int kernel, stride;
    MeanPool2d_stride1 meanpool2d_stride1;
    Downsample2d downsample2d;
};

#endif // POOL_H
