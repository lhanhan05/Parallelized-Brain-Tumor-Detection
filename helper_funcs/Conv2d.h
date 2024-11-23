// conv2d.h
#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include "resample.h"  // Include resample.h for Downsample2d class

// Class for 2D Convolution with a stride of 1
class Conv2d_stride1 {
public:
    Conv2d_stride1(int in_channels, int out_channels, int kernel_size);

    // Forward method
    std::vector<std::vector<std::vector<std::vector<double>>>> forward(
        const std::vector<std::vector<std::vector<std::vector<double>>>>& A);

    // Backward method
    std::vector<std::vector<std::vector<std::vector<double>>>> backward(
        const std::vector<std::vector<std::vector<std::vector<double>>>>& dLdZ);

    // Getters for gradients
    const std::vector<std::vector<std::vector<std::vector<double>>>>& get_dLdW() const;
    const std::vector<double>& get_dLdb() const;

private:
    int in_channels;
    int out_channels;
    int kernel_size;

    std::vector<std::vector<std::vector<std::vector<double>>>> W;  // Weights
    std::vector<double> b;                                          // Biases

    std::vector<std::vector<std::vector<std::vector<double>>>> dLdW; // Weight gradient
    std::vector<double> dLdb;                                         // Bias gradient

    std::vector<std::vector<std::vector<std::vector<double>>>> A;    // Input during forward pass
};

// Class for 2D Convolution with adjustable stride and padding
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding = 0);

    // Forward method
    std::vector<std::vector<std::vector<std::vector<double>>>> forward(
        const std::vector<std::vector<std::vector<std::vector<double>>>>& A);

    // Backward method
    std::vector<std::vector<std::vector<std::vector<double>>>> backward(
        const std::vector<std::vector<std::vector<std::vector<double>>>>& dLdZ);

private:
    int stride;
    int pad;
    
    Conv2d_stride1 conv2d_stride1;  // Convolution with stride of 1
    Downsample2d downsample2d;      // Downsampling class from resample.h
};

#endif  // CONV2D_H
