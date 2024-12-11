#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>
#include <algorithm>

class Identity {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
};

class Sigmoid {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
};

class Tanh {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
};

class ReLU {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
};

class GELU {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
    std::vector<std::vector<float>> Z;
};

class Softmax {
public:
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& Z);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dLdA);

private:
    std::vector<std::vector<float>> A;
};

#endif // ACTIVATION_H
