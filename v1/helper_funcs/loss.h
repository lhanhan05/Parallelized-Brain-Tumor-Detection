#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <cmath>

// Mean Squared Error Loss Class
class MSELoss {
public:
    // Forward pass: Compute MSE Loss
    float forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y);

    // Backward pass: Compute gradient of MSE Loss
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y);
};

// Cross Entropy Loss Class
class CrossEntropyLoss {
public:
    // Forward pass: Compute Cross Entropy Loss
    float forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y);

    // Backward pass: Compute gradient of Cross Entropy Loss
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y);
};

#endif // LOSS_H
