#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <numeric>

class Linear {
public:
    // Weights, biases, and helper matrices
    std::vector<std::vector<double>> W;  
    std::vector<std::vector<double>> b;  
    std::vector<std::vector<double>> Ones;  
    std::vector<std::vector<double>> Atemp;

    // Constructor to initialize weights and biases
    Linear(int in_features, int out_features);

    // Forward pass function
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& A);

    // Backward pass function
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& dLdZ);

private:
    // Debug flag (uncomment if needed)
    // bool debug;
};

#endif // LINEAR_H