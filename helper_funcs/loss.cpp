#include <iostream>
#include <cmath>
#include <vector>

class MSELoss {
public:
    // Forward pass for Mean Squared Error Loss
    float forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
        int N = A.size();
        int C = A[0].size();
        std::vector<std::vector<float>> Ones_N(N, std::vector<float>(1, 1.0f));
        std::vector<std::vector<float>> Ones_C(C, std::vector<float>(1, 1.0f));

        float mse = 0.0f;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                float se = (A[i][j] - Y[i][j]) * (A[i][j] - Y[i][j]);
                mse += se;
            }
        }
        mse /= (N * C);
        return mse;
    }

    // Backward pass for Mean Squared Error Loss
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
        int N = A.size();
        int C = A[0].size();
        std::vector<std::vector<float>> dLdA(N, std::vector<float>(C, 0.0f));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                dLdA[i][j] = 2 * (A[i][j] - Y[i][j]) / (N * C);
            }
        }

        return dLdA;
    }
};

class CrossEntropyLoss {
public:
    // Forward pass for Cross Entropy Loss
    float forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
        int N = A.size();
        int C = A[0].size();
        std::vector<std::vector<float>> Ones_N(N, std::vector<float>(1, 1.0f));
        std::vector<std::vector<float>> Ones_C(C, std::vector<float>(1, 1.0f));

        // Softmax calculation
        std::vector<std::vector<float>> softmax(N, std::vector<float>(C, 0.0f));
        for (int i = 0; i < N; ++i) {
            float sum_exp = 0.0f;
            for (int j = 0; j < C; ++j) {
                sum_exp += std::exp(A[i][j]);
            }
            for (int j = 0; j < C; ++j) {
                softmax[i][j] = std::exp(A[i][j]) / sum_exp;
            }
        }

        // Cross-entropy loss calculation
        float crossentropy = 0.0f;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                crossentropy -= Y[i][j] * std::log(softmax[i][j]);
            }
        }

        // Final loss
        float L = crossentropy / N;
        return L;
    }

    // Backward pass for Cross Entropy Loss
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
        int N = A.size();
        int C = A[0].size();
        std::vector<std::vector<float>> dLdA(N, std::vector<float>(C, 0.0f));

        // Softmax calculation
        std::vector<std::vector<float>> softmax(N, std::vector<float>(C, 0.0f));
        for (int i = 0; i < N; ++i) {
            float sum_exp = 0.0f;
            for (int j = 0; j < C; ++j) {
                sum_exp += std::exp(A[i][j]);
            }
            for (int j = 0; j < C; ++j) {
                softmax[i][j] = std::exp(A[i][j]) / sum_exp;
            }
        }

        // Gradient of the loss with respect to A
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                dLdA[i][j] = (softmax[i][j] - Y[i][j]) / N;
            }
        }

        return dLdA;
    }
};
