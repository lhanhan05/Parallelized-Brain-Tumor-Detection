#include "loss.h"

// MSELoss: Forward pass
float MSELoss::forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
    int N = A.size();
    int C = A[0].size();
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

// MSELoss: Backward pass
std::vector<std::vector<float>> MSELoss::backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
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

// CrossEntropyLoss: Forward pass
float CrossEntropyLoss::forward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
    int N = A.size();
    int C = A[0].size();
    std::vector<std::vector<float>> softmax(N, std::vector<float>(C, 0.0f));

    // Softmax computation
    for (int i = 0; i < N; ++i) {
        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum_exp += std::exp(A[i][j]);
        }
        for (int j = 0; j < C; ++j) {
            softmax[i][j] = std::exp(A[i][j]) / sum_exp;
        }
    }

    // Cross-entropy loss computation
    float crossentropy = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            crossentropy -= Y[i][j] * std::log(softmax[i][j]);
        }
    }

    return crossentropy / N;
}

// CrossEntropyLoss: Backward pass
std::vector<std::vector<float>> CrossEntropyLoss::backward(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& Y) {
    int N = A.size();
    int C = A[0].size();
    std::vector<std::vector<float>> dLdA(N, std::vector<float>(C, 0.0f));

    // Softmax computation
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

    // Gradient computation
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            dLdA[i][j] = (softmax[i][j] - Y[i][j]) / N;
        }
    }

    return dLdA;
}
