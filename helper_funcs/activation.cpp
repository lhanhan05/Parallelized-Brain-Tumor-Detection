#include "activation.h"

// Identity
std::vector<std::vector<float>> Identity::forward(const std::vector<std::vector<float>>& Z) {
    A = Z;
    return A;
}

std::vector<std::vector<float>> Identity::backward(const std::vector<std::vector<float>>& dLdA) {
    std::vector<std::vector<float>> dLdZ(A.size(), std::vector<float>(A[0].size(), 1.0));
    for (size_t i = 0; i < dLdA.size(); ++i) {
        for (size_t j = 0; j < dLdA[0].size(); ++j) {
            dLdZ[i][j] = dLdA[i][j] * dLdZ[i][j];
        }
    }
    return dLdZ;
}

// Sigmoid
std::vector<std::vector<float>> Sigmoid::forward(const std::vector<std::vector<float>>& Z) {
    A.resize(Z.size(), std::vector<float>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); ++i) {
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] = 1.0 / (1.0 + std::exp(-Z[i][j]));
        }
    }
    return A;
}

std::vector<std::vector<float>> Sigmoid::backward(const std::vector<std::vector<float>>& dLdA) {
    std::vector<std::vector<float>> dLdZ(dLdA.size(), std::vector<float>(dLdA[0].size()));
    for (size_t i = 0; i < dLdA.size(); ++i) {
        for (size_t j = 0; j < dLdA[0].size(); ++j) {
            dLdZ[i][j] = dLdA[i][j] * (A[i][j] * (1 - A[i][j]));
        }
    }
    return dLdZ;
}

// Tanh
std::vector<std::vector<float>> Tanh::forward(const std::vector<std::vector<float>>& Z) {
    A.resize(Z.size(), std::vector<float>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); ++i) {
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] = std::tanh(Z[i][j]);
        }
    }
    return A;
}

std::vector<std::vector<float>> Tanh::backward(const std::vector<std::vector<float>>& dLdA) {
    std::vector<std::vector<float>> dLdZ(dLdA.size(), std::vector<float>(dLdA[0].size()));
    for (size_t i = 0; i < dLdA.size(); ++i) {
        for (size_t j = 0; j < dLdA[0].size(); ++j) {
            dLdZ[i][j] = dLdA[i][j] * (1 - A[i][j] * A[i][j]);
        }
    }
    return dLdZ;
}

// ReLU
std::vector<std::vector<float>> ReLU::forward(const std::vector<std::vector<float>>& Z) {
    A.resize(Z.size(), std::vector<float>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); ++i) {
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] = std::max(0.0f, Z[i][j]);
        }
    }
    return A;
}

std::vector<std::vector<float>> ReLU::backward(const std::vector<std::vector<float>>& dLdA) {
    std::vector<std::vector<float>> dLdZ(dLdA.size(), std::vector<float>(dLdA[0].size()));
    for (size_t i = 0; i < dLdA.size(); ++i) {
        for (size_t j = 0; j < dLdA[0].size(); ++j) {
            dLdZ[i][j] = dLdA[i][j] * (A[i][j] > 0 ? 1.0f : 0.0f);
        }
    }
    return dLdZ;
}

// GELU
std::vector<std::vector<float>> GELU::forward(const std::vector<std::vector<float>>& Z) {
    A.resize(Z.size(), std::vector<float>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); ++i) {
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] = 0.5 * Z[i][j] * (1 + std::erf(Z[i][j] / std::sqrt(2)));
        }
    }
    return A;
}

std::vector<std::vector<float>> GELU::backward(const std::vector<std::vector<float>>& dLdA) {
    std::vector<std::vector<float>> dLdZ(dLdA.size(), std::vector<float>(dLdA[0].size()));
    for (size_t i = 0; i < dLdA.size(); ++i) {
        for (size_t j = 0; j < dLdA[0].size(); ++j) {
            float dAdZ = 0.5 * (1 + std::erf(Z[i][j] / std::sqrt(2))) + 
                         (Z[i][j] / std::sqrt(2 * M_PI)) * std::exp(-Z[i][j] * Z[i][j] / 2);
            dLdZ[i][j] = dLdA[i][j] * dAdZ;
        }
    }
    return dLdZ;
}

// Softmax
std::vector<std::vector<float>> Softmax::forward(const std::vector<std::vector<float>>& Z) {
    A.resize(Z.size(), std::vector<float>(Z[0].size()));
    for (size_t i = 0; i < Z.size(); ++i) {
        float max_val = *std::max_element(Z[i].begin(), Z[i].end());
        float sum = 0.0f;
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] = std::exp(Z[i][j] - max_val);
            sum += A[i][j];
        }
        for (size_t j = 0; j < Z[0].size(); ++j) {
            A[i][j] /= sum;
        }
    }
    return A;
}

std::vector<std::vector<float>> Softmax::backward(const std::vector<std::vector<float>>& dLdA) {
    size_t N = dLdA.size();
    size_t C = dLdA[0].size();
    std::vector<std::vector<float>> dLdZ(N, std::vector<float>(C));

    for (size_t i = 0; i < N; ++i) {
        std::vector<std::vector<float>> J(C, std::vector<float>(C, 0.0f));
        for (size_t m = 0; m < C; ++m) {
            for (size_t n = 0; n < C; ++n) {
                J[m][n] = (m == n) ? A[i][m] * (1 - A[i][m]) : -A[i][m] * A[i][n];
            }
        }
        for (size_t j = 0; j < C; ++j) {
            dLdZ[i][j] = 0.0f;
            for (size_t k = 0; k < C; ++k) {
                dLdZ[i][j] += dLdA[i][k] * J[k][j];
            }
        }
    }

    return dLdZ;
}
