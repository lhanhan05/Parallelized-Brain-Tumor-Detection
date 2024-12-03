#include <vector>
#include <numeric>

class Linear {
public:
    std::vector<std::vector<double>> W;  // Weights matrix
    std::vector<std::vector<double>> b;  // Bias matrix
    std::vector<std::vector<double>> Ones;  // Ones matrix for adding bias
    std::vector<std::vector<double>> Atemp;

    // bool debug;

    // Constructor to initialize weights and biases
    Linear(int in_features, int out_features) {
        W = std::vector<std::vector<double>>(out_features, std::vector<double>(in_features, 0.0));
        b = std::vector<std::vector<double>>(out_features, std::vector<double>(1, 0.0));
        // this->debug = debug;
    }

    // Forward pass function
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& A) {
        Atemp = A;
        int N = A.size();  // Number of samples
        Ones = std::vector<std::vector<double>>(N, std::vector<double>(1, 1.0));  // Fill Ones with 1's for each row

        // Initialize Z with zeros
        std::vector<std::vector<double>> Z(N, std::vector<double>(W.size(), 0.0));

        // Perform the matrix multiplication and addition
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < W.size(); ++j) {
                Z[i][j] = std::inner_product(A[i].begin(), A[i].end(), W[j].begin(), 0.0) + b[j][0];
            }
        }

        return Z;
    }

    // Backward pass function
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& dLdZ) {
        int N = dLdZ.size();
        int C1 = dLdZ[0].size();

        // Calculate dLdA
        std::vector<std::vector<double>> dLdA(N, std::vector<double>(W[0].size(), 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < W[0].size(); ++j) {
                for (int k = 0; k < C1; ++k) {
                    dLdA[i][j] += dLdZ[i][k] * W[k][j];
                }
            }
        }

        // Calculate dLdW and dLdb
        std::vector<std::vector<double>> dLdW(W.size(), std::vector<double>(W[0].size(), 0.0));
        std::vector<std::vector<double>> dLdb(b.size(), std::vector<double>(1, 0.0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C1; ++j) {
                for (int k = 0; k < W[0].size(); ++k) {
                    dLdW[j][k] += dLdZ[i][j] * Ones[i][0] * Atemp[i][k];
                }
                dLdb[j][0] += dLdZ[i][j] * Ones[i][0];
            }
        }

        // if (debug) {
        //     // Debugging output for dLdA
        // }

        return dLdA;
    }
};
