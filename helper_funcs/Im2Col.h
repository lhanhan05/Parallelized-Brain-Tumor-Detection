#include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;

namespace Im2Col {
    Tensor<float, 2> im2col(const Tensor<float, 4>& X, int k_height, int k_width, int padding, int stride);
    Tensor<float, 2> im2col_bw(const Tensor<float, 2>& grad_X_col, const DSizes<int, 4>& X_shape, int k_height, int k_width, int padding, int stride);
} 