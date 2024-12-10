#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
using namespace Eigen;

class SoftMaxCrossEntropyLoss {
public:
    SoftMaxCrossEntropyLoss() = default;

    float forward(const Tensor<float, 2>& logits, const Tensor<float, 2>& labels);

    Tensor<float, 2> backward();

    Tensor<int, 1> getPreds();

    float getAccu();

private:
    Tensor<float, 2> softmax;      // Softmax values
    Tensor<float, 2> labels;       // True labels
    Tensor<float, 2> contrast;     // Contrast loss
    Tensor<int, 1> preds;
};