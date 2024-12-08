#include "SoftMaxCrossEntropyLoss.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

class SoftMaxCrossEntropyLoss {
public:
    float forward(const Tensor<float, 2>& logits, const Tensor<float, 2>& labels) {
        // Transpose logits and labels
        Tensor<float, 2> logits_transposed = logits.transpose();
        this->labels = labels.transpose();

        // Compute softmax
        Tensor<float, 2> exp_logits = logits_transposed.exp();
        Tensor<float, 1> expsum = exp_logits.sum(0); // Sum along columns (axis=0)
        softmax = exp_logits / expsum.broadcast(logits_transposed.shape(0)); // Normalize along columns

        // Compute log(softmax)
        Tensor<float, 2> log_softmax = softmax.log();

        // Calculate cross-entropy loss
        Tensor<float, 2> loss_prod = log_softmax * labels;
        float loss_sum = -loss_prod.sum();

        // Predicted class labels (for computing contrastive loss)
        Tensor<int, 1> preds = softmax.argmax(0);  // Get the index of max in each column (class prediction)

        // Split predictions into 3 parts for contrastive loss
        int length = preds.shape(0);
        int split_size = length / 3;

        std::vector<Tensor<int, 1>> pred_parts;
        pred_parts.push_back(preds.slice(0, split_size));
        pred_parts.push_back(preds.slice(split_size, split_size));
        pred_parts.push_back(preds.slice(2 * split_size, length - 2 * split_size));

        std::vector<int> sims(length, 0);
        for (int i = 0; i < length; ++i) {
            std::unordered_map<int, int> element_count;
            for (auto& part : pred_parts) {
                int element = part(i);
                element_count[element]++;
            }
            sims[i] = std::max_element(element_count.begin(), element_count.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.second < b.second;
                })->second;
        }

        Tensor<float, 1> contrast_loss_section(sims.begin(), sims.end());
        contrast_loss_section = contrast_loss_section * -0.01f;
        Tensor<float, 1> contrast_loss = contrast_loss_section.tile(3);

        float contrast_sum = contrast_loss.sum();

        contrast = contrast_loss.reshape(labels.shape(0), 1) * labels;

        return 0.8f * loss_sum + 0.2f * contrast_sum;
    }

    Tensor<float, 2> backward() {
        Tensor<float, 2> softmax_back = softmax - labels.transpose();
        return 0.8f * softmax_back + 0.2f * contrast;
    }

    float getAccu() {
        Tensor<int, 1> preds = softmax.argmax(0);
        Tensor<int, 1> actuals = labels.argmax(0);
        int correctcount = (preds == actuals).sum();
        return static_cast<float>(correctcount) / preds.shape(0);
    }
};