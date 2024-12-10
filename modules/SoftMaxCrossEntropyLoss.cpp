#include "SoftMaxCrossEntropyLoss.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

float SoftMaxCrossEntropyLoss::forward(const Tensor<float, 2>& logits, const Tensor<float, 2>& labels) {
    // Transpose logits and labels
    Tensor<float, 2> logits_transposed = logits.shuffle(array<int, 2>({1, 0}));
    this->labels = labels.shuffle(array<int, 2>({1, 0}));

    // Compute softmax
    Tensor<float, 2> exp_logits = logits_transposed.exp();
    Tensor<float, 1> expsum = exp_logits.sum(0); // Sum along columns (axis=0)
    softmax = exp_logits / expsum.broadcast(logits_transposed.dimensions()[0]); // Normalize along columns

    // Compute log(softmax)
    Tensor<float, 2> log_softmax = softmax.log();

    // Calculate cross-entropy loss
    Tensor<float, 2> loss_prod = log_softmax * labels;
    float loss_sum = 0;

    for (int i = 0; i < loss_prod.dimension(0); ++i) {
        for (int j = 0; j < loss_prod.dimension(1); ++j) {
            loss_sum += loss_prod(i, j);
        }
    }

    // Predicted class labels (for computing contrastive loss)
    preds = softmax.argmax(0); 

    // Split predictions into 3 parts for contrastive loss
    int length = preds.dimensions()[0];
    int split_size = length / 3;

    std::vector<Tensor<int, 1>> pred_parts;
    pred_parts.push_back(preds.slice(array<Index, 1>{0}, array<Index, 1>{split_size}));
    pred_parts.push_back(preds.slice(array<Index, 1>{split_size}, array<Index, 1>{split_size}));
    pred_parts.push_back(preds.slice(array<Index, 1>{2 * split_size}, array<Index, 1>{length - 2 * split_size}));

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

    TensorMap<Tensor<int, 1>> contrast_loss_section(sims.data(), sims.size());
    contrast_loss_section = contrast_loss_section.cast<float>();
    contrast_loss_section = contrast_loss_section * -0.01f;
    int repeat_count = 3;
    Tensor<float, 1> contrast_loss(contrast_loss_section.dimension(0)*repeat_count);
    
    float contrast_sum = 0;
    for (int i = 0; i < repeat_count; ++i) {
        for (int j = 0; j < contrast_loss_section.dimension(0); ++j) {
            contrast_loss(i * contrast_loss_section.dimension(0) + j) = contrast_loss_section(j);
            contrast_sum += contrast_loss_section(j);
        }
    }

    contrast_loss = contrast_loss.reshape(DSizes<int, 1>(labels.dimension(0), 1)); 
    contrast = contrast_loss * labels;
    return 0.8f * loss_sum + 0.2f * contrast_sum;
}

Tensor<int, 1> SoftMaxCrossEntropyLoss::getPreds(){
    return preds;
}

Tensor<float, 2> SoftMaxCrossEntropyLoss::backward() {
    Tensor<float, 2> softmax_back = softmax - labels.shuffle(array<int, 2>({1, 0}));
    return 0.8f * softmax_back + 0.2f * contrast;
}

float SoftMaxCrossEntropyLoss::getAccu() {
    Tensor<int, 1> preds = softmax.argmax(0);
    Tensor<int, 1> actuals = labels.argmax(0);
    int correctcount = getCorrectCount(preds, actuals);
    return static_cast<float>(correctcount) / preds.dimensions()[0];
}

int getCorrectCount(Tensor<int, 1> preds, Tensor<int, 1> actuals){
    int count = 0;
    for (int i = 0; i < preds.dimension(0); ++i) {
        if(preds[i] == actuals[i]){
            count += 1;
        }
    }
    return count;
}