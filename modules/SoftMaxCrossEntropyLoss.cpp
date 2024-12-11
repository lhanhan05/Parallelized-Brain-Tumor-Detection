#include "SoftMaxCrossEntropyLoss.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

template <typename T>
Tensor<int, 1> manualArgmax(const Tensor<T, 2>& input) {
    Tensor<int, 1> argmaxOutput(input.dimension(0));

    for (int row = 0; row < input.dimension(0); ++row) {
        float max_value = input(row, 0);
        int max_index = 0;
        for (int col = 1; col < input.dimension(1); ++col) {
            if (input(row, col) > max_value) {
                max_value = input(row, col);
                max_index = col;
            }
        }
        argmaxOutput(row) = max_index;
    }
    return argmaxOutput;
}

float SoftMaxCrossEntropyLoss::forward(const Tensor<float, 2>& logits, const Tensor<int, 2>& labels) {
    // Transpose logits and labels
    Tensor<float, 2> logits_transposed = logits.shuffle(array<int, 2>({1, 0}));
    this->labels = labels.shuffle(array<int, 2>({1, 0}));

    // Compute softmax
    Tensor<float, 2> exp_logits = logits_transposed.exp();
    array<int, 1> dims = {0};
    Tensor<float, 1> expsum = exp_logits.sum(dims); // Sum along columns (axis=0)
    array<int, 2> logit_dims = {(int)exp_logits.dimension(0), 1};

    Tensor<float, 2> expsum_reshaped(exp_logits.dimension(0), exp_logits.dimension(1));

    for (int col = 0; col < exp_logits.dimension(1); ++col) {
        for (int row = 0; row < exp_logits.dimension(0); ++row) {
            expsum_reshaped(row, col) = expsum(col);
        }
    }
    softmax = exp_logits / expsum_reshaped;

    // Compute log(softmax)
    Tensor<float, 2> log_softmax = softmax.log();

    // Calculate cross-entropy loss
    Tensor<float, 2> loss_prod = log_softmax * labels.cast<float>();
    float loss_sum = 0;

    for (int i = 0; i < loss_prod.dimension(0); ++i) {
        for (int j = 0; j < loss_prod.dimension(1); ++j) {
            loss_sum += loss_prod(i, j);
        }
    }

    // Predicted class labels (for computing contrastive loss)
    preds = manualArgmax(softmax); 
    return loss_sum;

    // Split predictions into 3 parts for contrastive loss
    // int length = preds.dimensions()[0];
    // int split_size = length / 3;

    // std::vector<Tensor<int, 1>> pred_parts;
    // pred_parts.push_back(preds.slice(array<Index, 1>{0}, array<Index, 1>{split_size}));
    // pred_parts.push_back(preds.slice(array<Index, 1>{split_size}, array<Index, 1>{split_size}));
    // pred_parts.push_back(preds.slice(array<Index, 1>{2 * split_size}, array<Index, 1>{length - 2 * split_size}));

    // std::vector<int> sims(length, 0);
    // for (int i = 0; i < length; ++i) {
    //     std::unordered_map<int, int> element_count;
    //     for (auto& part : pred_parts) {
    //         int element = part(i);
    //         element_count[element]++;
    //     }
    //     sims[i] = std::max_element(element_count.begin(), element_count.end(),
    //         [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    //             return a.second < b.second;
    //         })->second;
    // }

    // TensorMap<Tensor<int, 1>> contrast_loss_section_int(sims.data(), sims.size());
    // Tensor<float, 1> contrast_loss_section = contrast_loss_section_int.cast<float>();
    // contrast_loss_section = contrast_loss_section * -0.01f;
    // int repeat_count = 3;
    // Tensor<float, 2> contrast_loss(contrast_loss_section.dimension(0)*repeat_count, 1);
    
    // float contrast_sum = 0;
    // for (int i = 0; i < repeat_count; ++i) {
    //     for (int j = 0; j < contrast_loss_section.dimension(0); ++j) {
    //         contrast_loss(i * contrast_loss_section.dimension(0) + j, 0) = contrast_loss_section(j);
    //         contrast_sum += contrast_loss_section(j);
    //     }
    // }

    // contrast_loss = contrast_loss.reshape(DSizes<int, 1>(labels.dimension(0), 1)); 
    // contrast = contrast_loss * labels;
    // return 0.8f * loss_sum + 0.2f * contrast_sum;
}

Tensor<int, 1> SoftMaxCrossEntropyLoss::getPreds(){
    return preds;
}

Tensor<float, 2> SoftMaxCrossEntropyLoss::backward() {
    Tensor<float, 2> softmax_back = softmax - labels.cast<float>();
    return softmax_back;
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

float SoftMaxCrossEntropyLoss::getAccu() {
    Tensor<int, 1> actuals = manualArgmax(labels);
    int correctcount = getCorrectCount(preds, actuals);
    return static_cast<float>(correctcount) / preds.dimensions()[0];
}
